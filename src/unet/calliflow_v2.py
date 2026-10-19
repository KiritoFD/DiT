"""CalliFlow-v2 — Deep-Thin Hourglass CNN U-Net with Dual Velocity Field.

Core Architectural Innovations:
  1. Base Velocity Field: v_base(z_t, t, g, s)
     - Governs character content, stroke connectivity, structural validity.
     - Strictly conditioned on Time, STD Soft Skeleton, and Script.
     - Never receives Writer ID (mathematically shielded to model E[v* | z_t, t, g, s]).
  2. Multi-scale Writer Residual Field: Delta v_writer(z_t, t, g, s, w)
     - Governs calligrapher structure shifts, stroke trajectory, and ink features.
     - Multi-scale extraction across 8^2 (structure), 16^2 (strokes), 32^2 (brush/ink).
     - DiC-style conditional channel gates driven by Writer embedding.
     - Output head zero-initialized for stable departure from base field.
  3. Single-Pass CFG Inference:
     - v_CFG = v_base + alpha * Delta v_writer in a single forward pass.
  4. Dual Flow Matching Supervision:
     - L = ||v_base + Delta v_writer - v*||^2 + 0.5 * ||v_base - v*||^2.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Attention, norm, timestep_embedding, get_activation
from .variants.modulation import bounded_gain, zero_linear


def zero_conv2d(in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1) -> nn.Conv2d:
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv


class ThinResBlock(nn.Module):
    """Lightweight ResBlock with AdaGN conditioning and optional residual gating."""
    def __init__(self, cin: int, cout: int, cond_dim: int = 512, act_fn: str = "gelu"):
        super().__init__()
        self.act = get_activation(act_fn)
        self.norm1 = norm(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2 = norm(cout)
        self.film = nn.Linear(cond_dim, 2 * cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.shortcut = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.residual_gate = zero_linear(cond_dim, cout)
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        cond_act = self.act(c)
        scale, shift = self.film(cond_act).to(h.dtype).chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        residual = self.conv2(self.act(h))
        gain = bounded_gain(self.residual_gate(cond_act).to(residual.dtype))
        residual = residual * gain[:, :, None, None]
        return self.shortcut(x) + residual


class WriterFieldBranch(nn.Module):
    """Multi-scale Writer residual field operating on shared U-Net features."""
    def __init__(
        self,
        base_channels: Tuple[int, int, int] = (128, 256, 384),
        writer_channels: Tuple[int, int, int] = (64, 96, 128),
        num_writers: int = 10,
        cond_dim: int = 256,
        act_fn: str = "gelu",
    ):
        super().__init__()
        self.act = get_activation(act_fn)
        act_layer = nn.GELU if act_fn == "gelu" else nn.SiLU
        self.num_writers = num_writers
        self.writer = nn.Embedding(num_writers + 1, 32)
        
        # Per-scale writer condition projection (8^2, 16^2, 32^2)
        self.writer_proj = nn.ModuleList([
            nn.Sequential(nn.Linear(32, cond_dim), act_layer(), nn.Linear(cond_dim, cond_dim))
            for _ in range(3)
        ])
        
        # Scale 2 (8x8): Overall structure & composition
        self.proj_in_2 = nn.Conv2d(base_channels[2], writer_channels[2], 1)
        self.gate_2 = zero_linear(cond_dim, writer_channels[2])
        self.blocks_2 = nn.ModuleList([
            ThinResBlock(writer_channels[2], writer_channels[2], cond_dim=cond_dim, act_fn=act_fn)
            for _ in range(2)
        ])
        self.up_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(writer_channels[2], writer_channels[1], 3, padding=1)
        )
        
        # Scale 1 (16x16): Stroke trajectories & turns
        self.proj_in_1 = nn.Conv2d(base_channels[1], writer_channels[1], 1)
        self.gate_1 = zero_linear(cond_dim, writer_channels[1])
        self.fuse_1 = nn.Conv2d(writer_channels[1] * 2, writer_channels[1], 3, padding=1)
        self.blocks_1 = nn.ModuleList([
            ThinResBlock(writer_channels[1], writer_channels[1], cond_dim=cond_dim, act_fn=act_fn)
            for _ in range(2)
        ])
        self.up_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(writer_channels[1], writer_channels[0], 3, padding=1)
        )
        
        # Scale 0 (32x32): Local brushwork, line weight & ink texture
        self.proj_in_0 = nn.Conv2d(base_channels[0], writer_channels[0], 1)
        self.gate_0 = zero_linear(cond_dim, writer_channels[0])
        self.fuse_0 = nn.Conv2d(writer_channels[0] * 2, writer_channels[0], 3, padding=1)
        self.blocks_0 = nn.ModuleList([
            ThinResBlock(writer_channels[0], writer_channels[0], cond_dim=cond_dim, act_fn=act_fn)
            for _ in range(2)
        ])
        
        # Zero-initialized velocity output head: Delta v_writer starts strictly at 0
        self.head = nn.Sequential(
            norm(writer_channels[0]),
            act_layer(),
            zero_conv2d(writer_channels[0], 4, 3, padding=1)
        )

    def forward(
        self,
        feat_dec_2: torch.Tensor,
        feat_dec_1: torch.Tensor,
        feat_dec_0: torch.Tensor,
        y_callig: torch.Tensor,
    ) -> torch.Tensor:
        w_emb = self.writer(y_callig)
        c2 = self.writer_proj[2](w_emb)
        c1 = self.writer_proj[1](w_emb)
        c0 = self.writer_proj[0](w_emb)
        
        # Scale 2 (8x8)
        h2 = self.proj_in_2(feat_dec_2)
        h2 = h2 * bounded_gain(self.gate_2(self.act(c2)))[:, :, None, None]
        for blk in self.blocks_2:
            h2 = blk(h2, c2)
        h2_up = self.up_2(h2)
        
        # Scale 1 (16x16)
        h1 = self.proj_in_1(feat_dec_1)
        h1 = self.fuse_1(torch.cat([h1, h2_up], dim=1))
        h1 = h1 * bounded_gain(self.gate_1(self.act(c1)))[:, :, None, None]
        for blk in self.blocks_1:
            h1 = blk(h1, c1)
        h1_up = self.up_1(h1)
        
        # Scale 0 (32x32)
        h0 = self.proj_in_0(feat_dec_0)
        h0 = self.fuse_0(torch.cat([h0, h1_up], dim=1))
        h0 = h0 * bounded_gain(self.gate_0(self.act(c0)))[:, :, None, None]
        for blk in self.blocks_0:
            h0 = blk(h0, c0)
            
        delta_v = self.head(h0)
        return delta_v


class CalliFlowV2(nn.Module):
    """Deep-Thin DiC Hourglass U-Net with Dual Velocity Fields.
    
    Channels: (128, 256, 384)
    Encoder ResBlocks per scale: 4
    Decoder ResBlocks per scale: 4
    Middle ResBlocks: 3 + SDPA Attention
    STD Injection: Multi-scale 32^2, 16^2, 8^2 with per-scale gates
    Base Velocity: v_base(z_t, t, g, s)
    Writer Residual: Delta v_writer(z_t, t, g, s, w)
    """
    std_soft_cond = True
    use_glyph_cond = True
    in_channels = 4
    out_channels = 4

    def __init__(
        self,
        channels: Tuple[int, int, int] = (128, 256, 384),
        writer_channels: Tuple[int, int, int] = (64, 96, 128),
        num_res_blocks: int = 4,
        middle_res_blocks: int = 3,
        num_writers: int = 10,
        num_scripts: int = 3,
        cond_dim: int = 512,
        act_fn: str = "gelu",
        drop_all: float = 0.05,
        drop_script: float = 0.08,
    ):
        super().__init__()
        self.channels = tuple(channels)
        self.num_res_blocks = num_res_blocks
        self.num_writers = num_writers
        self.num_scripts = num_scripts
        self.cond_dim = cond_dim
        self.act_fn = act_fn
        self.act = get_activation(act_fn)
        act_layer = nn.GELU if act_fn == "gelu" else nn.SiLU
        self.drop_all = drop_all
        self.drop_script = drop_script
        
        # Base condition embeddings: ONLY time and script!
        self.script = nn.Embedding(num_scripts + 1, 16)
        self.time_proj = nn.ModuleList([nn.Linear(128, cond_dim) for _ in range(3)])
        self.script_proj = nn.ModuleList([nn.Linear(16, cond_dim) for _ in range(3)])
        self.stage_cond = nn.ModuleList([
            nn.Sequential(act_layer(), nn.Linear(cond_dim, cond_dim))
            for _ in range(3)
        ])
        
        # Multi-scale STD spatial conditioning
        self.std_stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),
            act_layer(),
            nn.Conv2d(64, 64, 3, padding=1),
        )
        self.std_down = nn.ModuleList([
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.Conv2d(128, 192, 3, stride=2, padding=1),
        ])
        self.std_proj = nn.ModuleList([
            nn.Conv2d(s, c, 1) for s, c in zip((64, 128, 192), self.channels)
        ])
        self.std_gates = nn.ParameterList([nn.Parameter(torch.zeros(())) for _ in range(3)])
        self.skip_gates = nn.ParameterList([nn.Parameter(torch.zeros(())) for _ in range(3)])
        
        # Deep-Thin Base Backbone
        self.input = nn.Conv2d(4, self.channels[0], 3, padding=1)
        
        # Encoder: 4 ResBlocks per scale
        self.enc_stages = nn.ModuleList()
        self.enc_attns = nn.ModuleList()
        self.enc_downs = nn.ModuleList()
        for i, c in enumerate(self.channels):
            stage = nn.ModuleList([
                ThinResBlock(c, c, cond_dim=cond_dim, act_fn=act_fn)
                for _ in range(num_res_blocks)
            ])
            self.enc_stages.append(stage)
            self.enc_attns.append(Attention(c) if i > 0 else nn.Identity())
            if i < 2:
                self.enc_downs.append(nn.Conv2d(self.channels[i], self.channels[i + 1], 3, stride=2, padding=1))
                
        # Middle: 3 ResBlocks + Attention
        self.mid_blocks = nn.ModuleList([
            ThinResBlock(self.channels[-1], self.channels[-1], cond_dim=cond_dim, act_fn=act_fn)
            for _ in range(middle_res_blocks)
        ])
        self.mid_attn = Attention(self.channels[-1])
        
        # Decoder: 4 ResBlocks per scale
        self.dec_stages = nn.ModuleList()
        self.dec_attns = nn.ModuleList()
        self.dec_ups = nn.ModuleList()
        for i, c in enumerate(self.channels):
            # i=2: 384*2 -> 384, i=1: 256*2 -> 256, i=0: 128*2 -> 128
            stage = nn.ModuleList([
                ThinResBlock(c * 2 if j == 0 else c, c, cond_dim=cond_dim, act_fn=act_fn)
                for j in range(num_res_blocks)
            ])
            self.dec_stages.append(stage)
            self.dec_attns.append(Attention(c) if i > 0 else nn.Identity())
            if i > 0:
                self.dec_ups.append(nn.Conv2d(self.channels[i], self.channels[i - 1], 3, padding=1))
                
        # Base Velocity Head: zero-initialized
        self.base_head = nn.Sequential(
            norm(self.channels[0]),
            act_layer(),
            zero_conv2d(self.channels[0], 4, 3, padding=1)
        )
        
        # Dedicated Multi-scale Writer Branch
        self.writer_branch = WriterFieldBranch(
            base_channels=self.channels,
            writer_channels=writer_channels,
            num_writers=num_writers,
            cond_dim=256,
            act_fn=act_fn,
        )

    def _compute_base_conditions(self, t_emb: torch.Tensor, script_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(
            cond_fn(tp(t_emb) + sp(script_emb))
            for tp, sp, cond_fn in zip(self.time_proj, self.script_proj, self.stage_cond)
        )

    def _forward_backbone(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        g: torch.Tensor,
        y_script: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Base backbone forward: Strictly NO Writer ID is passed."""
        if self.training:
            all_drop = torch.rand(len(x), device=x.device) < self.drop_all
            sd = all_drop | (torch.rand(len(x), device=x.device) < self.drop_script)
            y_script = torch.where(sd, self.num_scripts, y_script)
            
        script_emb = self.script(y_script)
        t_emb = timestep_embedding(t)
        c0, c1, c2 = self._compute_base_conditions(t_emb, script_emb)
        conds = (c0, c1, c2)
        
        # Multi-scale STD Stem & Downsampling
        glyph = self.std_stem(g)
        std_feats = [glyph]
        for down_layer in self.std_down:
            glyph = self.act(down_layer(glyph))
            std_feats.append(glyph)
            
        # Encoder
        h = self.input(x)
        skips = []
        for i, stage in enumerate(self.enc_stages):
            proj_std = self.std_proj[i](std_feats[i]) * bounded_gain(self.std_gates[i])
            h = h + proj_std
            for blk in stage:
                h = blk(h, conds[i])
            h = self.enc_attns[i](h)
            skips.append(h)
            if i < 2:
                h = self.enc_downs[i](h)
                
        # Middle
        for blk in self.mid_blocks:
            h = blk(h, conds[2])
        h = self.mid_attn(h)
        
        # Decoder (from 8x8 up to 32x32)
        dec_feats = [None, None, None]
        # i = 2 (8x8)
        skip_2 = skips[2] * bounded_gain(self.skip_gates[2])
        h = torch.cat([h, skip_2], dim=1)
        for blk in self.dec_stages[2]:
            h = blk(h, conds[2])
        h = self.dec_attns[2](h)
        dec_feats[2] = h
        
        # i = 1 (16x16)
        h_up_1 = self.dec_ups[1](F.interpolate(h, scale_factor=2, mode="nearest"))
        skip_1 = skips[1] * bounded_gain(self.skip_gates[1])
        h = torch.cat([h_up_1, skip_1], dim=1)
        for blk in self.dec_stages[1]:
            h = blk(h, conds[1])
        h = self.dec_attns[1](h)
        dec_feats[1] = h
        
        # i = 0 (32x32)
        h_up_0 = self.dec_ups[0](F.interpolate(h, scale_factor=2, mode="nearest"))
        skip_0 = skips[0] * bounded_gain(self.skip_gates[0])
        h = torch.cat([h_up_0, skip_0], dim=1)
        for blk in self.dec_stages[0]:
            h = blk(h, conds[0])
        h = self.dec_attns[0](h)
        dec_feats[0] = h
        
        v_base = self.base_head(h)
        return v_base, (dec_feats[0], dec_feats[1], dec_feats[2])

    def forward_both(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y_callig: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        y_script: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute both v_base and Delta v_writer simultaneously in one pass."""
        if g is None:
            g = x.new_zeros((len(x), 1, 64, 64))
        if y_script is None:
            raise ValueError("Script condition required")
            
        v_base, (f0, f1, f2) = self._forward_backbone(x, t, g, y_script)
        delta_v = self.writer_branch(f2, f1, f0, y_callig)
        return v_base, delta_v

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y_callig: torch.Tensor,
        y_char: Optional[torch.Tensor] = None,
        g: Optional[torch.Tensor] = None,
        y_script: Optional[torch.Tensor] = None,
        **unused,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        v_base, delta_v = self.forward_both(x, t, y_callig, g=g, y_script=y_script)
        v_full = v_base + delta_v
        
        if self.training:
            # Dual FM training objective:
            # L = ||v_full - v*||^2 + w_base * ||v_base - v*||^2
            aux = {
                "v_base": v_base,
                "delta_v": delta_v,
                "w_base": 0.5,
            }
            return v_full, aux
            
        return v_full

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y_callig: torch.Tensor,
        y_char: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        g: Optional[torch.Tensor] = None,
        y_script: Optional[torch.Tensor] = None,
        **unused,
    ) -> torch.Tensor:
        """Instant Single-Pass CFG: v_CFG = v_base + alpha * Delta v_writer."""
        v_base, delta_v = self.forward_both(x, t, y_callig, g=g, y_script=y_script)
        if cfg_scale == 1.0:
            return v_base + delta_v
        return v_base + float(cfg_scale) * delta_v
