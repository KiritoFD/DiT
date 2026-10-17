"""Flow Matching U-Net (FM-UNet) for Chinese Calligraphy Synthesis.

Architecture follows standard SD-style multi-scale U-Net with Flow Matching:
- Multiscale: 32x32 (128) -> 16x16 (256) -> 8x8 (384) -> Mid (384) -> 8x8 (384) -> 16x16 (256) -> 32x32 (128)
- Normalization & Activation: GroupNorm + SiLU (no BatchNorm)
- Attention: Multi-Head Self-Attention using PyTorch SDPA (F.scaled_dot_product_attention)
- ResBlock: Residual convolutions with FiLM conditioning from Writer, Script, and Timestep
- Content: Multi-scale spatial STD condition injection into 32x32, 16x16, and 8x8 features
- Fast sampling: Condition caching for Heun ODE integration; cfg_scale == 1.0 fast single-pass
- Zero-initialized output convolution for smooth training start.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.fm_unet_condition import MultiScaleStdEncoder, WriterScriptCondition


# -----------------------------------------------------------------------------
# Primitive Building Blocks
# -----------------------------------------------------------------------------

class Upsample2d(nn.Module):
    """Interpolate + Conv2d upsampler."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Downsample2d(nn.Module):
    """Strided Conv2d downsampler."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResBlock(nn.Module):
    """Residual Block with FiLM conditioning.

    h_norm = GroupNorm(h)
    h_film = h_norm * (1 + gamma(c)) + beta(c)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        num_groups: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Conv 1
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # FiLM projection: maps c -> (gamma, beta) for out_channels
        self.film_proj = nn.Linear(cond_dim, out_channels * 2)

        # Conv 2
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Residual shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        # Initialize film projection to zero so modulation starts as identity
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, H, W]
            cond: [B, cond_dim] global conditioning vector
        """
        h = self.conv1(self.act1(self.norm1(x)))

        # FiLM modulation on norm2
        h = self.norm2(h)
        film = self.film_proj(cond)[:, :, None, None]
        gamma, beta = film.chunk(2, dim=1)
        h = h * (1.0 + gamma) + beta

        h = self.conv2(self.act2(h))
        return self.shortcut(x) + h


class AttentionBlock(nn.Module):
    """Multi-Head Self-Attention block using PyTorch SDPA."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        num_groups: int = 32,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, f"channels {channels} must divide num_heads {num_heads}"

        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)  # [B, 3*C, H, W]
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape to [B, num_heads, H*W, head_dim]
        q = q.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        k = k.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        v = v.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)

        # SDPA
        attn_out = F.scaled_dot_product_attention(q, k, v)  # [B, num_heads, H*W, head_dim]

        # Reshape back to [B, C, H, W]
        attn_out = attn_out.transpose(2, 3).contiguous().view(B, C, H, W)
        return x + self.proj_out(attn_out)


# -----------------------------------------------------------------------------
# FM-UNet Model Architecture
# -----------------------------------------------------------------------------

class FMUNet(nn.Module):
    """Flow Matching U-Net Backbone for Calligraphy Latents."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 3),  # (128, 256, 384)
        cond_dim: int = 512,
        num_calligraphers: int = 10,
        callig_embed_dim: int = 32,
        num_script_classes: int = 3,
        script_embed_dim: int = 16,
        time_freq_dim: int = 256,
        num_res_blocks: int = 2,
        attn_resolutions: Tuple[int, ...] = (16, 8),
        num_groups: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.cond_dim = cond_dim
        self.num_calligraphers = num_calligraphers
        self.num_script_classes = num_script_classes

        # 1. Conditioning modules
        self.cond_module = WriterScriptCondition(
            cond_dim=cond_dim,
            num_calligraphers=num_calligraphers,
            callig_embed_dim=callig_embed_dim,
            num_script_classes=num_script_classes,
            script_embed_dim=script_embed_dim,
            time_freq_dim=time_freq_dim,
        )
        self.std_encoder = MultiScaleStdEncoder(
            in_channels=1,
            base_channels=base_channels,
            channel_mult=channel_mult,
        )

        c1 = base_channels * channel_mult[0]  # 128
        c2 = base_channels * channel_mult[1]  # 256
        c3 = base_channels * channel_mult[2]  # 384

        # 2. Input Conv
        self.conv_in = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)

        # 3. Encoder Stage 1: 32x32, 128 channels
        self.enc1_b1 = ResBlock(c1, c1, cond_dim, num_groups)
        self.enc1_b2 = ResBlock(c1, c1, cond_dim, num_groups)
        self.down1 = Downsample2d(c1, c2)  # 32x32 -> 16x16

        # 4. Encoder Stage 2: 16x16, 256 channels + Attention
        self.enc2_b1 = ResBlock(c2, c2, cond_dim, num_groups)
        self.enc2_b2 = ResBlock(c2, c2, cond_dim, num_groups)
        self.enc2_attn = AttentionBlock(c2, num_heads=8, num_groups=num_groups)
        self.down2 = Downsample2d(c2, c3)  # 16x16 -> 8x8

        # 5. Encoder Stage 3: 8x8, 384 channels + Attention
        self.enc3_b1 = ResBlock(c3, c3, cond_dim, num_groups)
        self.enc3_b2 = ResBlock(c3, c3, cond_dim, num_groups)
        self.enc3_attn = AttentionBlock(c3, num_heads=8, num_groups=num_groups)

        # 6. Middle Stage: 8x8, 384 channels
        self.mid_b1 = ResBlock(c3, c3, cond_dim, num_groups)
        self.mid_attn = AttentionBlock(c3, num_heads=8, num_groups=num_groups)
        self.mid_b2 = ResBlock(c3, c3, cond_dim, num_groups)

        # 7. Decoder Stage 3: 8x8, 384 channels (fuse skip from enc3: c3 + c3 -> c3)
        self.dec3_b = ResBlock(c3 + c3, c3, cond_dim, num_groups)
        self.up3 = Upsample2d(c3)  # 8x8 -> 16x16

        # 8. Decoder Stage 2: 16x16, 256 channels (fuse skip from enc2: c3 + c2 -> c2)
        self.dec2_b = ResBlock(c3 + c2, c2, cond_dim, num_groups)
        self.up2 = Upsample2d(c2)  # 16x16 -> 32x32

        # 9. Decoder Stage 1: 32x32, 128 channels (fuse skip from enc1: c2 + c1 -> c1)
        self.dec1_b = ResBlock(c2 + c1, c1, cond_dim, num_groups)

        # 10. Output Head: GroupNorm -> SiLU -> Conv3x3 (zero initialized)
        self.out_norm = nn.GroupNorm(num_groups, c1)
        self.out_act = nn.SiLU()
        self.conv_out = nn.Conv2d(c1, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    @property
    def glyph_embedder(self):
        """Compatibility property for std_shape health telemetry."""
        return self.std_encoder.stem

    def effective_glyph_scale(self):
        """Compatibility helper for scale monitoring."""
        return self.std_encoder.scale_32

    def prepare_condition(
        self,
        g: Optional[torch.Tensor],
        y_callig: torch.Tensor,
        y_script: Optional[torch.Tensor] = None,
        force_drop_all: bool = False,
    ) -> Dict[str, Any]:
        """Pre-encode static conditions (STD multi-scale and Writer/Script embeddings).
        
        Allows caching during ODE sampling to eliminate redundant 100x STD conv runs.
        """
        # Encode multi-scale STD once
        std_32, std_16, std_8 = self.std_encoder(g) if g is not None else (None, None, None)
        return {
            "std_features": (std_32, std_16, std_8),
            "y_callig": y_callig,
            "y_script": y_script,
            "force_drop_all": force_drop_all,
        }

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y_callig: torch.Tensor,
        y_char: Optional[torch.Tensor] = None,
        g: Optional[torch.Tensor] = None,
        y_script: Optional[torch.Tensor] = None,
        cond_cache: Optional[Dict[str, Any]] = None,
        force_drop_all: bool = False,
    ) -> torch.Tensor:
        """Forward pass predicting velocity v in [B, 4, 32, 32].

        Args:
            x: [B, 4, 32, 32] noisy latent at time t
            t: [B] timesteps (scaled by 1000.0)
            y_callig: [B] calligrapher ID
            y_char: ignored (legacy compatibility)
            g: [B, 1, 64, 64] STD-soft condition
            y_script: [B] script ID
            cond_cache: optional precomputed dict from prepare_condition()
            force_drop_all: if True, drop all conditions to null tokens (for CFG)
        """
        # 1. Condition computation (or reuse cache)
        if cond_cache is not None:
            std_32, std_16, std_8 = cond_cache["std_features"]
            yc = cond_cache["y_callig"]
            ys = cond_cache["y_script"]
            fda = cond_cache["force_drop_all"] or force_drop_all
            cond_vec = self.cond_module(t, yc, ys, force_drop_all=fda)
        else:
            std_32, std_16, std_8 = self.std_encoder(g) if g is not None else (None, None, None)
            cond_vec = self.cond_module(t, y_callig, y_script, force_drop_all=force_drop_all)

        # 2. Input stem
        h = self.conv_in(x)  # [B, 128, 32, 32]

        # 3. Stage 1 (32x32)
        h = self.enc1_b1(h, cond_vec)
        if std_32 is not None:
            h = h + std_32
        h = self.enc1_b2(h, cond_vec)
        skip1 = h            # [B, 128, 32, 32]
        h = self.down1(h)    # [B, 256, 16, 16]

        # 4. Stage 2 (16x16)
        h = self.enc2_b1(h, cond_vec)
        if std_16 is not None:
            h = h + std_16
        h = self.enc2_b2(h, cond_vec)
        h = self.enc2_attn(h)
        skip2 = h            # [B, 256, 16, 16]
        h = self.down2(h)    # [B, 384, 8, 8]

        # 5. Stage 3 (8x8)
        h = self.enc3_b1(h, cond_vec)
        h = self.enc3_b2(h, cond_vec)
        h = self.enc3_attn(h)
        skip3 = h            # [B, 384, 8, 8]

        # 6. Middle (8x8)
        h = self.mid_b1(h, cond_vec)
        if std_8 is not None:
            h = h + std_8
        h = self.mid_attn(h)
        h = self.mid_b2(h, cond_vec)  # [B, 384, 8, 8]

        # 7. Decoder 3 (8x8 -> 16x16)
        h = torch.cat([h, skip3], dim=1)  # [B, 384+384, 8, 8]
        h = self.dec3_b(h, cond_vec)     # [B, 384, 8, 8]
        h = self.up3(h)                   # [B, 384, 16, 16]

        # 8. Decoder 2 (16x16 -> 32x32)
        h = torch.cat([h, skip2], dim=1)  # [B, 384+256, 16, 16]
        h = self.dec2_b(h, cond_vec)     # [B, 256, 16, 16]
        h = self.up2(h)                   # [B, 256, 32, 32]

        # 9. Decoder 1 (32x32)
        h = torch.cat([h, skip1], dim=1)  # [B, 256+128, 32, 32]
        h = self.dec1_b(h, cond_vec)     # [B, 128, 32, 32]

        # 10. Output head
        out = self.conv_out(self.act2_head(self.out_norm(h)))
        return out

    def act2_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_act(x)

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y_callig: torch.Tensor,
        y_char: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        g: Optional[torch.Tensor] = None,
        y_script: Optional[torch.Tensor] = None,
        cond_cache: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Forward with classifier-free guidance.

        Optimized path: When cfg_scale == 1.0, directly evaluate single conditional pass!
        """
        if abs(cfg_scale - 1.0) < 1e-5:
            return self.forward(
                x, t, y_callig, y_char=y_char, g=g, y_script=y_script,
                cond_cache=cond_cache, force_drop_all=False,
            )

        # Full 2B batched CFG evaluation
        B = x.shape[0]
        x_double = torch.cat([x, x], dim=0)
        t_double = torch.cat([t, t], dim=0)
        y_callig_double = torch.cat([y_callig, y_callig], dim=0)
        y_script_double = torch.cat([y_script, y_script], dim=0) if y_script is not None else None
        g_double = torch.cat([g, g], dim=0) if g is not None else None

        # Cond branch
        v_cond = self.forward(
            x, t, y_callig, y_char=y_char, g=g, y_script=y_script,
            cond_cache=cond_cache, force_drop_all=False,
        )
        # Uncond branch
        v_uncond = self.forward(
            x, t, y_callig, y_char=y_char, g=g, y_script=y_script,
            cond_cache=cond_cache, force_drop_all=True,
        )

        return v_uncond + cfg_scale * (v_cond - v_uncond)
