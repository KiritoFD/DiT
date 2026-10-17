"""Conditioning modules for Flow Matching U-Net (FM-UNet).

Provides:
- MultiScaleStdEncoder: Multi-scale spatial convolutional encoder for STD-soft [B, 1, 64, 64]
  emitting features at 32x32, 16x16, and 8x8.
- TimestepEmbedder: Sinusoidal timestep embedding + 2-layer MLP.
- WriterScriptCondition: ID embeddings (with optional pretrained table init),
  dropout masks, and fusion into global condition vector c.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedder followed by a 2-layer MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings.
        
        Args:
            t: 1D Tensor of timesteps (N,)
            dim: embedding dimension
            max_period: minimum frequency controls
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class MultiScaleStdEncoder(nn.Module):
    """Multi-scale spatial encoder for STD-soft condition.

    Input:
        g: [B, 1, 64, 64] float tensor in [0, 1] or soft continuous mass.

    Outputs:
        std_32: [B, 128, 32, 32]
        std_16: [B, 256, 16, 16]
        std_8:  [B, 384, 8, 8]
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 128,
        channel_mult: tuple = (1, 2, 3),
    ):
        super().__init__()
        c32 = base_channels * channel_mult[0]   # 128
        c16 = base_channels * channel_mult[1]   # 256
        c8  = base_channels * channel_mult[2]   # 384

        # Stem: 64x64 -> 32x32
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c32 // 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, c32 // 2),
            nn.SiLU(),
            nn.Conv2d(c32 // 2, c32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, c32),
            nn.SiLU(),
        )

        # Down 1: 32x32 -> 16x16
        self.down1 = nn.Sequential(
            nn.Conv2d(c32, c16, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, c16),
            nn.SiLU(),
            nn.Conv2d(c16, c16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, c16),
            nn.SiLU(),
        )

        # Down 2: 16x16 -> 8x8
        self.down2 = nn.Sequential(
            nn.Conv2d(c16, c8, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, c8),
            nn.SiLU(),
            nn.Conv2d(c8, c8, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, c8),
            nn.SiLU(),
        )

        # Learnable scale gates for each level (zero-init or small init for gentle injection)
        self.scale_32 = nn.Parameter(torch.zeros(1))
        self.scale_16 = nn.Parameter(torch.zeros(1))
        self.scale_8  = nn.Parameter(torch.zeros(1))

    def forward(self, g: torch.Tensor):
        if g is None:
            return None, None, None

        if g.dim() == 3:
            g = g.unsqueeze(1)
        elif g.dim() == 4 and g.shape[1] != 1:
            # Handle multi-channel g by taking mean or channel 0
            g = g[:, :1]

        f32 = self.stem(g)
        f16 = self.down1(f32)
        f8  = self.down2(f16)

        return (
            f32 * self.scale_32,
            f16 * self.scale_16,
            f8  * self.scale_8,
        )


class WriterScriptCondition(nn.Module):
    """Conditioning embedder for Writer ID, Script ID, and Timestep.

    Maintains:
    - Independent linear projections for Writer and Script.
    - Standard null class token at index `num_classes` for classifier-free guidance.
    - Dropping logic matching DiT training protocol.
    - Fuses t_emb, P_w(e_w), and P_s(e_s) via an MLP into a single global condition vector c.
    """

    def __init__(
        self,
        cond_dim: int = 512,
        num_calligraphers: int = 10,
        callig_embed_dim: int = 32,
        num_script_classes: int = 3,
        script_embed_dim: int = 16,
        time_freq_dim: int = 256,
        cond_drop_callig_prob: float = 0.08,
        cond_drop_script_prob: float = 0.08,
        cond_drop_all_prob: float = 0.05,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.num_calligraphers = num_calligraphers
        self.num_script_classes = num_script_classes
        self.cond_drop_callig_prob = cond_drop_callig_prob
        self.cond_drop_script_prob = cond_drop_script_prob
        self.cond_drop_all_prob = cond_drop_all_prob

        # Timestep embedder
        self.t_embedder = TimestepEmbedder(cond_dim, frequency_embedding_size=time_freq_dim)

        # Writer embedding (+1 for null class)
        self.y_callig_embedder = nn.Embedding(num_calligraphers + 1, callig_embed_dim)
        self.proj_callig = nn.Sequential(
            nn.Linear(callig_embed_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Script embedding (+1 for null class)
        self.y_script_embedder = nn.Embedding(num_script_classes + 1, script_embed_dim)
        self.proj_script = nn.Sequential(
            nn.Linear(script_embed_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Fusion MLP
        self.fuse_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def load_pretrained_tables(self, callig_table_path: str = None, font_table_path: str = None):
        """Optionally load pretrained numpy embedding tables."""
        import numpy as np
        if callig_table_path:
            tbl = np.load(callig_table_path)
            with torch.no_grad():
                self.y_callig_embedder.weight[:len(tbl)].copy_(torch.from_numpy(tbl))
        if font_table_path:
            tbl = np.load(font_table_path)
            with torch.no_grad():
                self.y_script_embedder.weight[:len(tbl)].copy_(torch.from_numpy(tbl))

    def _apply_dropout(self, y_callig: torch.Tensor, y_script: torch.Tensor, training: bool):
        """Apply condition dropout during training."""
        if not training:
            return y_callig, y_script

        N = y_callig.shape[0]
        device = y_callig.device

        # Null tokens
        null_callig = torch.full_like(y_callig, self.num_calligraphers)
        null_script = torch.full_like(y_script, self.num_script_classes) if y_script is not None else None

        # 1. Drop all
        if self.cond_drop_all_prob > 0:
            mask_all = torch.rand(N, device=device) < self.cond_drop_all_prob
            y_callig = torch.where(mask_all, null_callig, y_callig)
            if y_script is not None:
                y_script = torch.where(mask_all, null_script, y_script)

        # 2. Drop callig
        if self.cond_drop_callig_prob > 0:
            mask_c = torch.rand(N, device=device) < self.cond_drop_callig_prob
            y_callig = torch.where(mask_c, null_callig, y_callig)

        # 3. Drop script
        if self.cond_drop_script_prob > 0 and y_script is not None:
            mask_s = torch.rand(N, device=device) < self.cond_drop_script_prob
            y_script = torch.where(mask_s, null_script, y_script)

        return y_callig, y_script

    def forward(
        self,
        t: torch.Tensor,
        y_callig: torch.Tensor,
        y_script: torch.Tensor = None,
        force_drop_all: bool = False,
    ) -> torch.Tensor:
        """Fuse timestep, writer, and script into global condition vector c.

        Args:
            t: (N,) tensor scaled by 1000.0 (or normalized FM t)
            y_callig: (N,) calligrapher IDs
            y_script: (N,) script IDs (optional)
            force_drop_all: if True, all conditions set to null tokens (for CFG)
        """
        N = t.shape[0]
        device = t.device

        if force_drop_all:
            yc = torch.full_like(y_callig, self.num_calligraphers)
            ys = torch.full((N,), self.num_script_classes, dtype=torch.long, device=device)
        else:
            if y_script is None:
                ys = torch.zeros(N, dtype=torch.long, device=device)
            else:
                ys = y_script
            yc, ys = self._apply_dropout(y_callig, ys, self.training)

        # 1. Timestep embedding
        t_emb = self.t_embedder(t)

        # 2. Writer embedding & projection
        e_callig = self.y_callig_embedder(yc)
        p_callig = self.proj_callig(e_callig)

        # 3. Script embedding & projection
        e_script = self.y_script_embedder(ys)
        p_script = self.proj_script(e_script)

        # 4. Independent sum + fusion MLP
        fused = self.fuse_mlp(t_emb + p_callig + p_script)
        return fused
