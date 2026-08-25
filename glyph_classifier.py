"""Glyph classifier for 3-top30 latent space.

9401 classes (glyph = script_id × 7026 + character_id), input = VAE latent (4×32×32).

Design goals:
  - Low param count (~5M) so it runs fast on 8GB laptop GPU.
  - GroupNorm (not BatchNorm) → stable on small batches, transfers to eval.
  - SiLU activations → smooth gradients, usable as an auxiliary loss.
  - Penultimate 512-d embedding → usable as a feature-matching loss in diffusion.
  - Label smoothing 0.1 → soft targets, better gradient landscape.
  - Outputs both logits (9401) and embedding (512) for flexibility.

Usage:
  model = GlyphLatentClassifier(num_classes=9401, latent_channels=4)
  logits, embed = model(latent)         # training
  logits, embed = model(latent, return_embed=True)
  logits = model(latent, return_embed=False)  # inference (cheaper)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv3x3 → GroupNorm → SiLU.  Downsample with stride=2 when down=True.

    Residual connection added when cin==cout and not downsampling (identity).
    """
    def __init__(self, cin, cout, down=False, groups=8, residual=False):
        super().__init__()
        stride = 2 if down else 1
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        g = min(groups, cout)
        while cout % g != 0:
            g -= 1
        self.norm = nn.GroupNorm(g, cout)
        self.act = nn.SiLU(inplace=True)
        self.residual = residual and not down and cin == cout

    def forward(self, x):
        h = self.act(self.norm(self.conv(x)))
        if self.residual:
            return h + x
        return h


class GlyphLatentClassifier(nn.Module):
    """Compact CNN classifier on VAE latents.

    Input:  (B, 4, 32, 32)  — VAE latent (already scaled by vae_scaling_factor)
    Output: logits (B, num_classes), optionally embedding (B, embed_dim)

    Architecture (4×32×32 → embed_dim):
        conv 4→64  (32×32)
        conv 64→128  down (16×16)
        conv 128→256  down (8×8) + res
        conv 256→512  down (4×4) + res
        adaptive pool → 512
        dropout → fc 512→embed_dim (embedding)
        dropout → fc embed_dim→num_classes (classifier)

    Regularization:
      - Dropout after pool and in classifier head (configurable)
      - Optional latent noise injection (additive Gaussian) during training
    """
    def __init__(self, num_classes=9401, latent_channels=4, embed_dim=512,
                 dropout=0.3, latent_noise_std=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.latent_noise_std = latent_noise_std
        self.features = nn.Sequential(
            ConvBlock(latent_channels, 64, down=False),   # 32×32
            ConvBlock(64, 128, down=True),                 # 16×16
            ConvBlock(128, 256, down=True),                # 8×8
            ConvBlock(256, 256, down=False, residual=True),# 8×8 res
            ConvBlock(256, 512, down=True),                # 4×4
            ConvBlock(512, 512, down=False, residual=True),# 4×4 res
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed_head = nn.Linear(512, embed_dim)
        self.class_head = nn.Linear(embed_dim, num_classes)
        if dropout > 0:
            self.drop1 = nn.Dropout(dropout)
            self.drop2 = nn.Dropout(dropout)
        else:
            self.drop1 = nn.Identity()
            self.drop2 = nn.Identity()

    def forward(self, x, return_embed=True):
        if self.training and self.latent_noise_std > 0:
            x = x + torch.randn_like(x) * self.latent_noise_std
        h = self.features(x)            # (B, 512, 4, 4)
        h = self.pool(h).flatten(1)      # (B, 512)
        h = self.drop1(h)
        e = self.embed_head(h)           # (B, embed_dim)
        e = self.drop2(e)
        logits = self.class_head(e)     # (B, num_classes)
        if return_embed:
            return logits, e
        return logits


if __name__ == "__main__":
    from models import DiT_2Cond_models  # just to verify import path
    m = GlyphLatentClassifier(9401, 4)
    n = sum(p.numel() for p in m.parameters())
    print(f"params: {n/1e6:.2f}M")
    x = torch.randn(4, 4, 32, 32)
    logits, emb = m(x)
    print(f"logits: {logits.shape}, embed: {emb.shape}")
    logits_only = m(x, return_embed=False)
    print(f"logits only: {logits_only.shape}")
