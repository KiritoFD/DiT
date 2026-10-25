"""PixelWriter-v2: Multi-loss Siamese CNN with optional 1D/2D Haar Wavelet branches.

Refactored according to CJIG 2023 Multi-loss Siamese CNN calligraphy literature:
1. ResNet Residual Backbone (GroupNorm, 32 -> 96 -> 160 -> 256).
2. Multi-scale Channel Mean + Std Pooling (1024D).
3. Ablation-ready Haar Wavelet modules:
   - 'none': Pure D1 baseline
   - '1d': 1D Haar DWT on pooled 1024D representation (matching paper's global Haar)
   - '2d': 2D Spatial Haar DWT on ink intensity at 64x64 and 32x32 levels with learnable fusion
4. Cosine Classification Head with Same-Script Masking.
5. Mild data augmentation and background probe utility.
"""
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            nn.GroupNorm(8, cout),
            nn.SiLU(),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
            nn.GroupNorm(8, cout)
        )
        self.skip = nn.Conv2d(cin, cout, 1, stride, bias=False) if cin != cout or stride != 1 else nn.Identity()

    def forward(self, x):
        return F.silu(self.body(x) + self.skip(x))


def haar_dwt_2d(x):
    """2D Haar Discrete Wavelet Transform via tensor slicing.
    
    Decomposes [B, C, H, W] into four sub-bands:
    LL: Low-frequency approximation
    LH: Horizontal high-frequency (vertical edges)
    HL: Vertical high-frequency (horizontal edges)
    HH: Diagonal high-frequency (stroke corners / texture)
    """
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]
    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (-x00 - x01 + x10 + x11) * 0.5
    hl = (-x00 + x01 - x10 + x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return ll, lh, hl, hh


def haar_dwt_1d(v):
    """1D Haar Wavelet Transform on a 1D feature vector [B, D]."""
    v_even = v[:, 0::2]
    v_odd = v[:, 1::2]
    approx = (v_even + v_odd) * 0.7071067811865475
    detail = (v_even - v_odd) * 0.7071067811865475
    return torch.cat([approx, detail], dim=-1)


class Haar2DFusion(nn.Module):
    """Level-specific 2D Haar high-frequency projection and residual fusion."""
    def __init__(self, in_channels, out_channels, stride=2, hidden=32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, stride, 1, bias=False),
            nn.GroupNorm(min(4, hidden), hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, out_channels, 1, 1, 0, bias=False),
            nn.GroupNorm(8, out_channels)
        )
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, high_freq):
        return self.scale * self.proj(high_freq)



class PixelWriterV2(nn.Module):
    """Multi-loss Siamese CNN Writer Style Encoder (~3.82M params)."""
    def __init__(self, n_slots=23, haar_mode='2d', haar_dropout=0.1):
        super().__init__()
        assert haar_mode in ('none', '1d', '2d'), f"Unknown haar_mode: {haar_mode}"
        self.haar_mode = haar_mode
        self.haar_dropout = haar_dropout
        self.n_slots = n_slots

        # Standard CNN Backbone
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 5, 2, 2, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU()
        )
        self.stage1 = nn.Sequential(Block(32, 96, 2), Block(96, 96))
        self.stage2 = nn.Sequential(Block(96, 160, 2), Block(160, 160))
        self.stage3 = nn.Sequential(Block(160, 256, 2), Block(256, 256))

        # 2D Haar DWT Fusion Branches (Level 1: 64x64, Level 2: 32x32)
        if self.haar_mode == '2d':
            # LH, HL, HH concatenated = 3 channels
            self.haar_fusion_lvl1 = Haar2DFusion(in_channels=3, out_channels=96, hidden=32)
            self.haar_fusion_lvl2 = Haar2DFusion(in_channels=3, out_channels=160, hidden=48)
        else:
            self.haar_fusion_lvl1 = None
            self.haar_fusion_lvl2 = None

        # Projector MLP (1024 -> 384 -> 256)
        self.project = nn.Sequential(
            nn.LayerNorm(2 * (96 + 160 + 256)),
            nn.Linear(1024, 384),
            nn.SiLU(),
            nn.Linear(384, 256)
        )

        # Cosine Classification Head
        self.head = nn.Linear(256, n_slots, bias=False)

    def reset_head(self, n_slots, init_weights=None):
        """Dynamically reset the classification head for Stage A -> Stage B transition."""
        self.n_slots = n_slots
        dev = next(self.parameters()).device
        self.head = nn.Linear(256, n_slots, bias=False).to(dev)
        if init_weights is not None:
            with torch.no_grad():
                self.head.weight.copy_(init_weights)

    def extract_features(self, gray):
        """Extract multi-scale pooled features from grayscale images."""
        # Standard CNN forward
        x = self.stem(2 * gray - 1)  # [B, 32, 64, 64]

        # 2D Haar DWT branch
        if self.haar_mode == '2d':
            ink = 1.0 - gray  # Ink intensity in [0, 1]
            ll1, lh1, hl1, hh1 = haar_dwt_2d(ink)  # 64x64
            h1 = torch.cat([lh1, hl1, hh1], dim=1)
            # Optional Haar dropout to prevent overfitting on scan noise
            if self.training and self.haar_dropout > 0:
                mask1 = (torch.rand(len(gray), 1, 1, 1, device=gray.device) >= self.haar_dropout).float()
                h1 = h1 * mask1

            x = self.stage1(x)  # [B, 96, 64, 64]
            x = x + self.haar_fusion_lvl1(h1)

            ll2, lh2, hl2, hh2 = haar_dwt_2d(ll1)  # 32x32
            h2 = torch.cat([lh2, hl2, hh2], dim=1)
            if self.training and self.haar_dropout > 0:
                mask2 = (torch.rand(len(gray), 1, 1, 1, device=gray.device) >= self.haar_dropout).float()
                h2 = h2 * mask2

            p1 = [x.float().mean((2, 3)), x.float().var((2, 3), unbiased=False).add(1e-6).sqrt()]

            x = self.stage2(x)  # [B, 160, 32, 32]
            x = x + self.haar_fusion_lvl2(h2)
            p2 = [x.float().mean((2, 3)), x.float().var((2, 3), unbiased=False).add(1e-6).sqrt()]

            x = self.stage3(x)  # [B, 256, 16, 16]
            p3 = [x.float().mean((2, 3)), x.float().var((2, 3), unbiased=False).add(1e-6).sqrt()]
        else:
            x = self.stage1(x)
            p1 = [x.float().mean((2, 3)), x.float().var((2, 3), unbiased=False).add(1e-6).sqrt()]
            x = self.stage2(x)
            p2 = [x.float().mean((2, 3)), x.float().var((2, 3), unbiased=False).add(1e-6).sqrt()]
            x = self.stage3(x)
            p3 = [x.float().mean((2, 3)), x.float().var((2, 3), unbiased=False).add(1e-6).sqrt()]

        pooled = torch.cat(p1 + p2 + p3, dim=1)  # [B, 1024]

        # 1D Haar DWT branch on global vector (CJIG 2023 paper setting)
        if self.haar_mode == '1d':
            pooled = haar_dwt_1d(pooled)

        e = F.normalize(self.project(pooled), dim=-1)
        return e

    def forward(self, gray, script_mask=None):
        """Forward pass.
        
        Args:
            gray: Tensor [B, 1, H, W] in [0, 1].
            script_mask: Optional boolean or 0/1 Tensor [B, n_slots].
        """
        e = self.extract_features(gray)
        raw_logits = 16.0 * F.linear(e, F.normalize(self.head.weight, dim=-1))

        if script_mask is not None:
            masked_logits = torch.where(script_mask.bool(), raw_logits, raw_logits - 10000.0)
        else:
            masked_logits = raw_logits

        return {
            'embedding': e,
            'slot_logits': raw_logits,
            'masked_logits': masked_logits
        }


# Backward-compatible alias
WriterStylePixel = PixelWriterV2


def augment_gray(gray):
    """Mild affine, contrast/background/noise and blur; no morphology or flips."""
    b = len(gray)
    dev = gray.device
    angle = (torch.rand(b, device=dev) * 2 - 1) * 0.05236  # +/- 3 deg
    scale = 0.95 + 0.10 * torch.rand(b, device=dev)       # 0.95 - 1.05
    matrix = torch.zeros(b, 2, 3, device=dev)
    matrix[:, 0, 0] = angle.cos() * scale
    matrix[:, 0, 1] = -angle.sin() * scale
    matrix[:, 1, 0] = angle.sin() * scale
    matrix[:, 1, 1] = angle.cos() * scale
    matrix[:, :, 2] = (torch.rand(b, 2, device=dev) * 2 - 1) * 0.04  # translation +/- 4%
    grid = F.affine_grid(matrix, gray.shape, align_corners=False)
    ink = F.grid_sample(1 - gray, grid, align_corners=False, padding_mode='zeros')
    ink = ink * (0.85 + 0.3 * torch.rand(b, 1, 1, 1, device=dev))   # ink gain 0.85 - 1.15
    # Fixed sigma 0.65 gaussian blur with probability 0.25
    axis = torch.arange(-2, 3, device=dev).float()
    k = torch.exp(-axis.square() / (2 * 0.65**2))
    k = k / k.sum()
    blur = F.conv2d(F.pad(ink, (2, 2, 2, 2), mode='replicate'), (k[:, None] * k[None, :])[None, None])
    ink = torch.where(torch.rand(b, 1, 1, 1, device=dev) < 0.25, blur, ink)
    # Background brightness fluctuation and gaussian noise
    aug = 1 - ink - 0.03 * torch.rand(b, 1, 1, 1, device=dev) + 0.005 * torch.randn_like(gray)
    return aug.clamp(0, 1)


def erase_ink_keep_background(gray, threshold=0.6):
    """Background Probe: Erase dark ink strokes, replacing with estimated paper background."""
    bg_mask = (gray > threshold).float()
    bg_mean = (gray * bg_mask).sum(dim=(2, 3), keepdim=True) / bg_mask.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
    erased = torch.where(gray <= threshold, bg_mean + 0.01 * torch.randn_like(gray), gray)
    return erased.clamp(0, 1)
