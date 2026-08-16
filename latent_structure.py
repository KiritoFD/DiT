"""Low-memory structural supervision directly in cached VAE-latent space."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class LatentStructureProbe(nn.Module):
    """Small 32x32 latent-to-(Canny, skeleton) predictor.

    Train this separately on GT latents and freeze it before using its skeleton
    logits as a loss.  Keeping it frozen prevents the probe from adapting to a
    poor DiT prediction and makes the structural signal identifiable.
    """

    def __init__(self, in_channels=4, width=32, depth=2):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, width, 3, padding=1)
        self.blocks = nn.Sequential(*[_ResidualConv(width) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, 2, 1))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def downsample_structure(target, size):
    """Keep thin positive lines when mapping a 256x256 target to latent size."""
    if target is None or target.numel() == 0:
        return None
    return F.adaptive_max_pool2d(target.float(), size)


def _edge_weighted_gradient_loss(pred, target, edge_map, eps=1e-3):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    true_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    true_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    wx = torch.maximum(edge_map[:, :, :, 1:], edge_map[:, :, :, :-1])
    wy = torch.maximum(edge_map[:, :, 1:, :], edge_map[:, :, :-1, :])
    ex = ((pred_dx - true_dx).square() + eps * eps).sqrt().mean(dim=1, keepdim=True)
    ey = ((pred_dy - true_dy).square() + eps * eps).sqrt().mean(dim=1, keepdim=True)
    lx = (ex * wx).sum() / wx.sum().clamp_min(1.0)
    ly = (ey * wy).sum() / wy.sum().clamp_min(1.0)
    return 0.5 * (lx + ly)


def _balanced_bce_dice(logits, target, eps=1e-6):
    positives = target.sum()
    negatives = target.numel() - positives
    pos_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 10.0)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    probs = logits.sigmoid()
    dice = 1.0 - ((2.0 * (probs * target).sum() + eps)
                  / (probs.sum() + target.sum() + eps))
    return bce + dice


class LatentStructureLoss(nn.Module):
    """Canny gradient consistency plus optional frozen-probe skeleton loss."""

    def __init__(self, probe=None, max_timestep=500):
        super().__init__()
        self.probe = probe
        self.max_timestep = int(max_timestep)
        if self.probe is not None:
            self.probe.eval()
            self.probe.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.probe is not None:
            self.probe.eval()
        return self

    def forward(self, pred_x0, target_x0, timesteps, canny=None, skeleton=None):
        active = timesteps <= self.max_timestep
        zero = pred_x0.sum() * 0.0
        if not active.any():
            return {"canny": zero, "skeleton": zero}
        active_fraction = active.float().mean()

        pred = pred_x0[active].float()
        target = target_x0[active].float()
        size = pred.shape[-2:]
        canny_small = downsample_structure(
            canny[active] if canny is not None and canny.numel() else None, size)
        skel_small = downsample_structure(
            skeleton[active] if skeleton is not None and skeleton.numel() else None, size)

        canny_loss = zero
        if canny_small is not None:
            canny_loss = _edge_weighted_gradient_loss(
                pred, target, canny_small.to(pred.device)) * active_fraction

        skeleton_loss = zero
        if skel_small is not None:
            if self.probe is None:
                raise RuntimeError("skeleton latent loss requires a pretrained frozen probe")
            skeleton_logits = self.probe(pred)[:, 1:2]
            skeleton_loss = _balanced_bce_dice(
                skeleton_logits, skel_small.to(pred.device)) * active_fraction
        return {"canny": canny_loss, "skeleton": skeleton_loss}
