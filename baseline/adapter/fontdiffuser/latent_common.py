# -*- coding: utf-8 -*-
"""Shared plumbing for the LATENT adaptation of FontDiffuser.

Why: the official FontDiffuser is a pixel-space DDPM; at 256px its full
self-attention (4096 tokens x 128 heads) makes training infeasible on a 24GB
4090. Per project decision we move the WHOLE model into the project's latent
space (sd-vae-ft-ema, f8, 4x32x32, scaling 0.18215):

  - target latents : precomputed project shards  data/top10_style23/shards_img
  - content latents: precomputed via the same vae_io infra (adapter 06)
  - style latents  : same shards (style refs are training images)
  - VAE            : used ONLY at the two ends (shards are precomputed;
                     decode once at inference). No pixel tensors inside models.

Shard npz format: keys 'latents' (N,4,32,32 float16, scaled) + 'img_ids' (N,).
"""
import glob

import numpy as np
import torch

DIT_ROOT = "/root/Workspace/xy/DiT"
VAE_PATH = f"{DIT_ROOT}/pretrained_models/sd-vae-ft-ema"
TARGET_SHARDS = f"{DIT_ROOT}/data/top10_style23/shards_img"
CONTENT_SHARDS = f"{DIT_ROOT}/baseline/data/content_font/deng_shards"
SCALING = 0.18215
LATENT_CH, LATENT_HW = 4, 32


class ShardIndex:
    """img_id -> latent lookup over shard_*.npz (same format as shards_img)."""

    def __init__(self, shard_dir, device="cpu"):
        self.device = device
        self.files = sorted(glob.glob(f"{shard_dir}/shard_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no shard_*.npz under {shard_dir}")
        self.index = {}
        self.cache = {}
        for f in self.files:
            with np.load(f) as d:
                for j, iid in enumerate(d["img_ids"]):
                    self.index[int(iid)] = (f, j)

    def get(self, img_id):
        f, j = self.index[int(img_id)]
        arr = self.cache.get(f)
        if arr is None:
            with np.load(f) as d:
                arr = d["latents"]
            self.cache[f] = arr          # float16 (N,4,32,32)
        return torch.from_numpy(arr[j].astype(np.float32)).to(self.device)


def build_vae(device="cuda"):
    """Frozen project VAE (decode-side only; training never touches pixels)."""
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(VAE_PATH).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


@torch.no_grad()
def decode_latents(vae, z):
    """(B,4,32,32) scaled latents -> (B,3,256,256) [-1,1] float cpu."""
    return vae.decode((z.float() / SCALING).to(vae.device)).sample.detach().cpu()


@torch.no_grad()
def blank_latent(device="cuda"):
    """Latent of an all-white 256px image — the classifier-free 'drop' input
    (latent-space analogue of the pixel pipeline's torch.ones_like image)."""
    vae = build_vae(device)
    x = torch.ones(1, 3, 256, 256, device=device)      # [-1,1] white
    from diffusers import AutoencoderKL
    with torch.autocast("cuda", dtype=torch.float32):
        h = vae.quant_conv(vae.encoder(x))
    z = h[:, :h.shape[1] // 2] * SCALING
    del vae
    torch.cuda.empty_cache()
    return z.detach()
