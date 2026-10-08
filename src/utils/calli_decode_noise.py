"""Decode-side noise injection for Calli-VAE (fixes v71's flat-gray output).

Background (2026-10-09)
----------------------
Calli-VAE was trained with the convention

    x_recon = vae.decode(posterior.sample() / 0.18215)

so its decoder input must be ``sample / sf``.  Its posterior is wide:
per-element ``std`` mean ~= 0.88 (and it is *image dependent*), while the
posterior ``mean`` has std ~= 0.35.  Therefore the decoder genuinely needs the
noise term: feeding a noise-free latent collapses the output to gray.

v71's offline shards store ``mode * sf`` (std ~= 0.08).  At eval/inference the
DiT emits a latent of that same scale, and the old code did
``vae.decode(lat / sf)`` == ``decode(mode)``  ->  L1 ~= 0.77, flat gray.

The correct decoder input is ``sample / sf == (mode + std * eps) / sf``.  At
generation time we do not have the encoder logvar, but ``std`` is highly
predictable from ``mode`` (corr(|mode|, std) ~= -0.73; wide in flat regions,
tight on strokes).  A per-channel linear model over 6 local features (fit on
256 images from the VAE training distribution) recovers it well:

    method (unified / gt)        L1         SSIM
    no noise (v71 current)       0.60 / 0.71  0.45 / 0.35
    const 0.88                   0.089 / 0.109 0.823 / 0.787
    predicted std (this module)  0.031 / 0.042 0.924 / 0.884
    oracle true std              0.027 / 0.036 0.933 / 0.902
    correct offline encode       0.028          0.932

i.e. this salvage basically matches a full re-encode + retrain, without
touching the shards or the running DiT.

Usage
-----
    from src.utils.calli_decode_noise import decode_calli_latent
    x = decode_calli_latent(vae, lat, sf, add_noise=True)   # lat ~= mode*sf
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# Per-channel coefficients for  std ~= [1, |m|, m^2, avg3(|m|), localstd3(m), ||avg3(|m|)|-|m||]
# Fitted on 256 images from /home/ds/Workspace/moyi/data/unified_393k/imgs
# (tied to the calli_vae_step_25000 decoder).  See tools/fit_calli_std_coef.py.
CALLI_STD_COEF = [
    [0.9963222075215409, -0.17474414824635207, 0.1123495157881373,
     -0.3657267710796237, -0.2870154373697681, -0.151501721339276],
    [0.9762240281753896, -0.17427448253731606, 0.08873810693872879,
     0.11693835101164779, -0.37464475623762605, -0.10532029473642003],
    [0.9803008043460536, -0.03986487178615591, 0.03822547498814956,
     -0.06460517936922527, -0.260816186379907, -0.04860938639972361],
    [0.9778867583592412, -0.10177349666792607, 0.1409231348408501,
     -0.23901695421747937, -0.11989083075248493, -0.07308056229744928],
]

STD_FLOOR = 0.05


def _std_features(mode: torch.Tensor) -> torch.Tensor:
    """(N,C,H,W) mode -> (N,6,C,H,W) local features for the std predictor."""
    am = mode.abs()
    avg = F.avg_pool2d(am, 3, 1, 1)
    mu = F.avg_pool2d(mode, 3, 1, 1)
    var = (F.avg_pool2d(mode * mode, 3, 1, 1) - mu * mu).clamp_min(0.0)
    return torch.stack(
        [torch.ones_like(mode), am, mode * mode, avg, var.sqrt(), (avg - am).abs()],
        dim=1,
    )


def predict_std(mode: torch.Tensor, coef=None) -> torch.Tensor:
    """Predict the Calli-VAE posterior std from the posterior mean.

    ``mode`` is the *un-scaled* encoder mean (i.e. ``lat / sf``), shape (N,C,H,W).
    Returns a tensor of the same shape (clamped to >= STD_FLOOR).
    """
    coef = coef if coef is not None else CALLI_STD_COEF
    f = _std_features(mode.float())                      # (N,6,C,H,W)
    outs = []
    for ch in range(mode.shape[1]):
        c = torch.as_tensor(coef[ch], dtype=f.dtype, device=f.device).view(1, 6, 1, 1)
        outs.append((f[:, :, ch] * c).sum(1))
    return torch.stack(outs, dim=1).clamp_min(STD_FLOOR)


def decode_calli_latent(vae, lat: torch.Tensor, sf: float = 0.18215,
                        add_noise: bool = True, generator=None, coef=None):
    """Decode a Calli-VAE latent produced by a DiT.

    ``lat`` is the DiT output, in the same units as the offline shards
    (``mode * sf``).  Returns ``vae.decode(...).sample`` in [-1, 1].

    ``add_noise=True``  -> decoder input ``(mode + std*eps) / sf``  (correct)
    ``add_noise=False`` -> decoder input ``lat / sf`` == ``mode``    (v71 bug)
    """
    dtype = lat.dtype
    mode = (lat.float() / sf)
    if add_noise:
        std = predict_std(mode, coef)
        eps = torch.randn(mode.shape, device=mode.device, dtype=mode.dtype, generator=generator)
        u = (mode + std * eps) / sf
    else:
        u = mode
    return vae.decode(u.to(dtype)).sample


def _self_test(imgs, vae_path, sf=0.18215, n=16):
    """python -m src.utils.calli_decode_noise --imgs <dir> --vae <path>

    Verifies that decode-side noise injection on `mode*sf` reproduces the
    correct `decode(sample/sf)` reconstruction.
    """
    import glob

    import torchvision.transforms.functional as TF
    from diffusers import AutoencoderKL
    from PIL import Image

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(vae_path).to(dev).eval()
    files = sorted(glob.glob(imgs + "/*.png"))[:n]
    files += sorted(glob.glob(imgs + "/*.jpg"))[: max(0, n - len(files))]
    x = torch.stack([TF.to_tensor(Image.open(p).convert("RGB").resize((256, 256))) * 2 - 1
                     for p in files]).to(dev)
    with torch.no_grad():
        post = vae.encode(x).latent_dist
        mode = post.mean.float()
        std = post.std.float()
        sample = post.sample().float()
        repro = vae.decode((sample / sf).to(x.dtype)).sample.float()
        lat_mimic = mode * sf                            # what the v71 shards store
        bad = decode_calli_latent(vae, lat_mimic, sf, add_noise=False).float()   # decode(mode)
        good = decode_calli_latent(vae, lat_mimic, sf, add_noise=True).float()   # our salvage
    g = (x.float() + 1) / 2
    print("std: mean=%.3f | mode std=%.3f | predicted std mean=%.3f"
          % (std.mean(), mode.std(), predict_std(mode).mean()))

    def l1(r):
        return ((r - x.float()).abs().mean()).item()

    ok = (l1(good) < 0.06) and (l1(good) < 0.4 * l1(bad))
    print("L1  correct  decode(sample/sf)      = %.4f" % l1(repro))
    print("L1  v71 bug  decode(mode)            = %.4f" % l1(bad))
    print("L1  salvage  decode_calli(add_noise) = %.4f" % l1(good))
    print("salvage recovered %.0f%% of correct-encode quality"
          % (100 * (l1(bad) - l1(good)) / max(l1(bad) - l1(repro), 1e-9)))
    print("SELF-CHECK", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--sf", type=float, default=0.18215)
    ap.add_argument("--n", type=int, default=16)
    _a = ap.parse_args()
    _self_test(_a.imgs, _a.vae, _a.sf, _a.n)
