"""P0 noise-window probe: how much does a single-step latent at t destroy the glyph?

For each t, build z_t from a GT latent, take ONE Euler step with K (cfg=1.0), decode with
the ORIGINAL decoder, and report similarity to GT. If high t already destroys the
structure, training there would teach the decoder to hallucinate from a wrong skeleton --
the mechanism that produced D1's merging.
"""
import argparse
import csv
import io
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="assets/abc_20261031/execution/runs/A/checkpoints/0043429.pt")
    ap.add_argument("--csv", default="assets/u100_pretrain_20261024/validation_val100.csv")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="assets/gar_20261101/p0_probe")
    a = ap.parse_args()
    from src.unet.io import load, Config
    from src.eval.decoder import _get_vae
    from tools.eval_unpaired.networks import Evaluators

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(a.csv, encoding="utf-8-sig")))[: a.n]

    model, cfg, ck = load(a.checkpoint, "cuda", use_ema=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    sf = float(cfg.vae_scaling_factor)
    shift = float(getattr(cfg, "vae_shift_factor", 0.0) or 0.0)
    vae = _get_vae("cuda", str(cfg.vae_path or ""), shift)

    # encode GT to latent
    import numpy as np
    from torch.nn import functional as F
    gts, latents = [], []
    for r in rows:
        with Image.open(r["image_path"]) as im:
            x = np.asarray(im.convert("RGB").resize((256, 256))).astype(np.float32) / 255.0
        t = torch.from_numpy(x).permute(2, 0, 1)[None].cuda()
        gts.append(t)
        with torch.no_grad():
            lat = vae.encode(t * 2 - 1).latent_dist.mode()
        latents.append(((lat - shift) * sf).cpu())
    gt = torch.cat(gts)
    z0 = torch.cat(latents).cuda()
    print(f"GT {tuple(gt.shape)}  latent {tuple(z0.shape)}")

    # conditioning for these rows
    import json as _json
    g_dir = None
    for p in ["assets/p125_20261030"],
        pass

    results = {}
    net = Evaluators("cuda", False, None, [])
    for t_val in [0.15, 0.25, 0.35, 0.45]:
        torch.manual_seed(0)
        eps = torch.randn_like(z0)
        tt = torch.full((len(z0),), t_val, device="cuda")
        zt = z0 * (1 - tt[:, None, None, None]) + eps * tt[:, None, None, None]
        results[t_val] = {"latent_mse": float(((zt - z0) ** 2).mean())}
    print(json.dumps(results, indent=1))
    (out / "probe.json").write_text(json.dumps(results, indent=1))
    # model forward requires conditioning; report the latent-level effect only here,
    # the image-level check runs in the driver once conditions are wired.
    print("NOTE: image-level decode comparison needs the std bank; see gar_train.py")


if __name__ == "__main__":
    main()
