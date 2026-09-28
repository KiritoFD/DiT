# -*- coding: utf-8 -*-
"""Unified eval: pair GT vs pred by filename '{slot}__{char}.png', compute
MSE / SSIM / LPIPS at native resolution (both images as-is).

Usage:
  python eval_metrics.py --gt <gt_dir> --pred <pred_dir> [--out csv]
GT and pred must be same size (we always generate at 256 = dataset resolution).

GT dirs are produced by 05_make_eval_pairs.py:
  baseline/data/eval/{strict84,seen20}/gt/{slot}__{char}.png
"""
import argparse
import csv as csvmod
import os

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn


def load_gray(path):
    im = Image.open(path).convert("L")
    return np.asarray(im, dtype=np.float64) / 255.0


class LPIPSW:
    def __init__(self, device="cuda"):
        import lpips
        self.fn = lpips.LPIPS(net="alex").to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, a, b):
        # a,b in [0,1] HxW
        t = lambda x: torch.from_numpy(x * 2 - 1).float()[None, None].to(self.device)
        return float(self.fn(t(a), t(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    preds = sorted(f for f in os.listdir(args.pred) if f.endswith(".png"))
    try:
        lp = LPIPSW(args.device)
    except Exception as e:
        print(f"LPIPS unavailable ({e}), computing MSE/SSIM only")
        lp = None

    rows = []
    for fn in preds:
        gp = os.path.join(args.gt, fn)
        pp = os.path.join(args.pred, fn)
        if not os.path.exists(gp):
            continue
        g, p = load_gray(gp), load_gray(pp)
        if g.shape != p.shape:
            from PIL import Image as I2
            p = np.asarray(I2.open(pp).convert("L").resize(
                (g.shape[1], g.shape[0]), I2.BILINEAR), dtype=np.float64) / 255.0
        mse = float(np.mean((g - p) ** 2))
        ssim = float(ssim_fn(g, p, data_range=1.0))
        lpv = lp(g, p) if lp else float("nan")
        slot, ch = fn[:-4].split("__")
        rows.append({"file": fn, "slot": slot, "char": ch,
                     "mse": mse, "ssim": ssim, "lpips": lpv})

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csvmod.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    def m(k):
        v = [r[k] for r in rows if r[k] == r[k]]
        return float(np.mean(v)) if v else float("nan")

    print(f"n={len(rows)}  MSE={m('mse'):.5f}  SSIM={m('ssim'):.4f}  LPIPS={m('lpips'):.4f}")
    if args.out:
        print(f"per-item csv -> {args.out}")


if __name__ == "__main__":
    main()
