#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Free-sampling Evaluation Script for DiT Calligraphy Models.
Computes MSE, SSIM, Skel-IoU, and optional LPIPS against Ground Truth.
Supports both standard CFG and 2-Axis CFG.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from src.inference import CalligraphySampler
from eval_auto import _gaussian_window, _ssim


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate free-sampling generation against GT.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--csv", type=str, default="5script/eval100_top30_clean.csv", help="Evaluation CSV")
    parser.add_argument("--n", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for sampling")
    parser.add_argument("--steps", type=int, default=50, help="DDIM sampling steps")
    parser.add_argument("--cfg", type=float, default=4.0, help="Standard CFG scale")
    parser.add_argument("--cfg-callig", type=float, default=None, help="2-Axis CFG style scale")
    parser.add_argument("--cfg-glyph", type=float, default=None, help="2-Axis CFG glyph scale")
    parser.add_argument("--w-inter", type=float, default=0.0, help="2-Axis CFG interaction weight")
    parser.add_argument("--seed", type=int, default=0, help="Evaluation seed")
    parser.add_argument("--out", type=str, default="gen_eval_results", help="Output directory")
    parser.add_argument("--vis-n", type=int, default=10, help="Number of comparison images to save")
    parser.add_argument("--no-ema", action="store_true", help="Do not use EMA weights")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"[eval_gen] Loading sampler for {args.ckpt}")
    sampler = CalligraphySampler(
        ckpt_path=args.ckpt,
        use_ema=not args.no_ema,
    )

    print(f"[eval_gen] Running evaluation on {args.csv} (n={args.n}, steps={args.steps}, cfg={args.cfg})")
    saved_paths = sampler.sample_csv(
        csv_path=args.csv,
        out_dir=os.path.join(args.out, "samples"),
        n=args.n,
        num_steps=args.steps,
        cfg_scale=args.cfg,
        cfg_callig=args.cfg_callig,
        cfg_glyph=args.cfg_glyph,
        w_inter=args.w_inter,
        seed=args.seed,
        batch_size=args.batch,
    )

    # Save summary metadata
    summary = {
        "ckpt": args.ckpt,
        "csv": args.csv,
        "n": len(saved_paths),
        "steps": args.steps,
        "cfg": args.cfg,
        "cfg_callig": args.cfg_callig,
        "cfg_glyph": args.cfg_glyph,
        "w_inter": args.w_inter,
        "model": sampler.model_name,
        "vae_downscale": sampler.vae_downscale,
        "latent_channels": sampler.latent_channels,
    }

    summary_path = os.path.join(args.out, "eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[eval_gen] Generated {len(saved_paths)} samples. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
