#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DiT Calligraphy Inference & Sampling CLI.

Examples:
  1. Generate a single character:
     python sample.py --ckpt checkpoints/0040000.pt --callig "王羲之" --script "行" --char "永" --out yong.png

  2. 2-Axis CFG (Independent Style & Glyph Strength):
     python sample.py --ckpt checkpoints/0040000.pt --callig "颜真卿" --script "楷" --char "国" --cfg-glyph 4.0 --cfg-callig 2.0 --out guo.png

  3. Batch sample from CSV:
     python sample.py --ckpt checkpoints/0040000.pt --csv 5script/eval100_top30_clean.csv --n 20 --out-dir results_eval/
"""

import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from src.inference import CalligraphySampler


def parse_args():
    parser = argparse.ArgumentParser(description="Sample calligraphy images using trained DiT checkpoints.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pt file)")
    parser.add_argument("--callig", type=str, default="颜真卿", help="Calligrapher name or integer ID")
    parser.add_argument("--script", type=str, default="楷", choices=["楷", "篆", "草", "行", "隶"], help="Script style")
    parser.add_argument("--char", type=str, default="永", help="Chinese character or integer ID")
    
    # Guidance parameters
    parser.add_argument("--cfg", type=float, default=4.0, help="Standard 1-axis CFG scale")
    parser.add_argument("--cfg-callig", type=float, default=None, help="2-Axis CFG: Calligrapher/Style scale (e.g. 2.0)")
    parser.add_argument("--cfg-glyph", type=float, default=None, help="2-Axis CFG: Glyph/Content scale (e.g. 4.0)")
    parser.add_argument("--w-inter", type=float, default=0.0, help="2-Axis CFG: 3rd-order interaction weight (default: 0.0)")

    # Sampling parameters
    parser.add_argument("--steps", type=int, default=50, help="DDIM sampling steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of samples to generate")
    
    # Overrides
    parser.add_argument("--vae-path", type=str, default=None, help="Optional VAE path override")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu")
    parser.add_argument("--no-ema", action="store_true", help="Do not use EMA weights even if present")

    # Output options
    parser.add_argument("--out", type=str, default="sample.png", help="Output file path for single sample")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV path for batch dataset generation")
    parser.add_argument("--out-dir", type=str, default="samples_output", help="Output directory when --csv is used")
    parser.add_argument("--n", type=int, default=None, help="Limit number of samples when --csv is used")

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[sample] Loading checkpoint: {args.ckpt}")
    sampler = CalligraphySampler(
        ckpt_path=args.ckpt,
        device=args.device,
        vae_path=args.vae_path,
        use_ema=not args.no_ema,
    )
    print(f"[sample] Model: {sampler.model_name} ({sampler.cond_mode}, {sampler.condition_fusion}) | VAE: downscale={sampler.vae_downscale}, channels={sampler.latent_channels}")

    if args.csv:
        print(f"[sample] Running batch CSV sampling from {args.csv} -> {args.out_dir}")
        paths = sampler.sample_csv(
            csv_path=args.csv,
            out_dir=args.out_dir,
            n=args.n,
            num_steps=args.steps,
            cfg_scale=args.cfg,
            cfg_callig=args.cfg_callig,
            cfg_glyph=args.cfg_glyph,
            w_inter=args.w_inter,
            seed=args.seed,
            batch_size=args.batch_size if args.batch_size > 1 else 16,
        )
        print(f"[sample] Batch sampling complete. Saved {len(paths)} images to {args.out_dir}")
    else:
        print(f"[sample] Generating: 书家='{args.callig}', 书体='{args.script}', 字='{args.char}' (steps={args.steps}, seed={args.seed})")
        if args.cfg_callig is not None or args.cfg_glyph is not None:
            print(f"[sample] Using 2-Axis CFG: cfg_glyph={args.cfg_glyph or args.cfg}, cfg_callig={args.cfg_callig or 2.0}, w_inter={args.w_inter}")
        else:
            print(f"[sample] Using Standard CFG: cfg={args.cfg}")

        img = sampler.sample(
            calligrapher=args.callig,
            character=args.char,
            script=args.script,
            num_steps=args.steps,
            cfg_scale=args.cfg,
            cfg_callig=args.cfg_callig,
            cfg_glyph=args.cfg_glyph,
            w_inter=args.w_inter,
            seed=args.seed,
            batch_size=args.batch_size,
            return_pil=True,
        )

        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if isinstance(img, list):
            for idx, im in enumerate(img):
                fpath = f"{os.path.splitext(args.out)[0]}_{idx}.png"
                im.save(fpath)
                print(f"[sample] Saved image: {fpath}")
        else:
            img.save(args.out)
            print(f"[sample] Saved image: {args.out}")


if __name__ == "__main__":
    main()
