#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple eval poster: pick N sample indices from eval_samples, show gen+GT side by side.
Each row = one eval step. Columns = [gen0, gt0, gen1, gt1, ...] for N samples.
Top row = step labels with SSIM/MSE. Bottom row = nothing.
"""
import os, sys, json, glob, re, argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CELL = 128
GAP = 4
HEADER_H = 28
LABEL_H = 22

BG = (15, 17, 22)
CELL_BG = (30, 30, 30)
GRID = (60, 66, 76)

def step_key(d):
    m = re.search(r"step(\d+)", os.path.basename(d))
    return int(m.group(1)) if m else 0

def load_cell(path, bg=CELL_BG):
    try:
        img = Image.open(path).convert("RGB")
    except:
        img = Image.new("RGB", (CELL, CELL), bg)
    img.thumbnail((CELL, CELL), Image.LANCZOS)
    c = Image.new("RGB", (CELL, CELL), bg)
    c.paste(img, ((CELL - img.width) // 2, (CELL - img.height) // 2))
    return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir", help="path to eval_samples dir")
    ap.add_argument("--ckpt-dir", help="ckpt dir with eval_auto_*.json")
    ap.add_argument("--exp", default="")
    ap.add_argument("--n-samples", type=int, default=10, help="number of samples per row")
    ap.add_argument("--sample-offset", type=int, default=0, help="start index")
    ap.add_argument("--step-stride", type=int, default=1, help="show every N-th step")
    ap.add_argument("--max-steps", type=int, default=0, help="only steps <= this (0=all)")
    ap.add_argument("-o", "--out", default="poster.png")
    args = ap.parse_args()

    step_dirs = sorted(glob.glob(os.path.join(args.eval_dir, "step*")), key=step_key)
    step_dirs = [d for d in step_dirs if os.path.exists(os.path.join(d, "samples.json"))]
    if args.max_steps > 0:
        step_dirs = [d for d in step_dirs if step_key(d) <= args.max_steps]
    if args.step_stride > 1:
        step_dirs = step_dirs[::args.step_stride]

    if not step_dirs:
        print("No complete step dirs found")
        return

    # Load eval metrics
    ev_map = {}
    if args.ckpt_dir:
        for f in sorted(glob.glob(os.path.join(args.ckpt_dir, "eval_auto_*.json"))):
            d = json.load(open(f))
            ev_map[d["step"]] = d

    n = args.n_samples
    n_cols = n * 2  # gen + gt per sample
    W = GAP + n_cols * CELL + GAP
    H = HEADER_H + len(step_dirs) * (LABEL_H + CELL + GAP)

    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_lab = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
        font_s = font
        font_lab = font

    # Header
    draw.rectangle([0, 0, W, HEADER_H], fill=(0, 0, 0))
    draw.text((GAP, 6), f"GEN → GT pairs ({n} samples per step)", font=font, fill=(120, 200, 255))
    if args.exp:
        draw.text((W - GAP, 6), args.exp, anchor="ra", font=font_s, fill=(90, 100, 120))
    y = HEADER_H

    si = args.sample_offset
    for d in step_dirs:
        step = step_key(d)
        # Label row
        draw.rectangle([0, y, W, y + LABEL_H], fill=(0, 0, 0))
        parts = [f"STEP {step}"]
        ev = ev_map.get(step)
        if ev:
            parts.append(f"SSIM {ev.get('ssim',0):.3f}")
            parts.append(f"LPIPS {ev.get('lpips',0):.3f}")
            parts.append(f"MSE {ev.get('mse',0):.3f}")
        draw.text((12, y + (LABEL_H - 14) // 2), "    ".join(parts), font=font_s, fill=(255, 255, 255))
        y += LABEL_H

        # Image row
        x = GAP
        for i in range(n):
            idx = si + i
            gen_path = os.path.join(d, f"sample{idx}.png")
            gt_path = os.path.join(d, f"gt{idx}.png")
            canvas.paste(load_cell(gen_path), (x, y))
            x += CELL
            # thin separator
            draw.line([(x, y), (x, y + CELL)], fill=GRID)
            canvas.paste(load_cell(gt_path), (x, y))
            x += CELL
            draw.line([(x, y), (x, y + CELL)], fill=GRID)
        y += CELL + GAP

    canvas.save(args.out)
    print(f"Poster saved: {args.out} ({W}x{H}, {len(step_dirs)} steps, {n} samples each)")

if __name__ == "__main__":
    main()
