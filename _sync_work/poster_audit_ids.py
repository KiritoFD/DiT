#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""poster_audit_ids.py — 对指定 id 渲染: std(g) | GT(w7 训练目标) | GT(png 1px)。

用于肉眼判断"同一条 id 上标准骨架和真迹是不是同一个字"。
触发: audit_pairs.py 报出 std/GT 容差 clDice 低于错配对照的若干条。
用法: python _sync_work/poster_audit_ids.py --ids 17391,24176,25502 --out _ot_scratch/poster_audit.png
"""
import argparse
import csv
import glob
import os
import re
import sys

import numpy as np
import torch as th

os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from PIL import Image, ImageDraw                                 # noqa: E402


def load(d, ids):
    want, out = set(ids), {}
    for f in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        with np.load(f) as z:
            for j, i in enumerate(z["img_ids"]):
                if int(i) in want:
                    out[int(i)] = np.asarray(z["latents"][j], np.float32)
        if len(out) >= len(want):
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", default="_ot_scratch/poster_audit.png")
    ap.add_argument("--cell", type=int, default=256)
    a = ap.parse_args()
    ids = [int(x) for x in a.ids.split(",") if x.strip()]

    meta = {}
    for r in csv.DictReader(open("assets/val_skelnet.csv", encoding="utf-8")):
        m = re.search(r"(\d+)\.png$", r.get("image_path", ""))
        if m and int(m.group(1)) in ids:
            meta[int(m.group(1))] = r
    std = load("data/top10_style23/shards_std", ids)
    w7 = load("data/top10_style23/shards_gtskel_w7", ids)
    print(f"std 命中 {len(std)} / w7 命中 {len(w7)}")

    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        "data/pretrained/pretrained_models/sd-vae-ft-ema").to("cuda").eval()

    def dec(m):
        z = th.from_numpy(np.stack([m[i] for i in ids])).to("cuda")
        out = []
        with th.no_grad(), th.autocast("cuda", dtype=th.float16):
            for s in range(0, z.shape[0], 8):
                out.append((vae.decode((z[s:s + 8] / 0.18215).half()).sample.mean(1) < 0)
                           .float().cpu().numpy())
        return np.concatenate(out)

    d_std, d_w7 = dec(std), dec(w7)
    png = {i: (np.asarray(Image.open(
        f"data/top10_style23/gt_skel_png/{i:06d}.png").convert("L")) < 128)
        for i in ids}

    cell, lab = a.cell, 26
    cols = [("std(g)", d_std), ("GT(w7 tgt)", d_w7), ("GT(png)", None)]
    W = cell * 3 + 8
    H = lab * 2 + (cell + lab) * len(ids)
    cv = Image.new("L", (W, H), 255)
    dr = ImageDraw.Draw(cv)
    dr.text((6, 6), "audit: std vs GT (same id?)", fill=0)
    for c, (nm, _) in enumerate(cols):
        dr.text((c * cell + 6, lab + 4), nm, fill=0)
    for r, i in enumerate(ids):
        yy = lab * 2 + r * (cell + lab)
        for c, (nm, arr) in enumerate(cols):
            bits = png[i] if arr is None else arr[r]
            im = Image.fromarray(np.where(bits, 0, 255).astype(np.uint8))
            if cell != im.size[0]:
                im = im.resize((cell, cell), Image.LANCZOS)
            cv.paste(im, (c * cell, yy))
        ch = meta.get(i, {}).get("char") or "?"
        ca = meta.get(i, {}).get("calligrapher") or "?"
        dr.text((4, yy + cell + 2), f"id{i:06d}  csv字={ch}  {ca}", fill=0)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cv.save(a.out)
    print("poster ->", a.out, f"({W}x{H})")


if __name__ == "__main__":
    main()
