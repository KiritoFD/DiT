#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""poster_skel_variants.py — 各 SkelNet 变体 predskel 的**骨架级**对比 (看漂白)。

列: std(输入g) | w3 raw | w3 renorm | w7 raw | w7 renorm | GT(3px, v26 训练分布)
并打印每个变体的墨量比 (mean(dec<0) / mean(GT dec<0)) —— 漂白的定量指标。

用法: python tools/poster_skel_variants.py [--set seen20]
"""
import argparse
import csv
import glob
import os
import re
import sys

import numpy as np
import torch as th

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

B = "data/top10_style23"
VARIANTS = [
    ("std(input)", f"{B}/shards_std"),
    ("w3 raw", f"{B}/predskel_dit_seen20_w3raw"),
    ("w3 renorm", f"{B}/predskel_dit_seen20"),
    ("w7 raw", f"{B}/predskel_dit_seen20_w7raw"),
    ("w7 renorm", f"{B}/predskel_dit_seen20_w7ren"),
    ("GT(3px)", f"{B}/shards_aux_skel3"),
]


def load_map(d):
    mp = {}
    for sp in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        with np.load(sp) as z:
            lat = z["latents"]
            for j, i in enumerate(z["img_ids"]):
                mp[int(i)] = np.asarray(lat[j], dtype=np.float32)
    return mp


def csv_ids(p, n):
    out = []
    for r in csv.DictReader(open(p, encoding="utf-8")):
        m = re.search(r"(\d+)\.png", r["image_path"])
        if m:
            out.append(int(m.group(1)))
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="assets/eval_top10_seen_20.csv")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--cell", type=int, default=110)
    ap.add_argument("--vae", default="data/pretrained/pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--out", default="_ot_scratch/skel_variants.png")
    a = ap.parse_args()
    dev = "cuda"

    ids = csv_ids(a.csv, a.n)
    maps = {}
    for name, d in VARIANTS:
        if not os.path.isdir(d):
            print(f"[skip] {name}: 目录不存在 {d}")
            continue
        mp = load_map(d)
        have = [i for i in ids if i in mp]
        print(f"[{name}] {d}: 覆盖 {len(have)}/{len(ids)}")
        if have:
            maps[name] = mp

    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(a.vae).to(dev).eval()

    def dec(lat):
        t = th.from_numpy(np.stack(lat)).to(dev)
        with th.no_grad():
            return (vae.decode(t / 0.18215).sample.mean(1) < 0).cpu().numpy()

    imgs, inks = {}, {}
    gt_ink = None
    for name, _ in VARIANTS:
        if name not in maps:
            continue
        lat = [maps[name][i] for i in ids]
        b = dec(lat)
        imgs[name] = b
        # 逐样本墨量 (对 GT 归一)
        inks[name] = float(b.mean())
    if "GT(3px)" in inks:
        gt_ink = inks["GT(3px)"]
    print("\n=== 漂白定量 (墨量比, 对 GT 归一; <0.7 明显漂白) ===")
    for name, _ in VARIANTS:
        if name in inks:
            print(f"  {name:>12}: ink={inks[name]:.4f}  ratio={inks[name]/max(gt_ink,1e-9):.2f}")

    from PIL import Image, ImageDraw, ImageFont
    names = [n for n, _ in VARIANTS if n in imgs]
    cell, lab = a.cell, 24
    cv = Image.new("L", (len(names) * cell, len(ids) * (cell + lab)), 255)
    dr = ImageDraw.Draw(cv)
    try:
        fnt = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:                                          # noqa: BLE001
        fnt = ImageFont.load_default()
    for c, nm in enumerate(names):
        dr.text((c * cell + 4, 4), f"{nm} ({inks[nm]/max(gt_ink,1e-9):.2f}x)",
                fill=0, font=fnt)
    for r, iid in enumerate(ids):
        yy = r * (cell + lab) + lab
        for c, nm in enumerate(names):
            arr = np.where(imgs[nm][r], 0, 255).astype(np.uint8)
            cv.paste(Image.fromarray(arr).resize((cell, cell)), (c * cell, yy))
        dr.text((2, yy + cell - 14), str(iid), fill=150, font=fnt)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cv.save(a.out)
    print(f"[out] {a.out}")


if __name__ == "__main__":
    main()
