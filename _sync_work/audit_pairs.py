#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audit_pairs.py — 查「同一条 id 上: 标准骨架 / w7 目标 / GT png」是不是同一个字。

触发: poster 最后一行 (id20314) 左边是「昼」, 而 GT 列是像「書」的字。
判据: 容差 clDice (tol=3px) —— 同一个字的骨架之间应该显著高于不同字之间。
      这里拿"每个 id 自己的 std vs png" 与 "该 id 的 std vs 别人的 png" 做对照,
      前者若落在后者的分布里, 就是错配。
"""
import csv
import glob
import json
import os
import re
import sys

import numpy as np
import torch as th

os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from PIL import Image                                            # noqa: E402
from scipy.ndimage import binary_dilation                        # noqa: E402
from skimage.morphology import skeletonize                       # noqa: E402

ST = np.ones((3, 3), bool)
N = 64
STRIDE = 47


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


def cldice(p, t, tol=3):
    if p.sum() == 0 or t.sum() == 0:
        return 0.0
    sp, sg = skeletonize(p), skeletonize(t)
    if sp.sum() == 0 or sg.sum() == 0:
        return 0.0
    gt_t = binary_dilation(t, structure=ST, iterations=tol)
    pr_t = binary_dilation(p, structure=ST, iterations=tol)
    a = float((sp & gt_t).sum()) / float(sp.sum())
    b = float((sg & pr_t).sum()) / float(sg.sum())
    return 2 * a * b / max(a + b, 1e-12)


rows = list(csv.DictReader(open("assets/val_skelnet.csv", encoding="utf-8")))
print("csv 列:", list(rows[0].keys()))
cand = []
for r in rows:
    m = re.search(r"(\d+)\.png$", r.get("image_path", ""))
    if not m:
        continue
    iid = int(m.group(1))
    if os.path.exists(f"data/top10_style23/gt_skel_png/{iid:06d}.png"):
        cand.append((iid, r))
sel = cand[::STRIDE][:N]
ids = [i for i, _ in sel]
print(f"审计 {len(sel)} 条 (val 共 {len(cand)} 条有 GT png)")

std = load("data/top10_style23/shards_std", ids)
w7 = load("data/top10_style23/shards_gtskel_w7", ids)
print(f"std 命中 {len(std)} / w7 命中 {len(w7)}")

from diffusers.models import AutoencoderKL
dev = "cuda"
vae = AutoencoderKL.from_pretrained(
    "data/pretrained/pretrained_models/sd-vae-ft-ema").to(dev).eval()


def dec(m):
    z = th.from_numpy(np.stack([m[i] for i in ids])).to(dev)
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
png7 = {i: (np.asarray(Image.open(
    f"data/top10_style23/gt_skel_png_w7/{i:06d}.png").convert("L")) < 128)
    for i in ids if os.path.exists(f"data/top10_style23/gt_skel_png_w7/{i:06d}.png")}

same = np.array([cldice(d_std[k], png[ids[k]]) for k in range(len(ids))])
cross = np.array([cldice(d_std[k], png[ids[(k + 1) % len(ids)]]) for k in range(len(ids))])
same7 = np.array([cldice(d_w7[k], png[ids[k]]) for k in range(len(ids))])
print(f"\n容差 clDice(tol=3):")
print(f"  同一 id  std vs GTpng : 中位 {np.median(same):.4f}  均值 {same.mean():.4f}")
print(f"  错配对照 std vs 别人GT: 中位 {np.median(cross):.4f}  均值 {cross.mean():.4f}")
print(f"  同一 id  w7目标 vs GTpng: 中位 {np.median(same7):.4f} 均值 {same7.mean():.4f}")

order = np.argsort(same)
print(f"\n最差的 8 条 (疑似错配):")
for k in order[:8]:
    i = ids[k]
    r = [x for x in sel if x[0] == i][0][1]
    ch = r.get("char") or r.get("character") or r.get("glyph") or "?"
    print(f"    id{i:06d} 字={ch:>2} 书家={r.get('calligrapher', '?')}  "
          f"std/GT {same[k]:.4f}  (对照均值 {cross.mean():.4f})  "
          f"w7/GT {same7[k]:.4f}  path={r.get('image_path', '')}")

print(f"\n最好的 3 条 (应当高):")
for k in order[-3:]:
    i = ids[k]
    r = [x for x in sel if x[0] == i][0][1]
    ch = r.get("char") or r.get("character") or r.get("glyph") or "?"
    print(f"    id{i:06d} 字={ch:>2}  std/GT {same[k]:.4f}  w7/GT {same7[k]:.4f}")
