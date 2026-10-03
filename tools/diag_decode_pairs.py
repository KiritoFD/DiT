#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/diag_decode_pairs.py — 解码核验: 图像 latent 与骨架 latent 是否同一个字。
同时对比"错配源" 50k_v2_glyph15k/shards_std 在同一 id 上到底是哪个字。
"""
import os, sys
import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.eval.in_mem_eval import _get_vae

df = pd.read_csv("assets/train_top10_style23.csv")
vae = _get_vae("cuda")
SF = 0.18215


def load_shard(dirpath, idx, n_shards=8):
    """按全局 idx 定位分片并取单条 latent。假设分片按 5120 切分。"""
    per = 5120
    s = idx // per
    o = idx % per
    d = np.load(os.path.join(dirpath, "shard_%05d.npz" % s))
    return d["latents"][o:o + 1], int(d["img_ids"][o])


def dec(lat_np):
    t = torch.from_numpy(lat_np).float().to("cuda")
    with torch.no_grad():
        out = vae.decode(t / SF).sample
    return ((out.clamp(-1, 1) + 1) / 2)[0].mean(0).cpu().numpy()


def ascii_img(a, thresh=0.5, size=32):
    import numpy as np
    m = (a < thresh)
    # 缩到 32x32
    H, W = m.shape
    ys = (np.linspace(0, H - 1, size)).astype(int)
    xs = (np.linspace(0, W - 1, size)).astype(int)
    s = m[np.ix_(ys, xs)]
    return ["".join("#" if s[y, x] else "." for x in range(size)) for y in range(size)]


print("=" * 80)
print("[E] 逐个核验 top10 训练集 (图像 latent vs 骨架 latent 字形是否一致)")
print("=" * 80)
print("idx | iid | calligrapher·script·char | 图像墨迹 | 骨架墨迹 | 骨架∩图像/骨架 | 判定")

rows = df.to_dict("records")
test_idx = [0, 1, 2, 3, 4, 50, 500, 1000, 5000, 20000, 38582]
detail_dump = None
for idx in test_idx:
    if idx >= len(rows):
        continue
    r = rows[idx]
    lat_img, iid_i = load_shard("data/top10_style23/shards_img", idx)
    lat_std, iid_s = load_shard("data/top10_style23/shards_std", idx)
    im = dec(lat_img)
    sd = dec(lat_std)
    ink_img = int((im < 0.5).sum())
    ink_std = int((sd < 0.5).sum())
    inter = int(((im < 0.5) & (sd < 0.5)).sum())
    ratio = inter / max(1, ink_std)
    verdict = "OK-同字" if ratio > 0.30 else "!! 疑似错配"
    print("%6d | %5d | %s·%s·%s | %5d | %4d | %5.1f%% | %s" % (
        idx, iid_i, r["calligrapher"], r["script"], r["character"], ink_img, ink_std, ratio * 100, verdict))
    if idx == 1:
        detail_dump = (im, sd, r)

if detail_dump is not None:
    im, sd, r = detail_dump
    print()
    print("--- 可视化 idx=1 (%s·%s·'%s') ---" % (r["calligrapher"], r["script"], r["character"]))
    ai = ascii_img(im)
    asd = ascii_img(sd)
    for y in range(32):
        print("  IMG %s | STD %s" % (ai[y], asd[y]))

print()
print("=" * 80)
print("[F] 同一 id 在错配源 50k_v2_glyph15k/shards_std 上是哪个字 (应明显不同)")
print("=" * 80)
for idx in [0, 1, 2, 50, 500, 1000]:
    lat_std_top, _ = load_shard("data/top10_style23/shards_std", idx)
    # 50k_v2 分片每片 5056
    per = 5056
    s = idx // per; o = idx % per
    d2 = np.load("data/50k_v2_glyph15k/shards_std/shard_%05d.npz" % s)
    lat_std_50k = d2["latents"][o:o + 1]
    a = dec(lat_std_top)
    b = dec(lat_std_50k)
    inter = int(((a < 0.5) & (b < 0.5)).sum())
    r = rows[idx]
    # 50k_v2 对应的 CSV (旧 50k) 若存在
    print("idx=%5d top10字='%s' | top10∩50kv2 骨架像素重合=%d (若为 0~极小 → 二者字形无关)" % (
        idx, r["character"], inter))
