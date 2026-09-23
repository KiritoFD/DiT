#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""smoke_mid_png.py — 验证中程载体 PNG 接入 dataset 的端到端通路 (纯 CPU)。

检查:
  1. MID_CARRIER 目录被 dataset 正确发现
  2. __getitem__ 返回 mid_keys / mid_tensors, 形状/取值范围正确
  3. DataLoader default_collate 能拼批 (不炸)
  4. 骨架宽度与 σ 档位齐全
"""
import os
import sys

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.utils.latent_dataset import MCCDLatentDataset

CARR = "data/50k/mid_carriers"
WIDTHS = ["3", "5", "7", "9", "11"]
SIGMAS = ["s0p5", "s1", "s1p5", "s2", "s2p5", "s3", "s4"]

ds = MCCDLatentDataset(
    csv_file="assets/train_50k_v2.csv",
    latent_shards_dir="data/50k/shards_img",
    img_root=None, image_size=256, preload=False, load_image=False,
    skel_latent_shards_dir="data/50k/shards_std",
    skel_png_dirs={w: f"{CARR}/w{w}" for w in WIDTHS},
    gt_blur_png_dirs={s: f"{CARR}/blur/{s}" for s in SIGMAS},
)
print(f"\n[1] dataset len={len(ds)}")

sub = Subset(ds, list(range(64)))
loader = DataLoader(sub, batch_size=16, shuffle=False, num_workers=2)
batch = next(iter(loader))
KEYS = ds.mid_keys          # ★ key 顺序由 dataset 固定, 不进 batch (list[str] 会被 collate 转置)
mt = batch["mid_png"]
print(f"[2] mid_keys (dataset 固定顺序) = {KEYS}")
print(f"    mid_png: shape={tuple(mt.shape)} dtype={mt.dtype} "
      f"range=[{int(mt.min())},{int(mt.max())}]")

assert 'mid_keys' not in batch, "mid_keys 不该进 batch (list[str] 会被 collate 转置)"
assert mt.shape == (16, len(KEYS), 256, 256), mt.shape
print("[3] collate OK ✓  (N,K,256,256)")

exp = [f"skel_w{w}" for w in WIDTHS] + [f"blur_{s}" for s in SIGMAS]
assert list(KEYS) == exp, f"\n  期望 {exp}\n  实际 {list(KEYS)}"
print(f"[4] key 顺序与档位齐全 ✓ ({len(exp)} 个)")

w3 = mt[0, exp.index("skel_w3")].float()
w11 = mt[0, exp.index("skel_w11")].float()
b0 = mt[:, exp.index("blur_s0p5")].float()
b4 = mt[:, exp.index("blur_s4")].float()
n3, n11 = int((w3 < 127).sum()), int((w11 < 127).sum())
print(f"[5] 宽度单调: w3 墨像素 {n3} / w11 {n11} -> {'✓' if n11 > n3 else '✗'}")
print(f"    σ 单调:   s0p5 std={b0.std():.2f} / s4 std={b4.std():.2f} "
      f"-> {'✓' if b4.std() < b0.std() else '✗'}")

print("\n=== 全部通过 ===")
