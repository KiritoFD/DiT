#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/diag_skelset_space.py — 比较**真正相关**的两套骨架空间。

背景: 想"把 top10 SkelNet 的输出当 cond 喂给 fame ControlNet", 需要
  · top10 SkelNet 的输出空间 = top10 的 **GT 骨架** latent (v24 的 inst_skel 目录)
  · fame ControlNet 的 cond 空间 = fame 的 **1px GT 骨架** latent
只有这两者同源, 才能直接对接。之前的比较用的是 top10 的 *std* 骨架, 对不上号。

本脚本对比:
  A) top10 shards_aux_skel3   (SkelNet 的目标空间)
  B) fame final_skel_latents_fame_1px_v8 (ctrl 的 cond 空间)
按 glyph_id 对齐 -> 解码 IoU + 前景比。
"""
import csv
import glob
import os
import sys

import numpy as np
import torch

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

A_DIR = "data/top10_style23/shards_aux_skel3"
A_CSV = "assets/train_top10_style23.csv"
B_DIR = "data/skel/final_skel_latents_fame_1px_v8"
B_CSV = "data/archive/legacy_csv/train_fame.csv"
SCALING = 0.18215
N_DEC = 16


def index_dir(d):
    idx = {}
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        with np.load(f) as z:
            for j, i in enumerate(z["img_ids"]):
                idx[int(i)] = (f, j)
    return idx


def key_map(csvp, key="glyph_id"):
    out = {}
    for r in csv.DictReader(open(csvp, encoding="utf-8")):
        k = int(r[key])
        if k not in out:
            iid = int(r["img_id"]) if "img_id" in r else int(
                os.path.basename(r["image_path"])[:-4])
            out[k] = iid
    return out


class Lazy:
    def __init__(self, idx):
        self.idx, self.cur, self.arr = idx, {}, None

    def get(self, iid):
        f, j = self.idx[iid]
        if self.cur.get("f") != f:
            with np.load(f) as z:
                self.arr = z["latents"][j].astype(np.float32)
            self.cur = {"f": f}
        return self.arr


def main():
    for d in (A_DIR, B_DIR):
        if not os.path.isdir(d):
            print("✗ 目录不存在: %s" % d); return
    print("[1] 建索引 ...", flush=True)
    ia, ib = index_dir(A_DIR), index_dir(B_DIR)
    print("  A(top10 aux_skel3) ids=%d | B(fame 1px_v8) ids=%d"
          % (len(ia), len(ib)), flush=True)
    ga, gb = key_map(A_CSV), key_map(B_CSV)
    common = sorted(set(ga) & set(gb))
    print("  共同 glyph_id = %d" % len(common), flush=True)

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("pretrained_models/sd-vae-ft-ema").eval()
    LA, LB = Lazy(ia), Lazy(ib)

    def ink(lat):
        with torch.no_grad():
            z = torch.from_numpy(lat[None]) / SCALING
            g = vae.decode(z).sample[0].mean(0).numpy()
        return g < 0.0

    rng = np.random.default_rng(0)
    pick = [common[i] for i in rng.choice(len(common), min(N_DEC, len(common)), replace=False)]
    print("\n[2] 同 glyph 跨线 IoU", flush=True)
    ious, fa, fb = [], [], []
    for k, gid in enumerate(pick):
        a, b = ink(LA.get(ga[gid])), ink(LB.get(gb[gid]))
        u = np.logical_or(a, b).sum()
        ious.append(np.logical_and(a, b).sum() / max(u, 1))
        fa.append(a.mean()); fb.append(b.mean())
        print("   [%2d/%d] glyph=%-6d IoU=%.3f (top10_gt fg=%.4f fame_1px fg=%.4f)"
              % (k + 1, len(pick), gid, ious[-1], fa[-1], fb[-1]), flush=True)
    ious = np.array(ious)
    print("\n  IoU mean=%.4f median=%.4f max=%.4f" % (ious.mean(), np.median(ious), ious.max()))
    print("  前景比: top10_gt=%.4f  fame_1px=%.4f  (比值 %.2f)"
          % (np.mean(fa), np.mean(fb), np.mean(fa) / max(np.mean(fb), 1e-6)))
    print("\n[3] 判读")
    if ious.mean() > 0.85:
        print("  => 同源: top10 SkelNet 输出可直接当 fame ctrl 的 cond")
    elif ious.mean() > 0.6:
        print("  => 相近: 可试, 但需注意分布偏移")
    else:
        print("  => 不同源: 不能直接对接")


if __name__ == "__main__":
    main()
