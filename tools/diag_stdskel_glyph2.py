#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/diag_stdskel_glyph2.py — 按 glyph_id 对齐比较两套 std 骨架 (懒加载版)。

为什么重写: 之前一版 `load_shards()` 把 38583+51322 个 latent 全读进内存, 在
CPU 被占满时十几分钟不吐字。本版先只读每个 shard 的 `img_ids` 建索引 (极快),
再**按需**只加载用到的 shard。

判定: 同 glyph 跨线 IoU (解码到 256 墨迹图)。
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

TOP10_SH = "data/top10_style23/shards_std"
TOP10_CSV = "assets/train_top10_style23.csv"
FAME_SH = "data/archive/legacy_skeletons/std_skel1_latents_fame"
FAME_CSV = "data/archive/legacy_csv/train_fame.csv"
SCALING = 0.18215
N_DEC = 16


def index_dir(d):
    """img_id -> (shard_path, offset)  只读 img_ids, 不碰 latents。"""
    idx = {}
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        with np.load(f) as z:
            for j, i in enumerate(z["img_ids"]):
                idx[int(i)] = (f, j)
    return idx


def key_map(csvp, key):
    out = {}
    for r in csv.DictReader(open(csvp, encoding="utf-8")):
        k = int(r[key])
        if k in out:
            continue
        iid = int(r["img_id"]) if "img_id" in r else int(
            os.path.basename(r["image_path"])[:-4])
        out[k] = iid
    return out


class Lazy:
    def __init__(self, idx):
        self.idx = idx
        self.cur = {}
        self.arr = None

    def get(self, iid):
        f, j = self.idx[iid]
        if self.cur.get("f") != f:
            with np.load(f) as z:
                self.arr = z["latents"][j].astype(np.float32)
            self.cur = {"f": f}
            return self.arr
        return self.arr


def main():
    print("[1] 建索引 (只读 img_ids) ...", flush=True)
    i10 = index_dir(TOP10_SH)
    ifm = index_dir(FAME_SH)
    print("  top10 ids=%d | fame ids=%d" % (len(i10), len(ifm)), flush=True)

    g10 = key_map(TOP10_CSV, "glyph_id")
    gfm = key_map(FAME_CSV, "glyph_id")
    print("  top10 glyphs=%d | fame glyphs=%d" % (len(g10), len(gfm)), flush=True)

    common = sorted(set(g10) & set(gfm))
    print("  共同 glyph_id = %d" % len(common), flush=True)
    if not common:
        print("  ✗ 无共同 glyph"); return

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("pretrained_models/sd-vae-ft-ema").eval()
    print("  VAE 就绪", flush=True)

    L10, Lfm = Lazy(i10), Lazy(ifm)

    def ink(lat):
        with torch.no_grad():
            z = torch.from_numpy(lat[None]) / SCALING
            g = vae.decode(z).sample[0].mean(0).numpy()
        return g < 0.0

    rng = np.random.default_rng(0)
    pick = [common[i] for i in rng.choice(len(common), min(N_DEC, len(common)), replace=False)]
    print("\n[2] 同 glyph 跨线 IoU (解码 256 墨迹)", flush=True)
    ious, fa, fb = [], [], []
    for k, gid in enumerate(pick):
        a = ink(L10.get(g10[gid]))
        b = ink(Lfm.get(gfm[gid]))
        u = np.logical_or(a, b).sum()
        ious.append(np.logical_and(a, b).sum() / max(u, 1))
        fa.append(a.mean()); fb.append(b.mean())
        print("   [%2d/%d] glyph=%-6d IoU=%.3f (top10 fg=%.4f fame fg=%.4f)"
              % (k + 1, len(pick), gid, ious[-1], fa[-1], fb[-1]), flush=True)

    ious = np.array(ious)
    print("\n  IoU: mean=%.4f median=%.4f min=%.4f max=%.4f"
          % (ious.mean(), np.median(ious), ious.min(), ious.max()), flush=True)
    print("  前景比: top10=%.4f fame=%.4f" % (np.mean(fa), np.mean(fb)), flush=True)
    print("\n[3] 判读", flush=True)
    if ious.mean() > 0.85:
        print("  => 同源 (IoU %.3f)" % ious.mean())
    elif ious.mean() > 0.6:
        print("  => 相近但有差异 (IoU %.3f)" % ious.mean())
    else:
        print("  => 不同源 (IoU %.3f)" % ious.mean())


if __name__ == "__main__":
    main()
