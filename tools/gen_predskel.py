#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/gen_predskel.py — 预计算 SkelNet 的 **predskel** latent shards。

背景: v26-gtskel 用 GT 骨架训练。要在评测里同时报告「GT 骨架输入」与
「SkelNet predskel 输入」两套口径, 就需要把 predskel 预计算成 shards
(与 GT 骨架同样的格式: npz{latents(N,4,32,32) f16, img_ids(N)})。

SkelNet = `assets/deform_skel_top10_v1.pt` (top10 专精, style-follow 95.6%)。
它的 forward: `deform_skel(g_std, style)` -> predskel, 其中
  · g_std  = 标准字骨架 latent
  · style  = 书家风格向量, 取自 ckpt 内记录的 `style_emb`
             (= assets/callig_script_emb_top10.pt, 23x128), 索引 = pair_id(0..22)

**不手猜构造参数**: 直接用 v24_frozenskel 的 ckpt args (deform_skel=1) 走
`build_model_from_args` 建模型 —— 它内部会按正确参数构造 DeformSkel 并载入
`deform_ckpt`, 保证与训练时逐位一致。

用法:
    python tools/gen_predskel.py --set train
    python tools/gen_predskel.py --set strict84
    python tools/gen_predskel.py --set seen20
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import torch

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

V24_CKPT = ("assets/results/v24_frozenskel/"
            "20260928-125010-v24-frozenskel/checkpoints/0090000.pt")
STYLE_MAP = "assets/callig_script_id_map_top10.json"
STD_TOP10 = "data/top10_style23/shards_std"
STD_50K = "data/50k_v2_glyph15k/shards_std"

SETS = {
    # name: (csv, std_skel_dir, out_dir, id_from)
    "train": ("assets/train_top10_style23.csv", STD_TOP10,
              "data/top10_style23/predskel_train", "img_id"),
    "seen20": ("assets/eval_top10_seen_20.csv", STD_TOP10,
               "data/top10_style23/predskel_eval_seen20", "path"),
    "strict84": ("assets/eval_top10_strict_subset84.csv", STD_50K,
                 "data/top10_style23/predskel_eval_strict84", "path"),
}


def load_shard_index(d):
    idx = {}
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        with np.load(f) as z:
            for j, i in enumerate(z["img_ids"]):
                idx[int(i)] = (f, j)
    return idx


class Lazy:
    """按需取 latent。

    ⚠ [2026-09-29 修复] 原实现有个致命 bug:
        if self.cur != f:
            self.arr = z["latents"][j]     # ← 已经按 j 索引成**单张**
        return self.arr                    # ← 同一分片内所有 id 都返回这张!
      后果: 同一 shard 文件里的所有 id 拿到同一个 latent。
      实测 predskel_eval_strict84 84 条只有 21 个唯一值(75% 重复),
      seen20 20 条只有 18 个唯一值 -> **整条 pred 口径作废**。
    正确做法: 缓存**整个 latents 数组**, 再按 j 取。
    """

    def __init__(self, idx):
        self.idx, self.cur, self.arr = idx, None, None

    def get(self, iid):
        f, j = self.idx[iid]
        if self.cur != f:
            with np.load(f) as z:
                self.arr = z["latents"]        # ★ 存整个数组
            self.cur = f
        return self.arr[j].astype(np.float32)  # ★ 再按 j 取


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=list(SETS))
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    csvp, std_dir, out_dir, id_mode = SETS[a.set]
    rows = list(csv.DictReader(open(csvp, encoding="utf-8")))
    print(f"[predskel:{a.set}] {csvp}: {len(rows)} 行")

    # ---- 1) 建模型, 取出 deform_skel (参数由 v24 ckpt args 决定) ----
    from src.eval.model_io import build_model_from_args
    ck = torch.load(V24_CKPT, map_location="cpu", weights_only=False)
    aa = ck.get("args")
    aa = aa if isinstance(aa, dict) else vars(aa)
    assert int(aa.get("deform_skel", 0)) > 0, "v24 args 里 deform_skel 应为 1"
    model = build_model_from_args(aa, "cpu")
    ds = getattr(model, "deform_skel", None)
    assert ds is not None, "模型没构造出 deform_skel"
    ds = ds.eval()
    n_par = sum(p.numel() for p in ds.parameters())
    print(f"  SkelNet 就绪: {n_par:,} 参数, deform_ckpt={aa.get('deform_ckpt')}")
    del model, ck

    # ---- 2) style 表 + pair 映射 ----
    st = torch.load("assets/callig_script_emb_top10.pt", map_location="cpu",
                    weights_only=False)
    style_tab = (st["embedding"] if isinstance(st, dict) else st).float()
    pm = json.load(open(STYLE_MAP, encoding="utf-8"))["pair_map"]
    print(f"  style 表 {tuple(style_tab.shape)}, pair_map {len(pm)} 条")

    # ---- 3) 标准骨架索引 ----
    sidx = load_shard_index(std_dir)
    lazy = Lazy(sidx)
    print(f"  标准骨架 {std_dir}: {len(sidx)} ids")

    # ---- 4) 逐批生成 ----
    os.makedirs(out_dir, exist_ok=True)
    lat, ids, miss = [], [], 0
    shard_i = 0
    shard_size = 5000

    def flush():
        nonlocal lat, ids, shard_i
        if not lat:
            return
        p = os.path.join(out_dir, f"shard_{shard_i:05d}.npz")
        np.savez_compressed(p, latents=np.stack(lat).astype(np.float16),
                            img_ids=np.array(ids, dtype=np.int64))
        print(f"    -> {p}  ({len(ids)} 条)", flush=True)
        lat, ids = [], []
        shard_i += 1

    B = a.batch
    buf_g, buf_s, buf_id = [], [], []
    for k, r in enumerate(rows):
        iid = (int(r["img_id"]) if id_mode == "img_id"
               else int(os.path.basename(r["image_path"])[:-4]))
        if iid not in sidx:
            miss += 1
            continue
        key = f"{int(r['calligrapher_id'])}:{int(r['script_id'])}"
        pid = int(pm.get(key, 0))
        buf_g.append(lazy.get(iid))
        buf_s.append(style_tab[pid].numpy())
        buf_id.append(iid)
        if len(buf_g) >= B or k == len(rows) - 1:
            g = torch.from_numpy(np.stack(buf_g))
            s = torch.from_numpy(np.stack(buf_s))
            with torch.no_grad():
                out = ds(g, s)
            o = out[0] if isinstance(out, (tuple, list)) else out
            for j in range(o.shape[0]):
                lat.append(o[j].numpy())
                ids.append(buf_id[j])
            if len(lat) >= shard_size:
                flush()
            buf_g, buf_s, buf_id = [], [], []
            if (k + 1) % 5000 < B:
                print(f"    {k+1}/{len(rows)} ...", flush=True)
    flush()
    print(f"[predskel:{a.set}] DONE -> {out_dir}  缺标准骨架 {miss} 条")

    # ---- 5) 自检 ----
    tot = 0
    for f in sorted(glob.glob(out_dir + "/*.npz")):
        with np.load(f) as z:
            tot += len(z["img_ids"])
            if tot == len(z["img_ids"]):
                L = z["latents"][:4].astype(np.float32)
                print(f"  样例 latents {z['latents'].shape} {z['latents'].dtype} "
                      f"absmean={np.abs(L).mean():.4f}")
    print(f"  合计 {tot} 条")


if __name__ == "__main__":
    main()
