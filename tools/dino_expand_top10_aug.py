#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dino_expand_top10_aug.py — 把 REPA 的 DINO cache 从 top10(26,002) 扩到对称增强全集(77,823)。

为什么可以复用源图特征 (与 tools/dino_expand_base_sym.py 同一判据):
    build_top10_sym_aug.py 的增强只有**笔画粗细** (binary_dilation / binary_erosion),
    字形、位置、书家、书体全部不变 -> DINOv2 的 patch-token 基本不变。
    复用原图特征比"关掉 REPA"或"重跑 78k 张 DINO"都合理: 保住配方单变量, 且零额外前向。
    (先例: data/dino_cache/base_sym_v1 的 meta 就写着 "aug rows reuse source-image DINO feats")

img_id 约定 (来自 tools/build_top10_sym_aug.py):
    原图: 保持原 img_id
    tp  : 7000000 + 原 img_id      (变粗)
    tn  : 7100000 + 原 img_id      (变细)
本脚本按该约定反解源 id, 并用 CSV 的 `src_image_path`(内容派生的独立证据)二次交叉核对。

⚠ 不用文件名正则: 增强行 image_path 形如 `0006533_tp.png`, 正则 `(\d+)\.png` 匹配不到,
   会退化成 -1 并静默全表指向同一行 (tools/dino_expand_base_sym.py 用 (script,character)
   兜底, 但我们 top10 里同一 (书体,字) 有 10 个书家 -> 会取到随机书家的特征, 更隐蔽)。

输出 (cache_dir, 格式同 src/utils/dino_cache.py):
    feats.f16  — (N, 256, 384) float16 flat
    ids.npy    — (N,) int64   (= 增强 CSV 的 img_id, 顺序 = CSV 行序)
    meta.json  — {n, patches, dim, backbone, csv, src}

用法 (远端 4090):
  python tools/dino_expand_top10_aug.py \
      --src data/dino_cache/top10_v1 \
      --csv exp-std/csv/train_top10_aug_sym.csv \
      --src-csv exp-std/csv/train.csv \
      --out data/dino_cache/top10_aug_v1
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

TP_BASE = 7000000
TN_BASE = 7100000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/dino_cache/top10_v1")
    ap.add_argument("--csv", default="exp-std/csv/train_top10_aug_sym.csv")
    ap.add_argument("--src-csv", default="exp-std/csv/train.csv",
                    help="原图清单 (用于 src_image_path 交叉核对)")
    ap.add_argument("--out", default="data/dino_cache/top10_aug_v1")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    meta = json.load(open(os.path.join(a.src, "meta.json"), encoding="utf-8"))
    n0, P, D = int(meta["n"]), int(meta["patches"]), int(meta["dim"])
    ids = np.load(os.path.join(a.src, "ids.npy"))
    feats = np.fromfile(os.path.join(a.src, "feats.f16"), dtype=np.float16)
    assert feats.shape[0] == n0 * P * D, f"feats {feats.shape} != {n0}*{P}*{D}"
    feats = feats.reshape(n0, P, D)
    row_of = {int(i): j for j, i in enumerate(ids)}
    print(f"[src] {a.src}: n={n0:,} patches={P} dim={D}")

    # (内容派生核对表) 原图 src_image_path -> img_id
    src2iid = {}
    for r in csv.DictReader(open(a.src_csv, encoding="utf-8")):
        sp = (r.get("src_image_path") or "").strip()
        if sp:
            src2iid.setdefault(sp, int(r["img_id"]))
    print(f"[src-csv] {a.src_csv}: {len(src2iid):,} 个 src_image_path")

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    n = len(rows)
    out_ids = np.empty(n, dtype=np.int64)
    out_f = np.empty((n, P, D), dtype=np.float16)
    miss, xcheck_fail, n_aug = 0, 0, {"tp": 0, "tn": 0, "orig": 0}

    for k, r in enumerate(rows):
        tag = (r.get("aug") or "").strip()
        iid = int(r["img_id"])
        if tag == "tp":
            src_id = iid - TP_BASE
            n_aug["tp"] += 1
        elif tag == "tn":
            src_id = iid - TN_BASE
            n_aug["tn"] += 1
        else:
            src_id = iid
            n_aug["orig"] += 1
        j = row_of.get(src_id)
        if j is None:
            miss += 1
            j = 0
        out_ids[k] = iid
        out_f[k] = feats[j]
        # 交叉核对: 增强行的 src_image_path 应能独立解析出同一个 src_id
        if tag in ("tp", "tn"):
            sp = (r.get("src_image_path") or "").strip()
            if sp and src2iid.get(sp) != src_id:
                xcheck_fail += 1

    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "ids.npy"), out_ids)
    out_f.tofile(os.path.join(a.out, "feats.f16"))
    meta2 = dict(meta)
    meta2["n"] = n
    meta2["csv"] = a.csv
    meta2["src"] = f"{a.src} (expanded; aug rows reuse source-image DINO feats)"
    json.dump(meta2, open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"[done] {a.out}: n={n:,} (orig {n_aug['orig']:,} / tp {n_aug['tp']:,} / tn {n_aug['tn']:,})")
    print(f"[done] 源特征缺失 miss={miss} ({100*miss/max(n,1):.2f}%)  "
          f"交叉核对不符 xcheck_fail={xcheck_fail}")
    print(f"[done] ids 唯一性: {len(np.unique(out_ids)) == n}  "
          f"feats 大小 {os.path.getsize(os.path.join(a.out,'feats.f16'))/2**30:.1f} GiB")
    if miss or xcheck_fail:
        print("  !! 有行未对齐 —— 这些行的 REPA 目标是错的, 必须修好再训。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
