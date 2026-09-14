# -*- coding: utf-8 -*-
"""dino_expand_base_sym.py — 把 dino cache 从 base(54,892) 扩展到 158K 全对称增强.

为什么可以复用:
  aug_base_sym.py 的增强是"笔画粗细" (dilate/erode), 字形/位置/语义都不变
  -> DINO(CLS/patch) 特征基本不变, 增强图直接复用其原图的特征是合理的
     (比关掉 REPA 或重新跑 158k 张 DINO 便宜得多)。

格式 (与 src/data/dino_cache.py 一致):
  ids.npy   : (n,) int64  img_id
  feats.f16 : (n, patches, dim) float16  flat 存储
  meta.json : {n, patches, dim, backbone, csv}

产物: data/dino_cache/base_sym_v1/
"""
import csv
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(ROOT)

SRC = "data/dino_cache/base_v1"
OUT = "data/dino_cache/base_sym_v1"
CSV = "assets/train_base_sym.csv"
CSV0 = "assets/train_base_noaug.csv"


def main():
    meta = json.load(open(os.path.join(SRC, "meta.json"), encoding="utf-8"))
    n0, P, D = int(meta["n"]), int(meta["patches"]), int(meta["dim"])
    ids = np.load(os.path.join(SRC, "ids.npy"))
    feats = np.fromfile(os.path.join(SRC, "feats.f16"), dtype=np.float16)
    assert feats.shape[0] == n0 * P * D, f"feats {feats.shape} != {n0}*{P}*{D}"
    feats = feats.reshape(n0, P, D)
    print(f"[src] {SRC}: n={n0} patches={P} dim={D} ids={ids.shape}", flush=True)

    row_of = {int(i): j for j, i in enumerate(ids)}

    # (script, character) -> 原图 img_id
    rows0 = list(csv.DictReader(open(CSV0, encoding="utf-8")))
    ch2iid = {}
    for r in rows0:
        m = re.search(r"(\d+)\.png", r["image_path"])
        if not m:
            continue
        iid = int(m.group(1))
        if iid in row_of:
            ch2iid.setdefault((r.get("script", ""), r.get("character", "")), iid)
    print(f"[ch2iid] {len(ch2iid)} 可复用", flush=True)

    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    out_ids = np.empty(len(rows), dtype=np.int64)
    out_f = np.empty((len(rows), P, D), dtype=np.float16)
    miss = 0
    for k, r in enumerate(rows):
        m = re.search(r"(\d+)\.png", r.get("image_path", ""))
        iid = int(m.group(1)) if m else -1
        j = row_of.get(iid)
        if j is None:
            src = ch2iid.get((r.get("script", ""), r.get("character", "")))
            j = row_of.get(src) if src is not None else None
        if j is None:
            miss += 1
            j = 0                     # 兜底: 不该发生, 用第 0 个避免 shape 错
        out_ids[k] = iid
        out_f[k] = feats[j]

    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "ids.npy"), out_ids)
    out_f.tofile(os.path.join(OUT, "feats.f16"))
    meta2 = dict(meta)
    meta2["n"] = len(rows)
    meta2["csv"] = CSV
    meta2["src"] = f"{SRC} (expanded; aug rows reuse source-image DINO feats)"
    json.dump(meta2, open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[done] {OUT}: n={len(rows)} miss={miss} "
          f"({100*miss/max(len(rows),1):.2f}%)", flush=True)
    if miss:
        print("  !! WARNING: 有行没有 DINO 特征 —— 这些行的 REPA 目标无意义。", flush=True)


if __name__ == "__main__":
    main()
