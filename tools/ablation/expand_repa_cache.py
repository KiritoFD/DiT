#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expand_repa_cache.py — 把 REPA 的 DINO cache 从 top10 原图扩到任意「形态学增强」变体全集。

与历史脚本 `tools/dino_expand_top10_aug.py` 的关系
--------------------------------------------------
后者把 tp/tn 的加成基址 (7.0M/7.1M) 硬编码。本脚本把「tag -> img_id 加成基址」参数化：
  * 优先读 `build_aug_variants.py` 产出的 `<out_csv>.manifest.json`
  * 也支持 `--aug-bases "tp1=7000000,tn1=7100000,..."` 显式给出
两条路径都用源图特征填充增强行 —— 成立的前提是增强**只改笔画粗细**、不改字形/位置
(`build_aug_variants.py` 只生成这类增强)。若将来引入几何增强，必须改用 DINOv2 重算。

交叉核对: 增强行的 `src_image_path` 应能独立解析回源 img_id（内容派生的第二证据），
不符会打印告警并返回码 1。

用法
----
  python tools/ablation/expand_repa_cache.py \
      --src data/dino_cache/top10_v1 \
      --csv exp-std/csv/train_top10_aug_thick.csv \
      --src-csv exp-std/csv/train.csv \
      --out data/dino_cache/top10_aug_thick_v1
"""
import argparse
import csv
import json
import os
import sys

import numpy as np


def load_bases(a):
    if a.manifest:
        m = json.load(open(a.manifest, encoding="utf-8"))
        return {k: int(v) for k, v in m["id_base"].items()}, m.get("imgs_dir")
    if a.aug_bases:
        d = {}
        for kv in a.aug_bases.split(","):
            k, v = kv.split("=")
            d[k.strip()] = int(v)
        return d, None
    raise SystemExit("需要 --manifest 或 --aug-bases")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/dino_cache/top10_v1")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--src-csv", default="exp-std/csv/train.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=None, help="默认 <csv>.manifest.json")
    ap.add_argument("--aug-bases", default=None, help="tag=base,tag=base,...")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if a.manifest is None and a.aug_bases is None:
        cand = a.csv + ".manifest.json"
        a.manifest = cand if os.path.isfile(cand) else None
    bases, _imgs = load_bases(a)
    # 反向: base -> tag，用于从 iid 反解 tag/src
    base2tag = {v: k for k, v in bases.items()}
    print(f"[repa] csv={a.csv}  aug_bases={bases}")

    meta = json.load(open(os.path.join(a.src, "meta.json"), encoding="utf-8"))
    n0, P, D = int(meta["n"]), int(meta["patches"]), int(meta["dim"])
    ids = np.load(os.path.join(a.src, "ids.npy"))
    feats = np.fromfile(os.path.join(a.src, "feats.f16"), dtype=np.float16)
    assert feats.shape[0] == n0 * P * D, f"feats {feats.shape} != {n0}*{P}*{D}"
    feats = feats.reshape(n0, P, D)
    row_of = {int(i): j for j, i in enumerate(ids)}
    print(f"[repa] src={a.src}: n={n0:,} patches={P} dim={D}")

    src2iid = {}
    for r in csv.DictReader(open(a.src_csv, encoding="utf-8")):
        sp = (r.get("src_image_path") or "").strip()
        if sp:
            src2iid.setdefault(sp, int(r["img_id"]))
    print(f"[repa] src-csv={a.src_csv}: {len(src2iid):,} 个 src_image_path")

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    n = len(rows)
    out_ids = np.empty(n, dtype=np.int64)
    out_f = np.empty((n, P, D), dtype=np.float16)
    miss = xfail = 0
    counts = {}
    for k, r in enumerate(rows):
        iid = int(r["img_id"])
        tag = (r.get("aug") or "").strip()
        src_id = iid
        hit = None
        for b, t in base2tag.items():          # 变体行: iid - base = 源 id
            if iid >= b and iid - b in row_of:
                src_id, hit = iid - b, t
                break
        if tag == "":
            counts["orig"] = counts.get("orig", 0) + 1
        else:
            counts[tag] = counts.get(tag, 0) + 1
            if hit is None:
                miss += 1
            sp = (r.get("src_image_path") or "").strip()
            if sp and src_id in row_of and src2iid.get(sp) != src_id:
                xfail += 1
        j = row_of.get(src_id)
        if j is None:
            miss += 1
            j = 0
        out_ids[k] = iid
        out_f[k] = feats[j]

    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "ids.npy"), out_ids)
    out_f.tofile(os.path.join(a.out, "feats.f16"))
    meta2 = dict(meta)
    meta2.update({"n": n, "csv": a.csv,
                  "src": f"{a.src} (expanded; aug rows reuse source-image DINO feats)",
                  "aug_bases": bases})
    json.dump(meta2, open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"[repa] done -> {a.out}: n={n:,} counts={counts}")
    print(f"[repa] miss={miss} xcheck_fail={xfail} "
          f"ids_unique={len(np.unique(out_ids)) == n} "
          f"size={os.path.getsize(os.path.join(a.out, 'feats.f16')) / 2**30:.2f} GiB")
    if miss or xfail:
        print("  !! 有行未对齐 —— 这些行 REPA 目标是错的，必须修好再训。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
