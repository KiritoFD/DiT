#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""derive_noaug_csv.py — 从 3x 增强 CSV 抽"原版行"(aug 列为空) 生成 noaug 语料 CSV。

动机 (2026-10-09):
  仓库惯例是 **latent shards 编码超集一次, 用 data_csv 选择语料** ——
  消融 `noaug_c2ot` 用的就是 26k 的 data_csv + 77,823 行的 `shards_img_aug`,
  数据集按 **img_id** 查表 (见 src/utils/latent_dataset.py `_check_shard_names`)。
  因此"增强版编码"里天然包含原版样本, 无需二次编码。

本工具做三件事 (缺一不可, 否则会静默取错 latent 或跑错语料):
  1. 按 aug 列过滤出原版行 (aug 为空/空白), 保持原行序;
  2. 校验 img_id 唯一 (数据集按 id 查 latent, 重复即静默错位);
  3. 若给了 --shards, 校验 **全部 img_id ⊆ shards 的 img_id 集合** (覆盖率 100%)。

用法:
  python tools/derive_noaug_csv.py \
      --src exp-std/csv/train_top10_aug_sym.csv \
      --out exp-std/csv/train_top10_noaug.csv \
      --shards exp-std/data/shards_img_aug_calli_kl1e6_s12500_rsample
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np


def load_rows(p):
    with open(p, encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r), r.fieldnames


def shard_ids(shards_dir):
    ids = set()
    n = 0
    for sp in sorted(glob.glob(os.path.join(shards_dir, "shard_*.npz"))):
        with np.load(sp) as d:
            a = d["img_ids"]
            ids.update(int(x) for x in a)
            n += len(a)
    return ids, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="源 CSV (通常是 3x 增强版)")
    ap.add_argument("--out", required=True, help="输出 noaug CSV")
    ap.add_argument("--shards", default=None, help="可选: 校验 img_id 覆盖率")
    ap.add_argument("--aug-col", default="aug", help="区分增强/原版的列名")
    ap.add_argument("--force", action="store_true", help="已存在也覆盖")
    a = ap.parse_args()

    if os.path.exists(a.out) and not a.force:
        print(f"[skip] {a.out} 已存在 (--force 覆盖)")
        return 0

    rows, fields = load_rows(a.src)
    if a.aug_col not in fields:
        print(f"[fail] {a.src} 无 '{a.aug_col}' 列: {fields}")
        return 2
    orig = [r for r in rows if not (r.get(a.aug_col) or "").strip()]
    print(f"[derive] 源 {len(rows):,} 行 -> 原版 {len(orig):,} 行 (其余为增强视图)")

    ids = [int(r["img_id"]) for r in orig]
    uniq = len(set(ids)) == len(ids)
    print(f"[derive] 原版 img_id 唯一: {uniq}")
    if not uniq:
        return 2

    if a.shards:
        sids, nrows = shard_ids(a.shards)
        miss = [i for i in ids if i not in sids]
        print(f"[derive] shards {os.path.relpath(a.shards)}: {len(sids):,} 个 id / {nrows:,} 行")
        print(f"[derive] 原版 img_id 覆盖率: {len(ids) - len(miss)}/{len(ids)} "
              f"({'100% ✅' if not miss else f'缺 {len(miss)} ❌'})")
        if miss:
            print(f"         缺失样例: {miss[:5]}")
            return 2

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(orig)
    print(f"[derive] 写出 {a.out} ({len(orig):,} 行) ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
