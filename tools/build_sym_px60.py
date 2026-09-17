# -*- coding: utf-8 -*-
"""build_sym_px60.py — 为 fame-kxl-tj-px60 生成 v4 对称笔画粗细增强 + g 条件 shards.

背景 (2026-09-15):
  训练用的 csv 是 **sym** 版 (base + tp/tn 增强, 配方的一部分), 且 g 条件来自
  `skel_latent_shards_dir` —— 它按 **img_id** 查表 (src/utils/latent_dataset.py:105),
  所以 **每一行(含增强行)都必须有对应的标准字 latent**, 否则 KeyError。

  增强规则 (tools/aug_base_sym.py): tp/tn uid = 7000000/7100000 + **noaug 行号**。
  本数据集是 noaug 的**子集**, 且图未改动 -> 既有增强 PNG 可直接复用, 无需重算。

  标准字: 增强行的字形与同行 base 完全相同 -> latent 直接**复制**(不重新 encode)。

产物:
  assets/train_fame-kxl-tj-px60_sym.csv      base + tp + tn
  data/fame-kxl-tj-px60/shards_std_sym/      g 条件 latent (覆盖全部 sym id)
  data/fame-kxl-tj-px60/_report_sym.json

用法: python tools/build_sym_px60.py [--apply]
"""
import argparse
import csv
import glob
import json
import os
import shutil
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NAME = "fame-kxl-tj-px60"
BASE_CSV = f"assets/train_{NAME}.csv"
OUT_CSV = f"assets/train_{NAME}_sym.csv"
NOAUG = "assets/train_base_noaug.csv"
AUG_DIR = "data/imgs/base_sym"
STD_SHARDS_IN = f"data/{NAME}/shards_std"
STD_SHARDS_OUT = f"data/{NAME}/shards_std_sym"
UID_T, UID_N = 7000000, 7100000
SHARD = 5000


def load_id_map(d):
    mp = {}
    for sp in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        with np.load(sp) as z:
            for j, i in enumerate(z["img_ids"]):
                mp[int(i)] = (sp, j)
    return mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    naug = list(csv.DictReader(open(NOAUG, encoding="utf-8")))
    b2i = {os.path.basename(r["image_path"])[:-4]: k for k, r in enumerate(naug)}
    rows = list(csv.DictReader(open(BASE_CSV, encoding="utf-8")))
    print(f"[{NAME}] base {len(rows)} 行, noaug 索引 {len(b2i)} 个")

    out, base_of, miss_aug = [], {}, 0
    for r in rows:
        bid = os.path.basename(r["image_path"])[:-4]
        idx = b2i[bid]
        r0 = dict(r)
        r0["aug"] = ""
        out.append(r0)
        base_of[int(bid)] = int(bid)
        for kind, ub in (("tp", UID_T), ("tn", UID_N)):
            uid = ub + idx
            p = f"{AUG_DIR}/{uid}.png"
            if not os.path.exists(p):
                miss_aug += 1
                continue
            r2 = dict(r)
            r2["image_path"] = p
            r2["aug"] = kind
            out.append(r2)
            base_of[uid] = int(bid)

    ids = [os.path.basename(r["image_path"])[:-4] for r in out]
    print(f"  sym 行数 {len(out)}  (base {len(rows)} + aug {len(out)-len(rows)}), 缺增强图 {miss_aug}")
    print(f"  aug 分布 {dict(Counter(r['aug'] for r in out))}")
    print(f"  id 唯一 {len(set(ids))}/{len(ids)}" + ("" if len(set(ids)) == len(ids) else "  ⚠重复!"))

    # 标准字 latent: base 行取自本数据集新 encode 的 shards_std, 增强行复制同 base
    smap = load_id_map(STD_SHARDS_IN)
    print(f"  std 源 {STD_SHARDS_IN}: {len(smap)} ids")
    lack = [i for i in ids if base_of[int(i)] not in smap]
    print(f"  无对应 std latent 的行: {len(lack)}" + ("" if not lack else "  ⚠"))

    if not a.apply:
        print("\n[DRY-RUN] 加 --apply 执行。")
        return

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()), extrasaction="ignore")
        w.writeheader()
        for r in out:
            w.writerow(r)
    print(f"[apply] {OUT_CSV}: {len(out)} 行")

    if os.path.isdir(STD_SHARDS_OUT):
        shutil.rmtree(STD_SHARDS_OUT)
    os.makedirs(STD_SHARDS_OUT, exist_ok=True)
    buf, buf_ids, n_sh = [], [], 0
    for i in ids:
        i = int(i)
        sp, j = smap[base_of[i]]
        with np.load(sp) as z:
            buf.append(np.asarray(z["latents"][j]))
        buf_ids.append(i)
        if len(buf) >= SHARD:
            np.savez(f"{STD_SHARDS_OUT}/shard_{n_sh:05d}.npz",
                     latents=np.stack(buf), img_ids=np.array(buf_ids, dtype=np.int64))
            n_sh += 1
            buf, buf_ids = [], []
    if buf:
        np.savez(f"{STD_SHARDS_OUT}/shard_{n_sh:05d}.npz",
                 latents=np.stack(buf), img_ids=np.array(buf_ids, dtype=np.int64))
        n_sh += 1
    tot = 0
    for sp in sorted(glob.glob(f"{STD_SHARDS_OUT}/shard_*.npz")):
        with np.load(sp) as z:
            tot += z["img_ids"].shape[0]
    print(f"[apply] {STD_SHARDS_OUT}: {n_sh} shards, {tot} latents")
    json.dump({"rows": len(out), "missing_aug": miss_aug, "std_latents": tot,
               "std_shards": n_sh}, open(f"data/{NAME}/_report_sym.json", "w",
                                          encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
