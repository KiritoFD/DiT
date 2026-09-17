# -*- coding: utf-8 -*-
"""在**原有 eval 集**基础上小改，产出与 50k 训练集兼容的 v13 评测集。

## 为什么不能直接用原 eval 集
`src/eval/inference.py:483` 是 `callig_id_map.get(_cid, _cid)` ——
**不在词表里就退回原始 id**。原 eval 集有 5 个书家（39/346/401/483/806）
在任何训练数据里都是 0 张（base 词表里的"空号"），v13 的 45 人词表不覆盖
-> 退回原始 id（如 806）-> `y_callig_embedder(806)` 而表只有 46 行
-> **CUDA 索引越界**（v13 在 step5000 就是这么崩的）。

## 做法（最小改动）
保留**原有评测图**（这样现成的 `shards_std_eval` 还能用，不必重编码），
只**剔除书家不在 50k 词表里的行**。

## 同时校验
- 与 50k 训练集**不重合**（按 image_path 与 img_id 双向查）
- 行数尽量接近原来的 ~200，保持可比

用法:
    python tools/build_eval_v13.py
"""
import argparse
import csv
import json
import os
import re
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

TRAIN = "assets/train_50k.csv"
CMAP = "assets/callig_id_map_50k.json"


def img_id_of(p):
    m = re.search(r"(\d+)\.png$", str(p))
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=TRAIN)
    ap.add_argument("--cmap", default=CMAP)
    ap.add_argument("--out-dir", default="assets")
    a = ap.parse_args()

    cmap = {int(k): int(v) for k, v in
            json.load(open(a.cmap, encoding="utf-8"))["id_map"].items()}
    train_rows = list(csv.DictReader(open(a.train, encoding="utf-8")))
    train_paths = {r["image_path"] for r in train_rows}
    train_ids = {img_id_of(r["image_path"]) for r in train_rows}
    print(f"[eval-v13] 训练集 {len(train_rows)} 行, 词表 {len(cmap)} 个书家")

    # ⚠ seen 与 strict 的**重合语义相反**：
    #   strict = 未见图 -> 必须剔除与训练重合的
    #   seen   = 模型见过的图 -> **就该**与训练重合，剔了反而变成"没见过的"
    #   （第一版我把两者都剔了，seen 从 10 行变 8 行 —— 那是错的。）
    for src, dst, note, drop_overlap_ok in (
        ("assets/eval_fame3_strict_clean_v9.csv", "assets/eval_v13_strict.csv",
         "strict（未见图，跨书家泛化）", True),
        ("assets/eval_seen_v10.csv", "assets/eval_v13_seen.csv",
         "seen（模型见过的图）", False),
    ):
        if not os.path.exists(src):
            print(f"[eval-v13] (缺) {src}")
            continue
        rows = list(csv.DictReader(open(src, encoding="utf-8")))
        keep, drop_calig, drop_overlap = [], [], []
        for r in rows:
            cid = int(r["calligrapher_id"])
            if cid not in cmap:
                drop_calig.append(cid)
                continue
            p = r["image_path"]
            if drop_overlap_ok and (p in train_paths or img_id_of(p) in train_ids):
                drop_overlap.append(p)
                continue
            # 映射到新索引，写进新列（训练/评测两侧都用同一张表，这里只是留痕）
            r = dict(r)
            r["calligrapher_idx"] = cmap[cid]
            keep.append(r)

        if not keep:
            print(f"[eval-v13] {src}: 全被剔了，跳过")
            continue

        with open(dst, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(keep[0].keys()))
            w.writeheader()
            w.writerows(keep)
        print(f"[eval-v13] {note}")
        print(f"    {src} {len(rows)} 行 -> {dst} {len(keep)} 行")
        print(f"    剔除：书家不在词表 {len(drop_calig)} 行 "
              f"({sorted(set(drop_calig))})；与训练重合 {len(drop_overlap)} 行")
        cids = sorted(set(int(r["calligrapher_id"]) for r in keep))
        print(f"    保留书家 {len(cids)} 个，索引范围 "
              f"{min(cmap[c] for c in cids)}..{max(cmap[c] for c in cids)}")


if __name__ == "__main__":
    main()
