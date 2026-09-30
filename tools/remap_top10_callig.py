#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/remap_top10_callig.py — 把 top10 CSV 的 `calligrapher_id` 重映射成槽位索引。

## 为什么需要
`ControlNet` harness (`src/train/legacy/train_controlnet.py`) 构造数据集时**不传**
`callig_id_map` / `callig_script_map`，于是 `MCCDLatentDataset` 里的
`map_callig_script(cid, sid, None)` **直接返回原始 calligrapher_id**。
而 top10 的 csv 用的是**原始 id**（值域 53..9001），v25 的书家表只有 **23 行**
→ 直接用会**索引越界**（或静默用错书家）。

## 做法
按 `assets/callig_script_id_map_top10.json` 的 `pair_map`，把
`(calligrapher_id, script_id) -> pair_id(0..22)`，写回 `calligrapher_id` 列。
这样 harness 拿到的就是 0..22 —— **与 v25 训练时 `map_callig_script` 得到的完全一致**，
于是零代码改动即可「和 v25 一样」。

`script_id` / `glyph_id` / `character_id` 原样保留（warm-start 下 v25 是
no_char_cond=True，char 路径不参与，无需动）。

用法:
    python tools/remap_top10_callig.py \
        --csv assets/train_top10_style23.csv \
        --out assets/train_top10_style23_slotmap.csv
"""
import argparse
import csv
import json
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8")

MAP = "assets/callig_script_id_map_top10.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--map", default=MAP)
    a = ap.parse_args()

    m = json.load(open(a.map, encoding="utf-8"))
    pm = m["pair_map"]
    print(f"[remap] pair_map {len(pm)} 条, pair 值域 "
          f"{min(int(v) for v in pm.values())}..{max(int(v) for v in pm.values())}")

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    fields = list(rows[0].keys())
    n_hit = n_miss = 0
    out_rows = []
    for r in rows:
        key = f"{int(r['calligrapher_id'])}:{int(r['script_id'])}"
        if key in pm:
            r = dict(r)
            r["calligrapher_id"] = str(int(pm[key]))
            n_hit += 1
        else:
            n_miss += 1
        out_rows.append(r)

    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    ids = sorted({int(r["calligrapher_id"]) for r in out_rows})
    print(f"[remap] {a.csv} -> {a.out}")
    print(f"  命中 pair_map {n_hit}, 未命中 {n_miss}")
    print(f"  新 calligrapher_id: {len(ids)} 个, 值域 {ids[0]}..{ids[-1]}")
    print(f"  (v25 的 num_calligraphers=23 -> {'✅ 在范围内' if ids[-1] < 23 else '❌ 越界'})")


if __name__ == "__main__":
    main()
