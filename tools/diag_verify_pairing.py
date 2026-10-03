#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/diag_verify_pairing.py — 独立核验 top10_style23 数据集骨架/图像配准，以及被误用的 50k_v2 旧目录。
不信任已有断言，直接解码 latent 比较字形。
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

print("=" * 78)
print("[A] 分片键结构对比")
print("=" * 78)
for name in ["shards_img", "shards_std", "shards_aux_skel3"]:
    d = np.load("data/top10_style23/" + name + "/shard_00000.npz")
    print("  top10/" + name, "->", sorted(d.files),
          {k: tuple(d[k].shape) for k in d.files})

d2 = np.load("data/50k_v2_glyph15k/shards_std/shard_00000.npz")
print("  50k_v2_glyph15k/shards_std ->", sorted(d2.files),
      {k: tuple(d2[k].shape) for k in d2.files})

print()
print("=" * 78)
print("[B] ID 范围与重叠")
print("=" * 78)
di = np.load("data/top10_style23/shards_img/shard_00000.npz")["img_ids"]
ds = np.load("data/top10_style23/shards_std/shard_00000.npz")["img_ids"]
d2i = d2["img_ids"]
print("  top10  img_ids[0:5] =", di[:5], " std[0:5] =", ds[:5])
print("  top10  img id range: %d .. %d  (n=%d)" % (di.min(), di.max(), len(di)))
print("  50k    img id range: %d .. %d  (n=%d)" % (d2i.min(), d2i.max(), len(d2i)))
print("  id 集合重叠数量:", len(set(di.tolist()) & set(d2i.tolist())))
print("  top10 img_ids == std_ids ?", bool((di == ds).all()))

# 全部分片的 ID 串接，与 CSV 的 img_id 集合比较
all_ids = np.concatenate([
    np.load("data/top10_style23/shards_img/shard_%05d.npz" % i)["img_ids"]
    for i in range(8)
])
print("  全 8 分片 img_ids 总数:", len(all_ids), "唯一:", len(set(all_ids.tolist())))
df = pd.read_csv("assets/train_top10_style23.csv")
print("  CSV 行数:", len(df), " CSV img_id 唯一数:", df["img_id"].nunique())
si = set(all_ids.tolist()); ci = set(df["img_id"].astype(int).tolist())
print("  shard_ids ⊂ csv_ids ?", si.issubset(ci), " csv ⊂ shard ?", ci.issubset(si))
print("  在 shard 但不在 csv 的 id:", list(si - ci)[:10])
print("  在 csv 但不在 shard 的 id:", list(ci - si)[:10])

print()
print("=" * 78)
print("[C] CSV 关键列 & 书家/书体分布")
print("=" * 78)
print("  columns:", list(df.columns))
for col in ["calligrapher", "script"]:
    if col in df.columns:
        vc = df[col].value_counts()
        print("  %s: %d 类" % (col, len(vc)))
        print("   ", dict(list(vc.items())[:25]))
if "calligrapher" in df.columns and "script" in df.columns:
    pair = df.groupby(["calligrapher", "script"]).size().sort_values(ascending=False)
    print("  (calligrapher x script) 组合数:", len(pair))
    print("  前 25 组合:", dict(list(pair.items())[:25]))

print()
print("=" * 78)
print("[D] callig_script_id_map_top10.json")
print("=" * 78)
csmap_p = "assets/callig_script_id_map_top10.json"
if os.path.exists(csmap_p):
    m = json.load(open(csmap_p, encoding="utf-8"))
    print("  顶层键:", list(m.keys())[:10])
    print("  内容预览:", json.dumps(m, ensure_ascii=False)[:1200])
else:
    print("  !! 不存在:", csmap_p)
