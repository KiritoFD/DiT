"""从 data/50k/shards_std 构建 gradio 的 bank —— **用正确的 id 映射**。

## 之前为什么错（实测）
shards_std 的 img_ids **不是 csv 行号**，而是**标准字形库的编号**:
    data/50k/std/{img_id}.png     该目录 52,457 个文件，id 范围 0..52456 —— 完全对应
之前写 rows[int(v)] -> 错位 -> 输入"小"生成出"将"。

## 正确做法
csv 每行有 std_path（如 data/50k/std/019888.png）-> 提取数字 = img_id
于是 img_id -> (script, character) 一一对应，绝不错位。

## bank 格式（gradio 期望）
    keys:    "书体|字"   （如 "楷|小"）
    latents: (N, 4, 32, 32)
"""
import argparse
import csv
import glob
import os
import re

import numpy as np

os.chdir("/root/Workspace/xy/DiT")

ap = argparse.ArgumentParser()
ap.add_argument("--shards", default="data/50k/shards_std")
ap.add_argument("--csv", default="assets/train_50k_v2.csv")
ap.add_argument("--out", default="_sync_work/skel_bank_v13_std.npz")
a = ap.parse_args()

# 1) shards: id -> (shard 文件, 偏移)
files = sorted(glob.glob(os.path.join(a.shards, "shard_*.npz")))
id2loc = {}
for f in files:
    z = np.load(f)
    for j, iid in enumerate(z["img_ids"]):
        id2loc[int(iid)] = (f, j)
print(f"  shards: {len(files)} 个, 唯一 id {len(id2loc)}")

_lat_cache = {}


def get_lat(iid):
    f, j = id2loc[iid]
    if f not in _lat_cache:
        _lat_cache[f] = np.load(f)["latents"]
    return _lat_cache[f][j]


# 2) csv: std_path -> img_id -> (script, character)
rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
print(f"  csv: {len(rows)} 行")

seen = {}
n_nostd = 0
n_miss = 0
for r in rows:
    sp = r.get("std_path", "")
    m = re.search(r"(\d+)", os.path.basename(sp))
    if not m:
        n_nostd += 1
        continue
    gid = int(m.group(1))
    if gid not in id2loc:
        n_miss += 1
        continue
    key = f"{r['script']}|{r['character']}"
    if key not in seen:
        seen[key] = gid
print(f"  无 std_path: {n_nostd}; id 不在 shards: {n_miss}; 唯一 key: {len(seen)}")

out_keys = sorted(seen.keys())
out_lat = np.stack([get_lat(seen[k]) for k in out_keys]).astype(np.float16)
print(f"  -> {out_keys[:5]}")
np.savez(a.out, keys=np.array(out_keys), latents=out_lat)
print(f"  written {a.out}  ({out_lat.shape})")

for t in ("楷|小", "行|小", "楷|阜", "行|阜", "隶|阜", "楷|将"):
    print(f"   {t}: {'在' if t in seen else '不在'}")
