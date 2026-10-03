"""探针: eval_real200_cache 的来源与 shards 覆盖情况 (决定 exp-std 到底该用哪些数据)。"""
import collections
import csv
import glob
import os

import numpy as np
import torch as th

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)

c = th.load("exp-std/data/eval_real200_cache.pt", map_location="cpu", weights_only=False)
rows = c["rows"]
ids = [int(i) for i in c["img_ids"]]
print(f"[eval200] n={len(ids)}")
print(f"[eval200] source 分布 = {dict(collections.Counter(r.get('source') for r in rows))}")
print(f"[eval200] slot 分布  = {dict(collections.Counter(r.get('slot_name') for r in rows))}")
print(f"[eval200] 首行 = {{'image_path': {rows[0].get('image_path')}, "
      f"'std_path': {rows[0].get('std_path')}, 'img_id': {rows[0].get('img_id')}}}")


def ids_of(dirp):
    s = set()
    for f in sorted(glob.glob(os.path.join(dirp, "shard_*.npz"))):
        with np.load(f) as z:
            s |= {int(x) for x in z["img_ids"]}
    return s


held = set(ids)
for d in ("exp-std/data/shards_img", "exp-std/data/shards_std_w7",
          "exp-std/data/shards_gtskel_w7", "data/top10_style23/shards_std",
          "data/top10_style23/shards_aux_skel3"):
    if not os.path.isdir(d):
        print(f"[shard] {d}  (不存在)")
        continue
    s = ids_of(d)
    print(f"[shard] {d:44s} n={len(s):6d} | eval200 命中 {len(held & s):3d}/200")

print(f"[png] 目标图存在 = {sum(os.path.exists(r.get('image_path', '')) for r in rows)}/200")
print(f"[png] std 图存在 = {sum(os.path.exists(r.get('std_path', '') or '') for r in rows)}/200")

for f in sorted(glob.glob("assets/*.csv")) + sorted(glob.glob("exp-std/csv/*.csv")):
    try:
        with open(f, encoding="utf-8", errors="ignore") as fh:
            hit = [i for i in ids[:40] if str(i) in fh.read()[:10 ** 7]]
        if hit:
            print(f"[csv] 含 eval200 id: {f}  (命中 {len(hit)}/40)")
    except Exception as e:  # noqa: BLE001
        print(f"[csv] {f} 读取失败 {e}")

# 训练 csv 的 slot / source 分布 (对比是否能对齐)
with open("exp-std/csv/train_top10_style23_real.csv", encoding="utf-8") as fh:
    tr = list(csv.DictReader(fh))
print(f"[train] n={len(tr)} slot={len({r['slot_name'] for r in tr})} "
      f"source={dict(collections.Counter(r.get('source') for r in tr))}")
