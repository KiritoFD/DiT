# -*- coding: utf-8 -*-
"""检查 clean 修复图 id 与旧 shard id 的映射，确认"局部替换"可行性。"""
import os, sys, csv, re, glob
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)

# clean csv 里指向 clean 目录的行 (1430)
clean_rows = []
all_rows = []
for r in csv.DictReader(open("assets/train_fame_clean.csv", encoding="utf-8")):
    all_rows.append(r)
    if "clean" in r["image_path"]:
        m = re.search(r"(\d+)\.png$", r["image_path"])
        clean_rows.append((int(m.group(1)), r["image_path"]))

print(f"total rows={len(all_rows)}  clean rows={len(clean_rows)}")
ids_clean = sorted(iid for iid, _ in clean_rows)
print(f"clean img_ids: min={ids_clean[0] if ids_clean else None} max={ids_clean[-1] if ids_clean else None} n={len(ids_clean)}")

# 旧 shard 的 id 集合
old_ids = set()
shard_files = sorted(glob.glob("data/latents/final_latents_fame/shard_*.npz"))
print(f"old shard files: {len(shard_files)}")
for sp in shard_files[:3]:
    d = np.load(sp)
    print(f"  {sp}: latents={d['latents'].shape} img_ids={d['img_ids'][:5]}... (n={len(d['img_ids'])})")
    d.close()

# 全部旧 id
for sp in shard_files:
    d = np.load(sp)
    old_ids.update(int(x) for x in d["img_ids"])
    d.close()
print(f"old shard total ids: {len(old_ids)}")

# clean id 是否都在旧 shard 里
in_old = [i for i in ids_clean if i in old_ids]
not_in_old = [i for i in ids_clean if i not in old_ids]
print(f"clean ids in old shards: {len(in_old)}/{len(ids_clean)}")
print(f"clean ids NOT in old shards: {len(not_in_old)} -> {not_in_old[:10]}")

# csv 里的 img_id 是否唯一（无重复）
csv_ids = [int(re.search(r"(\d+)\.png$", r["image_path"]).group(1)) for r in all_rows]
print(f"csv ids unique: {len(set(csv_ids)) == len(csv_ids)}  (n={len(csv_ids)})")
