# -*- coding: utf-8 -*-
"""_fix_fame3_e_skel.py — skel 重键(按 shard 分组) + 合并 csv + 校验."""
import csv
import glob
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ORIG_CSV = "assets/train_fame3_clean_v8.csv"
AUG_CSV = "assets/train_fame3_e.csv"
SKEL_OUT = "data/skel/std_skel1_latents_fame_e"
ORIG_SKEL_DIR = "data/skel/std_skel1_latents_fame3_v8"
FULL_CSV = "assets/train_fame3_e_full.csv"
IMG_OUT = "data/imgs/final_imgs_fame_e"
LAT_OUT = "data/latents/final_latents_fame_e"
SHARD_SIZE = 5056
SHARD_START = 10000
ID_BASE = 1_000_000

orig_rows = list(csv.DictReader(open(ORIG_CSV, encoding="utf-8")))
aug_rows = list(csv.DictReader(open(AUG_CSV, encoding="utf-8")))
orig_ids = [int(re.search(r"(\d+)\.png", r["image_path"]).group(1)) for r in orig_rows]
assert len(aug_rows) == 3 * len(orig_rows)
new_ids = [ID_BASE + k for k in range(len(aug_rows))]
aug_orig_id = [orig_ids[k // 3] for k in range(len(aug_rows))]

# ---------- skel 重键: 按 shard 分组, 每 shard 只解压一次 ----------
skel_index = {}
for sp in sorted(glob.glob(os.path.join(ORIG_SKEL_DIR, "shard_*.npz"))):
    d = np.load(sp)
    for j, iid in enumerate(d["img_ids"]):
        skel_index[int(iid)] = (sp, j)
    d.close()

t0 = time.time()
by_shard = defaultdict(list)  # shard -> [(out_row, j)]
for k, oid in enumerate(aug_orig_id):
    sp, j = skel_index[oid]
    by_shard[sp].append((k, j))

out_lat = np.empty((len(aug_rows), 4, 32, 32), dtype=np.float16)
for sp, items in by_shard.items():
    d = np.load(sp)
    lat = d["latents"]
    for k, j in items:
        out_lat[k] = lat[j]
    d.close()
print(f"[skel] gathered {out_lat.shape[0]} in {time.time() - t0:.0f}s")

sk_shard_i = SHARD_START
for s in range(0, len(out_lat), SHARD_SIZE):
    sl = min(s + SHARD_SIZE, len(out_lat))
    out = os.path.join(SKEL_OUT, f"shard_{sk_shard_i:05d}.npz")
    np.savez_compressed(out, latents=out_lat[s:sl],
                        img_ids=np.array(new_ids[s:sl], dtype=np.int64))
    sk_shard_i += 1
print(f"[skel] wrote {sk_shard_i - SHARD_START} shards")

# ---------- 合并 csv ----------
with open(FULL_CSV, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(orig_rows[0].keys()) + ["aug"])
    w.writeheader()
    for r in orig_rows:
        r2 = dict(r)
        r2["aug"] = "orig"
        w.writerow(r2)
    for k, r in enumerate(aug_rows):
        r2 = dict(r)
        r2["image_path"] = f"{IMG_OUT}/{new_ids[k]}.png"
        r2["aug"] = r.get("aug", "b")
        w.writerow(r2)
n = sum(1 for _ in open(FULL_CSV, encoding="utf-8")) - 1
print(f"[csv] {FULL_CSV}: {n} rows (期望 {len(orig_rows) + len(aug_rows)})")

# ---------- 校验 ----------
def collect_ids(d_):
    ids = set()
    for sp in sorted(glob.glob(os.path.join(d_, "shard_*.npz"))):
        z = np.load(sp)
        ids.update(int(x) for x in z["img_ids"])
        z.close()
    return ids

all_ids = set(orig_ids) | set(new_ids)
lat_ids = collect_ids(LAT_OUT)
skel_ids = collect_ids(SKEL_OUT)
imgs = set(int(re.search(r"(\d+)\.png", p).group(1))
           for p in glob.glob(os.path.join(IMG_OUT, "*.png")))
miss = {k: len(all_ids - v) for k, v in
        (("img", imgs), ("lat", lat_ids), ("skel", skel_ids))}
print(f"[verify] ids={len(all_ids)} img={len(imgs)} lat={len(lat_ids)} "
      f"skel={len(skel_ids)} miss={miss}")
if not any(miss.values()):
    print("ALL OK ✓")
else:
    sys.exit(1)
