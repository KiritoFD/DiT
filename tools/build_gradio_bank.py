"""从 v13 训练时真正用的 g 来源（data/50k/shards_std）构建 gradio 的 bank。

⚠ 为什么必须重建:
  现在 gradio 用的是 _sync_work/skel_bank_std1_v8.npz（v8 时代，9-12 生成），
  而 v13/v15 的 skel_latent_shards_dir = data/50k/shards_std。
  骨架 latent 的版本不同 -> 分布不匹配 -> 生成的图很奇怪。

bank 格式（gradio 期望）:
  keys    : "书体|字"   （如 "楷|阜"）
  latents : (N, 4, 32, 32)
"""
import argparse
import csv
import os

import numpy as np

os.chdir("/root/Workspace/xy/DiT")

ap = argparse.ArgumentParser()
ap.add_argument("--shards", default="data/50k/shards_std")
ap.add_argument("--csv", default="assets/train_50k_v2.csv")
ap.add_argument("--out", default="_sync_work/skel_bank_v13_std.npz")
a = ap.parse_args()

# 1) 读 shards: {latents, img_ids}
import glob

files = sorted(glob.glob(os.path.join(a.shards, "shard_*.npz")))
if not files:
    raise SystemExit(f"找不到 shard: {a.shards}")
LAT, IDS = [], []
for f in files:
    z = np.load(f)
    LAT.append(z["latents"])
    IDS.append(z["img_ids"])
lat = np.concatenate(LAT, 0)
ids = np.concatenate(IDS, 0)
print(f"  shards: {len(files)} 个, latent {lat.shape}, ids {ids.shape}")

# 2) csv 行序 -> (script, character)
rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
print(f"  csv: {len(rows)} 行")


def key_of(i):
    r = rows[i]
    return f"{r['script']}|{r['character']}"


# ids 可能是行号字符串，也可能是文件名
keys = []
ok_idx = []
for j, v in enumerate(ids):
    s = str(v)
    try:
        i = int(s)
    except ValueError:
        i = -1
    if 0 <= i < len(rows):
        keys.append(key_of(i))
        ok_idx.append(j)
    else:
        keys.append(None)
print(f"  可映射: {len(ok_idx)}/{len(ids)}")

# 3) 去重（同一 script|char 只留第一个）
seen = {}
for j in ok_idx:
    k = keys[j]
    if k and k not in seen:
        seen[k] = j
print(f"  去重后: {len(seen)} 条")

out_keys = list(seen.keys())
out_lat = np.asarray([lat[j] for j in seen.values()], dtype=lat.dtype)
print(f"  -> {out_keys[:3]}")
np.savez(a.out, keys=np.array(out_keys), latents=out_lat)
print(f"  written {a.out}  ({out_lat.shape})")

for t in ("楷|阜", "行|阜", "隶|阜"):
    print(f"   {t}: {'在' if t in seen else '不在'}")
