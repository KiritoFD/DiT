# -*- coding: utf-8 -*-
"""
_build_fame3_e.py — 构建 fame3-e 增强训练集 (全部新目录, orig 用 symlink 共享).

输入: data/imgs/fame3-e/<id>_{b1,b2,b3}.png (85,155 张) + train_fame3_clean_v8.csv (28,385 行)
输出:
  data/imgs/final_imgs_fame_e/          orig 图 symlink + 增强图拷贝 (纯数字 id)
  data/latents/final_latents_fame_e/       orig shard symlink + 新 encode shard (shard_10000+)
  data/skel/std_skel1_latents_fame_e/   orig skel shard symlink + 重键 skel shard (g 与 orig 相同)
  assets/train_fame3_e_full.csv   orig 行 + 增强行 (113,540)
新 id = 1_000_000 起连续编号 (与 orig 无冲突).
"""
import csv
import glob
import os
import shutil
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DEVICE = torch.device("cuda")

ORIG_CSV = "assets/train_fame3_clean_v8.csv"
AUG_CSV = "assets/train_fame3_e.csv"
IMG_OUT = "data/imgs/final_imgs_fame_e"
LAT_OUT = "data/latents/final_latents_fame_e"
SKEL_OUT = "data/skel/std_skel1_latents_fame_e"
FULL_CSV = "assets/train_fame3_e_full.csv"
ORIG_IMG_DIR = "data/imgs/final_imgs_fame_v8"
ORIG_LAT_DIR = "data/latents/final_latents_fame_v8"
ORIG_SKEL_DIR = "data/skel/std_skel1_latents_fame3_v8"
VAE_PATH = "data/pretrained/sd-vae-ft-ema"
SCALING_FACTOR = 0.18215
SHARD_SIZE = 5000
VAE_BS = 64
ID_BASE = 1_000_000
SHARD_START = 10000

for d in (IMG_OUT, LAT_OUT, SKEL_OUT):
    os.makedirs(d, exist_ok=True)

orig_rows = list(csv.DictReader(open(ORIG_CSV, encoding="utf-8")))
aug_rows = list(csv.DictReader(open(AUG_CSV, encoding="utf-8")))
print(f"orig={len(orig_rows)} aug={len(aug_rows)}")

# ---------- 1. orig 图 symlink ----------
n_link = 0
for r in orig_rows:
    src = os.path.join(ROOT, r["image_path"])
    dst = os.path.join(IMG_OUT, os.path.basename(src))
    if not os.path.exists(dst):
        os.symlink(os.path.relpath(src, IMG_OUT), dst)
        n_link += 1
print(f"[imgs] orig symlinks: {n_link} new (total {len(os.listdir(IMG_OUT))})")

# ---------- 2. orig shard symlink ----------
n_link = 0
for sp in glob.glob(os.path.join(ORIG_LAT_DIR, "shard_*.npz")):
    dst = os.path.join(LAT_OUT, os.path.basename(sp))
    if not os.path.exists(dst):
        os.symlink(os.path.relpath(sp, LAT_OUT), dst)
        n_link += 1
for sp in glob.glob(os.path.join(ORIG_SKEL_DIR, "shard_*.npz")):
    dst = os.path.join(SKEL_OUT, os.path.basename(sp))
    if not os.path.exists(dst):
        os.symlink(os.path.relpath(sp, SKEL_OUT), dst)
        n_link += 1
print(f"[shards] orig symlinks: {n_link} new")

# ---------- 3. id 分配 + 增强图拷贝 ----------
new_ids = []
t0 = time.time()
for k, r in enumerate(aug_rows):
    src = os.path.join(ROOT, r["image_path"])
    nid = ID_BASE + k
    dst = os.path.join(IMG_OUT, f"{nid}.png")
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)
    new_ids.append(nid)
print(f"[imgs] aug copied: {len(new_ids)} in {time.time() - t0:.0f}s")

# ---------- 4. VAE encode 增强图 ----------
from diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained(VAE_PATH).to(DEVICE)
vae.eval()
for p in vae.parameters():
    p.requires_grad_(False)


def read_img(path):
    img = Image.open(path).convert("L")
    a = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    a = np.stack([a, a, a], axis=-1)
    return np.transpose(a, (2, 0, 1))


t0 = time.time()
lat_buf, id_buf = [], []
shard_i = SHARD_START


def flush():
    global shard_i, lat_buf, id_buf
    if not id_buf:
        return
    lat = np.stack(lat_buf, 0).astype(np.float16)
    ids = np.array(id_buf, dtype=np.int64)
    out = os.path.join(LAT_OUT, f"shard_{shard_i:05d}.npz")
    np.savez_compressed(out, latents=lat, img_ids=ids)
    print(f"  {out}: {lat.shape}", flush=True)
    shard_i += 1
    lat_buf, id_buf = [], []


n_enc = 0
for i in range(0, len(new_ids), VAE_BS):
    nid_b = new_ids[i:i + VAE_BS]
    xs = np.stack([read_img(os.path.join(IMG_OUT, f"{n}.png")) for n in nid_b], 0)
    xt = torch.from_numpy(xs).to(DEVICE)
    with torch.no_grad():
        lat = vae.encode(xt).latent_dist.sample() * SCALING_FACTOR
        if lat.shape[-1] != 32:
            lat = torch.nn.functional.interpolate(lat, size=32, mode="bilinear")
        lat = lat.float().cpu().numpy()
    lat_buf.extend(lat[k] for k in range(lat.shape[0]))
    id_buf.extend(nid_b)
    n_enc += len(nid_b)
    if len(id_buf) >= SHARD_SIZE:
        flush()
    if (i // VAE_BS) % 128 == 0:
        el = time.time() - t0
        print(f"  encode {n_enc}/{len(new_ids)}  {n_enc / max(el, 1):.0f}/s", flush=True)
flush()
print(f"[latents] encoded {n_enc} in {time.time() - t0:.0f}s")

# ---------- 5. skel latent 重键拷贝 ----------
orig_ids = []
for r in orig_rows:
    import re
    orig_ids.append(int(re.search(r"(\d+)\.png", r["image_path"]).group(1)))
skel_index = {}
for sp in sorted(glob.glob(os.path.join(ORIG_SKEL_DIR, "shard_*.npz"))):
    d = np.load(sp)
    for j, iid in enumerate(d["img_ids"]):
        skel_index[int(iid)] = (sp, j)
    d.close()
t0 = time.time()
# aug 行与 orig 行同序映射: aug csv 每 3 行对应 1 个 orig id? 验证: aug csv 行数 = orig*3
# 安全做法: aug csv 保留了 orig 的全部列, 但没有 orig img_id. 用 orig_ids 顺序 + 计数:
# aug csv 生成顺序是 for i in rows: for kind in (a,b,c) -> 不适用 (v3 只有 b1/b2/b3)
# = for i in rows: for kind in (b1,b2,b3), 即每 orig id 恰好 3 行连续
assert len(aug_rows) == 3 * len(orig_rows), (len(aug_rows), len(orig_rows))
aug_orig_id = []
for k, r in enumerate(aug_rows):
    aug_orig_id.append(orig_ids[k // 3])
buf, bid = [], []
sk_shard_i = SHARD_START


def flush_skel():
    global sk_shard_i, buf, bid
    if not bid:
        return
    lat = np.stack(buf, 0).astype(np.float16)
    ids = np.array(bid, dtype=np.int64)
    out = os.path.join(SKEL_OUT, f"shard_{sk_shard_i:05d}.npz")
    np.savez_compressed(out, latents=lat, img_ids=ids)
    print(f"  {out}: {lat.shape}", flush=True)
    sk_shard_i += 1
    buf, bid = [], []


for k, (nid, oid) in enumerate(zip(new_ids, aug_orig_id)):
    sp, j = skel_index[oid]
    d = np.load(sp)
    buf.append(d["latents"][j])
    d.close()
    bid.append(nid)
    if len(bid) >= SHARD_SIZE:
        flush_skel()
flush_skel()
print(f"[skel] rekeyed {len(bid) + (sk_shard_i - SHARD_START) * SHARD_SIZE} in {time.time() - t0:.0f}s")

# ---------- 6. 合并 csv ----------
with open(FULL_CSV, "w", encoding="utf-8", newline="") as f:
    fields = list(orig_rows[0].keys()) + ["aug"]
    w = csv.DictWriter(f, fieldnames=fields)
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

# ---------- 7. 校验 ----------
lat_ids = set()
for sp in sorted(glob.glob(os.path.join(LAT_OUT, "shard_*.npz"))):
    d = np.load(sp)
    lat_ids.update(int(x) for x in d["img_ids"])
    d.close()
skel_ids = set()
for sp in sorted(glob.glob(os.path.join(SKEL_OUT, "shard_*.npz"))):
    d = np.load(sp)
    skel_ids.update(int(x) for x in d["img_ids"])
    d.close()
all_ids = set(orig_ids) | set(new_ids)
miss_lat = all_ids - lat_ids
miss_skel = all_ids - skel_ids
print(f"[verify] ids={len(all_ids)} lat={len(lat_ids)} skel={len(skel_ids)} "
      f"miss_lat={len(miss_lat)} miss_skel={len(miss_skel)}")
if not miss_lat and not miss_skel:
    print("ALL OK ✓")
else:
    print("MISSING! abort")
    sys.exit(1)
