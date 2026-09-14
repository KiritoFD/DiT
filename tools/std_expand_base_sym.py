# -*- coding: utf-8 -*-
"""std_expand_base_sym.py — 为 158K 全对称增强数据展开标准字形 (g 条件) latent.

为什么需要:
  aug_base_sym.py 给增强图分配了新 img_id 段 (tp: 7000000+idx / tn: 7100000+idx),
  而 std skel 是按 **原图 img_id** 索引的 -> 增强图查不到 g, 会导致 g 全零
  (历史上表现为"条件无效"的假象, 训练会静默退化)。

  增强是"笔画粗细" (dilate/erode), 字符与位置不变 -> g 直接**按 (script, character)
  复用原字形的 std skel**, 无需任何变换。

取 latent 的优先级:
  1) 该行 img_id 直接在 base_full 里命中 (原图行)
  2) 否则按 (script, character) 复用其原图的 std skel (增强行)

⚠ 坑 (2026-09-15 踩到):
  * 不能用 std_skel3_latents_base 当 uid->latent 表: 它只有"未被 fame 覆盖而新生成"
    的 5,627 个 uid, 另外 4,365 个字在 fame 的 std skel 里 -> 覆盖率只有 77%。
    完整版是 **std_skel3_latents_base_full** (合并目录, img_id-keyed)。
  * 同目录下的 expand_*.npz 是 img_id-keyed, shard_*.npz 是 uid-keyed, 别混用。

产物: data/skel/std_skel3_latents_base_sym/shard_{n:05d}.npz
"""
import csv
import glob
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(ROOT)

CSV = "assets/train_base_sym.csv"
FULL = "data/skel/std_skel3_latents_base_full"     # 完整 std skel (img_id-keyed)
OUT_DIR = "data/skel/std_skel3_latents_base_sym"
SHARD_N = 5000


def load_full():
    """img_id -> latent index, 同时缓存各 npz 的 latents."""
    idx, cache = {}, {}
    for sp in sorted(glob.glob(os.path.join(FULL, "*.npz"))):
        z = np.load(sp)
        cache[sp] = z["latents"]
        for j, i in enumerate(z["img_ids"]):
            idx[int(i)] = (sp, j)
    return idx, cache


def main():
    idx, cache = load_full()
    print(f"[base_full] {len(idx)} img_ids from {len(cache)} npz", flush=True)

    # (script, character) -> 原图 img_id (从 base 原图建)
    rows0 = list(csv.DictReader(open("assets/train_base_noaug.csv", encoding="utf-8")))
    ch2iid = {}
    for r in rows0:
        m = re.search(r"(\d+)\.png", r["image_path"])
        if not m:
            continue
        iid = int(m.group(1))
        if iid in idx:
            ch2iid.setdefault((r.get("script", ""), r.get("character", "")), iid)
    print(f"[ch2iid] {len(ch2iid)} (script,character) 可复用", flush=True)

    def get(iid, key):
        if iid in idx:
            sp, j = idx[iid]
            return cache[sp][j]
        src = ch2iid.get(key)
        if src is not None:
            sp, j = idx[src]
            return cache[sp][j]
        return None

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    buf, ids, n_shard, miss = [], [], 0, 0
    for r in rows:
        m = re.search(r"(\d+)\.png", r.get("image_path", ""))
        iid = int(m.group(1)) if m else -1
        v = get(iid, (r.get("script", ""), r.get("character", "")))
        if v is None:
            miss += 1
            continue
        buf.append(v)
        ids.append(iid)
        if len(buf) >= SHARD_N:
            np.savez(os.path.join(OUT_DIR, f"shard_{n_shard:05d}.npz"),
                     latents=np.stack(buf).astype(np.float16),
                     img_ids=np.array(ids, dtype=np.int64))
            n_shard += 1
            buf, ids = [], []
            if n_shard % 8 == 0:
                print(f"  shard {n_shard} ...", flush=True)
    if buf:
        np.savez(os.path.join(OUT_DIR, f"shard_{n_shard:05d}.npz"),
                 latents=np.stack(buf).astype(np.float16),
                 img_ids=np.array(ids, dtype=np.int64))
        n_shard += 1

    tot = sum(np.load(sp)["img_ids"].shape[0]
              for sp in glob.glob(os.path.join(OUT_DIR, "shard_*.npz")))
    print(f"[done] {OUT_DIR}: {n_shard} shards, {tot} rows covered, "
          f"miss={miss} ({100*miss/max(len(rows),1):.2f}%)", flush=True)
    if miss:
        print(f"  !! WARNING: {miss} 行的 g 会全零 —— 训练时条件无效。", flush=True)
    else:
        print("  ✓ g 覆盖率 100%", flush=True)


if __name__ == "__main__":
    main()
