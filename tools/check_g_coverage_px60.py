# -*- coding: utf-8 -*-
"""check_g_coverage_px60.py — 起训前最后一道闸: g 覆盖 + 与图 latent 逐行对齐.

失败模式: 某个 img_id 在 skel shard 里缺失 -> latent_dataset 直接 KeyError;
          eval 侧缺失 -> g=ZERO (字条件静默失效)。
"""
import csv
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NAME = "fame-kxl-tj-px60"
BASE = f"data/{NAME}"


def load_ids(d):
    s = set()
    for sp in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        with np.load(sp) as z:
            s.update(int(i) for i in z["img_ids"])
    return s


def main():
    img = load_ids(f"{BASE}/shards_img")
    std = load_ids(f"{BASE}/shards_std")
    std_eval = load_ids(f"{BASE}/shards_std_eval")
    print(f"shards_img      {len(img)}")
    print(f"shards_std      {len(std)}")
    print(f"shards_std_eval {len(std_eval)}")

    rows = list(csv.DictReader(open(f"assets/train_{NAME}.csv", encoding="utf-8")))
    tid = {int(os.path.basename(r["image_path"])[:-4]) for r in rows}
    print(f"\ntrain csv {len(rows)} 行, {len(tid)} id")
    print(f"  img 覆盖: {len(tid - img)} 缺" + (f" {sorted(tid-img)[:5]}" if tid - img else ""))
    print(f"  std 覆盖: {len(tid - std)} 缺" + (f" {sorted(tid-std)[:5]}" if tid - std else ""))
    print(f"  ★ img/std id 集合一致: {img == std}")

    for p in ("assets/eval_seen_v10.csv", "assets/eval_fame3_strict_clean_v9.csv"):
        rs = list(csv.DictReader(open(p, encoding="utf-8")))
        ids = {int(os.path.basename(r["image_path"])[:-4]) for r in rs}
        miss = ids - std_eval
        print(f"  {os.path.basename(p)}: {len(ids)} id, 缺 g {len(miss)}"
              + (f" {sorted(miss)[:5]}" if miss else "  ✓"))
        # 与训练 id 的交集 (seen 应大部分命中; strict 应为 0 -> 必须靠 eval 目录)
        print(f"      与 train id 交集 {len(ids & tid)}, 与 shards_std 交集 {len(ids & std)}")


if __name__ == "__main__":
    main()
