# -*- coding: utf-8 -*-
"""
fix_polarity.py — 反相修复: 1,681 张 UniCalli 反色图 + 下游重跑准备.

1) 反相 base_polarity_bad.csv 中 verdict=inverted 的图 (255-x)
2) 删除这些 id 的 final_skel3_base / final_canny_base PNG (重新生成)
3) 打印已有 img shard 中含这些 id 的文件 (encode 需删除重编)
"""
import csv
import glob
import multiprocessing as mp
import os
import re
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")


def invert(path):
    a = np.asarray(Image.open(path).convert("L"))
    Image.fromarray(255 - a).save(path)


def main():
    # base csv 提供 img_id -> image_path (bad.csv 只存了 src+img_id)
    id2path = {}
    for r in csv.DictReader(open("assets/train_base_noaug.csv", encoding="utf-8")):
        iid = int(re.search(r"(\d+)\.png", r["image_path"]).group(1))
        id2path[iid] = r["image_path"]

    flagged = []
    for r in csv.DictReader(open("base_polarity_bad.csv", encoding="utf-8")):
        if r["verdict"] == "inverted":
            flagged.append((int(r["img_id"]), id2path.get(int(r["img_id"]))))
    missing = sum(1 for _, p in flagged if p is None)
    print(f"inverting {len(flagged)} images (missing {missing})...", flush=True)
    paths = [p for _, p in flagged if p and os.path.exists(p)]
    with mp.Pool(32) as pool:
        pool.map(invert, paths)
    print(f"inverted {len(paths)}.", flush=True)

    # 删除下游 PNG (skel3/canny 需重生成)
    n_s = n_c = 0
    for iid, _ in flagged:
        s = f"data/skel/final_skel3_base/{iid}.png"
        c = f"data/aux/final_canny_base/{iid}.png"
        if os.path.exists(s):
            os.remove(s)
            n_s += 1
        if os.path.exists(c):
            os.remove(c)
            n_c += 1
    print(f"removed downstream pngs: skel {n_s}, canny {n_c}", flush=True)

    # 已有 img shards 中含 flagged 的文件 (需删除重编)
    bad_shards = set()
    for sp in glob.glob("data/latents/final_latents_base_shards/shard_*.npz"):
        with np.load(sp) as d:
            ids = set(int(i) for i in d["img_ids"])
            if ids & set(flagged):
                bad_shards.add(sp)
    print(f"img shards containing flagged ids: {len(bad_shards)}")
    for sp in sorted(bad_shards):
        print(f"  {sp}")


if __name__ == "__main__":
    main()
