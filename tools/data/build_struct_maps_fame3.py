# -*- coding: utf-8 -*-
"""生成结构监督用像素图 (白线黑底, 与 probe 训练缓存 >127=正类 一致).

对 train_fame3_clean_v8.csv 每行:
  - skeleton: skeletonize(墨迹) -> 白线黑底
  - canny   : cv2.Canny -> 白边黑底
输出: struct_fame3/skel/<id>.png, struct_fame3/canny/<id>.png
"""
import argparse
import csv
import os
import re
from multiprocessing import Pool

import numpy as np
from PIL import Image


def _one(task):
    iid, path, out_skel, out_canny = task
    try:
        with Image.open(path) as im:
            g = np.asarray(im.convert("L"), dtype=np.uint8)
    except Exception:
        return iid
    ink = g < 128
    try:
        from skimage.morphology import skeletonize
        sk = skeletonize(ink)
    except ImportError:
        from scipy.ndimage import binary_erosion, generate_binary_structure
        im2 = ink.copy()
        sk = np.zeros_like(ink)
        st = generate_binary_structure(2, 2)
        while im2.any():
            er = binary_erosion(im2, structure=st)
            sk |= im2 & ~er
            im2 = er
    Image.fromarray(np.where(sk, 255, 0).astype(np.uint8)).save(out_skel)
    try:
        import cv2
        ed = cv2.Canny(g, 50, 150)
    except Exception:
        from skimage.feature import canny as _canny
        ed = (_canny(g, sigma=1.5) * 255).astype(np.uint8)
    Image.fromarray(ed.astype(np.uint8)).save(out_canny)
    return iid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="5script/train_fame3_clean_v8.csv")
    ap.add_argument("--img-root", default="final_imgs_fame_v8")
    ap.add_argument("--out", default="struct_fame3")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    sk_d = os.path.join(args.out, "skel")
    ca_d = os.path.join(args.out, "canny")
    os.makedirs(sk_d, exist_ok=True)
    os.makedirs(ca_d, exist_ok=True)
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    tasks = []
    for r in rows:
        m = re.search(r"(\d+)\.png", r["image_path"])
        if not m:
            continue
        iid = int(m.group(1))
        tasks.append((iid, os.path.join(args.img_root, f"{iid}.png"),
                      os.path.join(sk_d, f"{iid}.png"), os.path.join(ca_d, f"{iid}.png")))
    print(f"[struct] {len(tasks)} images -> {args.out}", flush=True)
    with Pool(args.workers) as pool:
        for n, _ in enumerate(pool.imap_unordered(_one, tasks, chunksize=256), 1):
            if n % 10000 == 0:
                print(f"  {n}/{len(tasks)}", flush=True)
    print(f"[struct] done: skel={len(os.listdir(sk_d))} canny={len(os.listdir(ca_d))}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
