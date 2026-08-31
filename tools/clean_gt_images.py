# -*- coding: utf-8 -*-
"""clean_gt_images.py — GT 噪点图像清洗 (非剔除).

只处理 blacklist 命中的图, 做保守修复 (不动主笔画):
  1) 孤立小墨斑: 删除与主字(最大连通域)不连通的、面积 < 0.05% 全图的前景 -> 填白
  2) 边缘侵扰: 边界 EDGE px 环带内, 清除与主字不连通的前景 -> 填白
  3) 连通域分析在灰度二值 (a<128) 上做; 修复只把前景像素置白, 保持原灰度质感

输出: clean_root/<同 img_id>.png (只写 blacklist 命中的图; 未命中图由训练侧
      fallback 到原目录, 数据量不变)

用法:
  /opt/conda/bin/python tools/clean_gt_images.py \
      --audit gt_audit.csv --blacklist gt_blacklist_fame.csv \
      --img-root final_imgs_256 --out-root final_imgs_256_clean
"""
import os
import sys
import csv
import argparse
import numpy as np
from multiprocessing import Pool
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

SMALL_AREA = 0.0005  # 小墨斑面积阈值 (占全图比例)


def clean_one(task):
    src, dst, edge = task
    try:
        img = Image.open(src)
        a = np.asarray(img.convert("L"), dtype=np.uint8).copy()
    except Exception as e:
        return ("error", src, str(e))
    ink = a < 128
    if not ink.any():
        return ("skip_blank", src, "")
    try:
        from scipy import ndimage
        lab, nlab = ndimage.label(ink)
        if nlab == 0:
            return ("skip", src, "")
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        main_label = int(sizes.argmax())
    except Exception as e:
        return ("error", src, str(e))

    H, W = a.shape
    changed = False
    # 掩码: 非主连通域的前景
    foreign = ink & (lab != main_label)

    # 1) 删除孤立小墨斑 (非主连通域且面积 < SMALL_AREA*全图)
    small_thresh = SMALL_AREA * H * W
    for lb in range(1, nlab + 1):
        if lb == main_label:
            continue
        area = int((lab == lb).sum())
        if area < small_thresh:
            foreign |= (lab == lb)

    # 2) 边缘环带内的非主前景 (即使是较大碎块, 贴边多半是裁切残片)
    b = edge
    border_mask = np.zeros_like(ink)
    border_mask[:b, :] = True
    border_mask[-b:, :] = True
    border_mask[:, :b] = True
    border_mask[:, -b:] = True
    foreign |= (foreign & border_mask)

    if foreign.any():
        a[foreign] = 255
        changed = True

    if not changed:
        return ("clean", src, "no-op")

    # 写出: 保持原图 mode (L/RGB)
    if img.mode == "L":
        Image.fromarray(a, "L").save(dst)
    else:
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
        g = a < 128
        # 把灰度修复映射回 RGB: 被修复像素置白
        rgb[foreign] = 255
        Image.fromarray(rgb, "RGB").save(dst)
    return ("cleaned", src, "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-root", default="final_imgs_256")
    ap.add_argument("--out-root", default="final_imgs_256_clean")
    ap.add_argument("--blacklist", default="gt_blacklist_fame.csv")
    ap.add_argument("--edge", type=int, default=8)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    black = []
    with open(args.blacklist, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            black.append(r["img_id"])
    print(f"[clean] blacklist {len(black)} images -> {args.out_root}", flush=True)

    tasks = []
    for iid in black:
        src = os.path.join(args.img_root, f"{iid}.png")
        dst = os.path.join(args.out_root, f"{iid}.png")
        if not os.path.isfile(src):
            print(f"  missing {src}")
            continue
        tasks.append((src, dst, args.edge))

    stat = {}
    with Pool(args.workers) as p:
        for i, (status, src, msg) in enumerate(
                p.imap_unordered(clean_one, tasks, chunksize=8)):
            stat[status] = stat.get(status, 0) + 1
            if status == "error":
                print(f"  ERROR {src}: {msg}")
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(tasks)}", flush=True)
    print(f"[done] {stat}")


if __name__ == "__main__":
    main()
