#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""poster_skel64_dataset.py — 抽样本看**落盘 64² 骨架数据集**。

列 (从左到右):
  1) 原迹 64² (降采样后的墨)       —— 书法真迹, 粗笔画
  2) GT 骨架 64² 1px              —— 训练目标 x0
  3) 标准字 64² (降采样后的墨)      —— 印刷体标准字
  4) std 骨架 64² 1px             —— 条件 g
  5) 原迹 256² (参考)
  6) GT 骨架 256² PNG (参考)

用法: python tools/poster_skel64_dataset.py --split train --n 8
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/top10_style23/skel64")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--cell", type=int, default=140)
    ap.add_argument("--out", default="_ot_scratch/skel64_dataset.png")
    a = ap.parse_args()

    p = os.path.join(a.dataset, f"{a.split}.npz")
    z = np.load(p)
    ids = [int(i) for i in z["ids"]]
    img_ink, std_ink = z["img_ink"], z["std_ink"]
    gt_skel, std_skel = z["gt_skel"], z["std_skel"]
    print(f"[{a.split}] n={len(ids)} shape={img_ink.shape} dtype={img_ink.dtype} "
          f"max={img_ink.max()}")
    sc = 255.0 if img_ink.max() > 1 else 1.0
    print(f"  墨占比: 原迹 {img_ink.mean()/sc:.4f} | GT骨架 {gt_skel.mean()/sc:.4f} | "
          f"标准字 {std_ink.mean()/sc:.4f} | std骨架 {std_skel.mean()/sc:.4f}")

    # 抽样: 均匀铺开, 避开前 200 条 (避免全同一个字)
    step = max(1, (len(ids) - 200) // max(a.n, 1))
    sel = [200 + k * step for k in range(a.n)]
    sel = [i for i in sel if i < len(ids)]

    from PIL import Image, ImageDraw, ImageFont
    try:
        fnt = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:                                          # noqa: BLE001
        fnt = ImageFont.load_default()

    cols = ["原迹64", "GT骨架64(x0)", "标准字64", "std骨架64(cond)",
            "原迹256", "GT骨架256"]
    cell, lab = a.cell, 26
    cv = Image.new("L", (len(cols) * cell, len(sel) * (cell + lab)), 255)
    dr = ImageDraw.Draw(cv)
    for c, nm in enumerate(cols):
        dr.text((c * cell + 4, 6), nm, fill=0, font=fnt)
    for r, k in enumerate(sel):
        yy = r * (cell + lab) + lab
        iid = ids[k]
        tiles = []
        for key in (img_ink, gt_skel, std_ink, std_skel):
            b = (key[k].astype(np.float32) / sc > 0.5)
            tiles.append(np.where(b, 0, 255).astype(np.uint8))
        # 256² 参考
        for fp in (f"data/top10_style23/imgs/{iid:06d}.png",
                   f"data/top10_style23/gt_skel_png/{iid:06d}.png"):
            if os.path.exists(fp):
                tiles.append(np.asarray(Image.open(fp).convert("L")))
            else:
                tiles.append(np.full((256, 256), 255, np.uint8))
        for c, t in enumerate(tiles):
            im = Image.fromarray(t).resize((cell, cell), Image.NEAREST)
            cv.paste(im, (c * cell, yy))
        dr.text((4, yy + cell - 15), f"id{iid}", fill=140, font=fnt)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cv.save(a.out)
    print(f"[out] {a.out}  ids={[ids[k] for k in sel]}")


if __name__ == "__main__":
    main()
