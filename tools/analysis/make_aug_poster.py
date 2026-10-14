# -*- coding: utf-8 -*-
"""make_aug_poster.py — 增广族四臂(A1..A4) vs 无增强(arm4) vs 原增强(arm2) 视觉对照

全部 100k 终值（A4 为 75k, 其 100k 还在跑）；同机同评测集, 零重生成。
"""
import csv
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IDX = [131, 100, 86, 67, 30, 177, 173, 22]
COLS = [("gt", "GT", (0, 0, 0)), ("arm4_noaug", "无增强", (0, 130, 0)),
        ("arm2_sym3x", "原增强", (0, 90, 200)), ("A1_morphE", "A1形态学", (170, 0, 90)),
        ("A2_elastic", "A2弹性", (200, 90, 0)), ("A3_grid", "A3网格", (120, 0, 170)),
        ("A4_affine", "A4仿射", (0, 120, 150))]
META = {int(r["idx"]): r for r in csv.DictReader(
    open("expert_pack/callig_dit_pack_20261011/02_sample_grids/grid_index.csv", encoding="utf-8"))}

MS = ["ssim", "lpips", "frag", "ink_iou", "ink_ssim", "hole", "nn_ssim", "skel_iou", "tgt_spec"]
ROWS = [
    ("arm4_noaug(无增强)", "5 项最佳: ssim/lpips/ink_iou/ink_ssim/tgt_spec", (0, 130, 0)),
    ("arm2_sym3x(原增强)", "frag 最佳 1.944 ✓ hole 最佳 0.0418 ✓", (0, 90, 200)),
    ("A3_grid(网格)", "唯一好过无增强的 frag (−0.126) ✓", (120, 0, 170)),
    ("A2_elastic(弹性)", "nn_ssim/skel_iou 最佳 ✓ frag 尚可", (200, 90, 0)),
    ("A1_morphE(形态学)", "frag 最差 2.507 ✗ 全面落后", (170, 0, 90)),
]

CELL, PADL, HDR = 118, 130, 96
NCOL = len(COLS)
NFTR = 30 + len(ROWS) * 17
W = PADL + CELL * NCOL + 16
H = HDR + CELL * len(IDX) + NFTR
img = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(img)


def font(sz, bold=False):
    for f in (("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
              "C:/Windows/Fonts/simhei.ttf"):
        if os.path.exists(f):
            try:
                return ImageFont.truetype(f, sz)
            except Exception:
                pass
    return ImageFont.load_default()


f_t, f_h, f_s = font(19, True), font(13), font(11)
d.text((8, 8), "数据增强族消融 — 全部 batch512×100k (=51.2M 样本, 同 187 张评测集)",
       fill=(0, 0, 0), font=f_t)
d.text((8, 32), "结论: 五种增强(含原有 sym3x)在同样本下, ssim/lpips/ink_iou/ink_ssim 全部不如【不增强】"
                "  |  A系列为 REPA=0(已实测混杂仅1~7%, 远小于此处差距)", fill=(80, 80, 80), font=f_s)
d.text((8, 48), "增强的代价集中在 stroke 断裂(frag): 原增强 1.944 最好; 形态学 2.507 最差; 不增强 2.369",
       fill=(80, 80, 80), font=f_s)
for j, (key, lab, col) in enumerate(COLS):
    x = PADL + j * CELL
    step = "75k" if key == "A4_affine" else ("—" if key == "gt" else "100k")
    d.text((x + 4, HDR - 34), lab, fill=col, font=f_h)
    d.text((x + 4, HDR - 18), step, fill=(150, 150, 150), font=f_s)
for i, idx in enumerate(IDX):
    y = HDR + i * CELL
    m = META.get(idx, {})
    seen = "SEEN" if m.get("combo_seen") in ("True", True) else "UNSEEN"
    d.text((6, y + 12), f"#{idx}", fill=(0, 0, 0), font=f_h)
    d.text((6, y + 30), f"{m.get('calligrapher','')}", fill=(70, 70, 70), font=f_s)
    d.text((6, y + 46), f"{m.get('script','')}/{m.get('character','')}", fill=(0, 0, 0), font=f_s)
    d.text((6, y + 64), seen, fill=(190, 0, 0) if seen == "UNSEEN" else (0, 120, 0), font=f_s)
    for j, (key, lab, col) in enumerate(COLS):
        p = (f"_local_ckpt/aug_all/gt{idx}.png" if key == "gt"
             else f"_local_ckpt/aug_all/{key}/g{idx}.png")
        x = PADL + j * CELL
        if os.path.exists(p):
            img.paste(Image.open(p).convert("RGB").resize((CELL, CELL)), (x, y))
        d.rectangle([x, y, x + CELL, y + CELL],
                    outline=(70, 200, 70) if key == "arm4_noaug" else (218, 218, 218),
                    width=2 if key == "arm4_noaug" else 1)
y0 = HDR + CELL * len(IDX) + 6
d.line([(0, y0 - 3), (W, y0 - 3)], fill=(150, 150, 150))
d.text((8, y0), "100k 各指标排名（8 臂中，越靠前越好）", fill=(0, 0, 0), font=f_h)
yy = y0 + 20
RANK = {"ssim": "arm4 0.6098 > A1 0.6045 > A2 0.6044 > A3 0.6031",
        "lpips": "arm4 0.3252 > A1/A2 0.3304 > A3 0.3313",
        "frag": "arm2 1.944 ✓ > A3 2.242 > arm4 2.369 > A2 2.412 > A1 2.507",
        "ink_iou": "arm4 0.3467 > A2 0.3381 > A3 0.3338 > A1 0.3328",
        "ink_ssim": "arm4 0.4606 > A1 0.4561 > A2 0.4548 > A3 0.4518",
        "hole": "arm2 0.0418 ✓ > arm4 0.0470 > A3 0.0489 > A1 0.0511",
        "nn_ssim": "A2 0.6871 > A1 0.6868 > arm4 0.6840 > A3 0.6818",
        "skel_iou": "A2 0.0301 > A1 0.0298 > arm4 0.0296 > arm2 0.0275",
        "tgt_spec": "arm4 0.0339 > A2 0.0312 > arm2 0.0296 > A3 0.0282"}
for m in MS:
    d.text((12, yy), f"{m:<9} {RANK[m]}", fill=(30, 30, 30), font=f_s)
    yy += 16
out = "ANALYSIS/aug_family_100k.png"
img.save(out)
print(f"  -> {out}  ({img.size[0]}x{img.size[1]})")