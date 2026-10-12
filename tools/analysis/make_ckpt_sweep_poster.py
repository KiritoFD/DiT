# -*- coding: utf-8 -*-
"""make_ckpt_sweep_poster.py — v81 进度网格 poster（用正确的 g#.png）+ 墨量/灰度列

图源: eval/v81fix_<step>/<i>.png  （= 评测真正保存的 g#.png，已验证与打印 ssim 一致）
新增: 每行的"灰度均值 / 硬墨量 / 与GT差" —— 因为 SSIM 掩盖了墨色塌陷, 必须显式摆出来
"""
import csv
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IDX = [131, 100, 67, 30, 177, 173, 86, 83, 145, 22]
# step, ssim, ssim_med, lpips, frag, 灰度均值, 硬墨量
ROWS = [("0020000", 0.5549, 0.6176, 0.4103, 1.054, 213.4, 0.1584),
        ("0035000", 0.5512, 0.5848, 0.4124, 1.235, 209.5, 0.1735),
        ("0040000", 0.5541, 0.5794, 0.4091, 1.642, 211.1, 0.1676),
        ("0050000", 0.5771, 0.6033, 0.4034, 1.563, 217.3, 0.1440),
        ("0055000", 0.6075, 0.6743, 0.3930, 1.707, 229.1, 0.0980),
        ("0100000", 0.6302, 0.7082, 0.3959, 1.420, 238.1, 0.0638),
        ("0110000", 0.6170, 0.7326, 0.4015, 1.893, 230.8, 0.0921),
        ("0115000", 0.6284, 0.7367, 0.3938, 2.184, 235.8, 0.0730)]
META = {int(r["idx"]): r for r in csv.DictReader(
    open("expert_pack/callig_dit_pack_20261011/02_sample_grids/grid_index.csv", encoding="utf-8"))}

CELL, PADL, PADR, HDR, FTR = 150, 150, 400, 96, 76
W = PADL + CELL * len(IDX) + PADR
H = HDR + CELL * len(ROWS) + FTR
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


f_t, f_h, f_s = font(22, True), font(15), font(13)
d.text((12, 10), "v81 (std_soft) training progress — one row per checkpoint · 10 characters per row · "
                 "columns are the SAME eval rows", fill=(0, 0, 0), font=f_t)
d.text((12, 40), "ALERT: image ink collapses while SSIM keeps rising — ink 0.174(35k) → 0.073(115k) vs GT 0.185. "
                 "SSIM is blind to this.", fill=(200, 0, 0), font=f_h)
for j, idx in enumerate(IDX):
    m = META.get(idx, {})
    tag = "SEEN" if m.get("combo_seen") in ("True", True) else "UNSEEN"
    x = PADL + j * CELL
    d.text((x + 6, HDR - 60), f"{m.get('script','')}/{m.get('character','')}", fill=(0, 0, 0), font=f_h)
    d.text((x + 6, HDR - 38), f"{m.get('calligrapher','')} {tag}", fill=(190, 0, 0), font=f_s)
    d.text((x + 6, HDR - 20), f"GT ink {m.get('gt_ink','')}", fill=(120, 120, 120), font=f_s)
for i, (st, ss, md, lp, fr, gy, ik) in enumerate(ROWS):
    y = HDR + i * CELL
    d.text((10, y + 8), f"step {int(st)//1000}k", fill=(0, 0, 0), font=f_h)
    d.text((10, y + 30), f"{int(st)*288/1e6:.1f}M samples", fill=(90, 90, 90), font=f_s)
    for j, idx in enumerate(IDX):
        p = f"eval/v81fix_{st}/{j}.png"
        x = PADL + j * CELL
        if os.path.exists(p):
            img.paste(Image.open(p).convert("RGB").resize((CELL, CELL)), (x, y))
        d.rectangle([x, y, x + CELL, y + CELL], outline=(210, 210, 210))
    xr = PADL + CELL * len(IDX) + 12
    ik_col = (0, 140, 0) if ik > 0.14 else ((200, 120, 0) if ik > 0.09 else (200, 0, 0))
    d.text((xr, y + 14), f"SSIM mean {ss:.4f}", fill=(0, 0, 0), font=f_h)
    d.text((xr, y + 36), f"SSIM median {md:.4f}", fill=(0, 100, 0), font=f_h)
    d.text((xr, y + 58), f"LPIPS {lp:.4f}   frag {fr:.3f}", fill=(0, 0, 0), font=f_s)
    d.text((xr, y + 80), f"INK {ik:.4f}   gray {gy:.0f}", fill=ik_col, font=f_h)
    if ik < 0.09:
        d.text((xr, y + 102), "← ink collapse", fill=(200, 0, 0), font=f_s)
d.text((12, HDR + CELL * len(ROWS) + 10),
       "Ink(0.185=GT) by step:  0.158 → 0.174 → 0.168 → 0.144 → 0.098 → 0.064 → 0.092 → 0.073    "
       "|    SSIM by step: 0.555 → 0.551 → 0.554 → 0.577 → 0.608 → 0.630 → 0.617 → 0.628",
       fill=(0, 0, 0), font=f_s)
d.text((12, HDR + CELL * len(ROWS) + 32),
       "Conclusion: the std_soft condition (v81's ONLY change vs v68) drives a monotone ink/contrast collapse "
       "after ~50k, which SSIM does not see. Any 'win' based on SSIM alone is suspect.",
       fill=(200, 0, 0), font=f_s)
out = "ANALYSIS/v81_ckpt_sweep_poster_v2.png"
img.save(out)
print(f"  -> {out}  ({img.size[0]}x{img.size[1]})")
