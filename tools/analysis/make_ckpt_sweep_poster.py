# -*- coding: utf-8 -*-
"""make_ckpt_sweep_poster.py — v81(std_soft) 训练进度网格 poster：一行一个 ckpt, 一行 10 字

数据来源: eval/v81sw_<step>/<idx>.png （在 48 上生成后回传）
右侧面板: 每个 ckpt 的 10 字子集指标（SSIM 均值/中位、LPIPS、frag）——取自生成时的同口径打印
"""
import csv
import os
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IDX = [131, 100, 67, 30, 177, 173, 86, 83, 145, 22]
# (step, 该 10 字子集的 ssim 均值, ssim 中位, lpips, frag)  ← 生成时同口径打印
ROWS = [("0020000", 0.5549, 0.6176, 0.4103, 1.054), ("0035000", 0.5512, 0.5848, 0.4124, 1.235),
        ("0040000", 0.5541, 0.5794, 0.4091, 1.642), ("0050000", 0.5771, 0.6033, 0.4034, 1.563),
        ("0055000", 0.6075, 0.6743, 0.3930, 1.707), ("0100000", 0.6302, 0.7082, 0.3959, 1.420),
        ("0110000", 0.6170, 0.7326, 0.4015, 1.893), ("0115000", 0.6284, 0.7367, 0.3938, 2.184)]
META = {int(r["idx"]): r for r in csv.DictReader(
    open("expert_pack/callig_dit_pack_20261011/02_sample_grids/grid_index.csv", encoding="utf-8"))}

CELL, PADL, PADR, HDR, FTR = 150, 150, 330, 74, 40
W = PADL + CELL * len(IDX) + PADR
H = HDR + CELL * len(ROWS) + FTR
img = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(img)


def font(sz, bold=False):
    for f in (("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
              "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc"):
        if os.path.exists(f):
            try:
                return ImageFont.truetype(f, sz)
            except Exception:
                pass
    return ImageFont.load_default()


f_t, f_h, f_s = font(21, True), font(15), font(13)
d.text((12, 12), "v81 (std_soft spatial condition) — training progress: one row per checkpoint, 10 characters per row",
       fill=(0, 0, 0), font=f_t)
d.text((12, 42), "same 10 eval rows in every row (mix of SEEN/UNSEEN x simple/complex) · cfg1.0 / heun-50 / gamma0 · batch 288",
       fill=(90, 90, 90), font=f_s)
for j, idx in enumerate(IDX):
    m = META.get(idx, {})
    tag = "SEEN" if m.get("combo_seen") in ("True", True) else "UNSEEN"
    x = PADL + j * CELL
    d.text((x + 6, HDR - 56), f"{m.get('script','')}/{m.get('character','')}", fill=(0, 0, 0), font=f_h)
    d.text((x + 6, HDR - 36), f"{m.get('calligrapher','')}  {tag}", fill=(190, 0, 0), font=f_s)
for i, (st, ss, md, lp, fr) in enumerate(ROWS):
    y = HDR + i * CELL
    d.text((10, y + 10), f"step {int(st)//1000}k", fill=(0, 0, 0), font=f_h)
    d.text((10, y + 32), f"{int(st)*288/1e6:.1f}M samples", fill=(90, 90, 90), font=f_s)
    d.text((10, y + 58), f"LR {2.4e-5 if i>4 else 4.8e-5:.1e}", fill=(150, 150, 150), font=f_s)
    for j, idx in enumerate(IDX):
        p = f"eval/v81sw_{st}/{idx}.png"
        x = PADL + j * CELL
        if os.path.exists(p):
            img.paste(Image.open(p).convert("RGB").resize((CELL, CELL)), (x, y))
        d.rectangle([x, y, x + CELL, y + CELL], outline=(220, 220, 220))
    xr = PADL + CELL * len(IDX) + 8
    d.rectangle([PADL + CELL * len(IDX), y, xr + PADR - 20, y + CELL], outline=(230, 230, 230))
    d.text((xr, y + 22), f"SSIM mean {ss:.4f}", fill=(0, 0, 0), font=f_h)
    d.text((xr, y + 46), f"SSIM median {md:.4f}", fill=(0, 110, 0), font=f_h)
    d.text((xr, y + 70), f"LPIPS {lp:.4f}", fill=(0, 0, 0), font=f_s)
    d.text((xr, y + 90), f"frag {fr:.3f}", fill=(190, 0, 0), font=f_s)
d.text((12, HOR := HDR + CELL * len(ROWS) + 8),
       "SSIM mean 0.5549 -> 0.6284   |   SSIM median 0.6176 -> 0.7367 (typical char improves far more than hard ones)   |   "
       "LPIPS 0.410 -> 0.394   |   frag 1.05 -> 2.18 (WORSE: the condition trades fragmentation for SSIM/LPIPS)",
       fill=(0, 0, 0), font=f_s)
out = "ANALYSIS/v81_ckpt_sweep_poster_20261012.png"
img.save(out)
print(f"  -> {out}  ({img.size[0]}x{img.size[1]})")
