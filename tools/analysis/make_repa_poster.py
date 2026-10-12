# -*- coding: utf-8 -*-
"""make_repa_poster.py — A0(REPA=0) vs v68(REPA=0.03) 匹配步数视觉对照 + 全指标符号检验

图源（零生成, 均为训练时自己保存的评测图）:
  _local_ckpt/evals_cmp/v68/step<XXXXXXX>/g<idx>.png   (REPA 0.03)
  _local_ckpt/evals_cmp/A0/step<XXXXXXX>/g<idx>.png    (REPA 0)
  _local_ckpt/evals_cmp/*/gt<idx>.png                  (同一份真迹, 两边应一致 -> 顺带核对映射)
布局: 行=字(8 个, 含 见/未见 × 简/繁), 列= GT | (5k,20k,50k,85k) × (v68, A0)
底部: 全指标符号检验结论
"""
import csv
import os
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IDX = [131, 100, 86, 67, 30, 177, 173, 22]
STEPS = ["0005000", "0020000", "0050000", "0085000"]
META = {int(r["idx"]): r for r in csv.DictReader(
    open("expert_pack/callig_dit_pack_20261011/02_sample_grids/grid_index.csv", encoding="utf-8"))}
# 符号检验结果（A0 − v68；由 tools/analysis/cmp_repa.py 得出）
SIG = [("ink_iou", -0.00117, "5/17", 0.975, "A0 显著更差 ✗"),
       ("hole", +0.00339, "5/17", 0.975, "A0 显著更差 ✗"),
       ("nn_ssim", -0.00060, "5/17", 0.975, "A0 显著更差 ✗"),
       ("frag", +0.04707, "6/17", 0.928, "偏 A0 更差(不显著)"),
       ("skel_iou", -0.00004, "6/17", 0.928, "偏 A0 更差(不显著)"),
       ("ssim", -0.00009, "9/17", 0.500, "掷硬币 = 无差异"),
       ("lpips", +0.00038, "8/17", 0.685, "掷硬币 = 无差异"),
       ("mse", -0.00279, "11/17", 0.166, "偏 A0 更好(不显著)"),
       ("ink_ssim", +0.00080, "12/17", 0.072, "偏 A0 更好(不显著)"),
       ("tgt_spec", +0.00033, "12/17", 0.072, "偏 A0 更好(不显著)")]

CELL, PADL, HDR, FTR = 128, 150, 108, 250
NCOL = 1 + len(STEPS) * 2
W = PADL + CELL * NCOL + 16
H = HDR + CELL * len(IDX) + FTR
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


f_t, f_h, f_s = font(21, True), font(14), font(12)
d.text((10, 10), "REPA ablation on the v68 recipe — A0 (w_repa=0) vs v68 (w_repa=0.03)",
       fill=(0, 0, 0), font=f_t)
d.text((10, 38), "both: same data (aug_sym 78k) / SD-VAE shards / batch288 / 200k cosine / C2OT ON / "
                 "same eval rows, same saved images (no re-generation)", fill=(90, 90, 90), font=f_s)
# 表头
d.text((10, HDR - 34), "GT", fill=(0, 0, 0), font=f_h)
for j, st in enumerate(STEPS):
    x = PADL + (1 + j * 2) * CELL
    d.text((x + 4, HDR - 56), f"step {int(st)//1000}k", fill=(0, 0, 0), font=f_h)
    d.text((x + 4, HDR - 36), "v68", fill=(0, 90, 200), font=f_s)
    d.text((x + CELL + 4, HDR - 36), "A0", fill=(200, 0, 0), font=f_s)
for i, idx in enumerate(IDX):
    y = HDR + i * CELL
    m = META.get(idx, {})
    tag = "SEEN" if m.get("combo_seen") in ("True", True) else "UNSEEN"
    d.text((8, y + 10), f"#{idx}", fill=(0, 0, 0), font=f_h)
    d.text((8, y + 30), f"{m.get('calligrapher','')}", fill=(60, 60, 60), font=f_s)
    d.text((8, y + 48), f"{m.get('script','')}/{m.get('character','')}", fill=(0, 0, 0), font=f_s)
    d.text((8, y + 68), tag, fill=(190, 0, 0), font=f_s)
    d.text((8, y + 86), f"GTink {m.get('gt_ink','')}", fill=(140, 140, 140), font=f_s)
    p = f"_local_ckpt/evals_cmp/v68/gt{idx}.png"
    if os.path.exists(p):
        img.paste(Image.open(p).convert("RGB").resize((CELL, CELL)), (PADL - CELL // 2 + 12, y))
    for j, st in enumerate(STEPS):
        for k, mdl in enumerate(("v68", "A0")):
            q = f"_local_ckpt/evals_cmp/{mdl}/step{st}/g{idx}.png"
            x = PADL + (1 + j * 2 + k) * CELL
            if os.path.exists(q):
                img.paste(Image.open(q).convert("RGB").resize((CELL, CELL)), (x, y))
            d.rectangle([x, y, x + CELL, y + CELL], outline=(215, 215, 215))
d.line([(0, HDR + CELL * len(IDX)), (W, HDR + CELL * len(IDX))], fill=(150, 150, 150))
y0 = HDR + CELL * len(IDX) + 8
d.text((10, y0), "All-metric paired sign test over 17 matched eval points (5k..85k) — "
                 "Δ = A0 − v68, p = one-sided binomial", fill=(0, 0, 0), font=f_h)
yy = y0 + 24
for m, dm, w, p, v in SIG:
    col = (200, 0, 0) if "更差" in v else ((0, 130, 0) if "更好" in v else (110, 110, 110))
    d.text((14, yy), f"{m:<10}  Δ={dm:+.5f}   A0 wins {w:<6} p={p:.3f}   {v}",
           fill=col, font=f_s)
    yy += 19
out = "ANALYSIS/repa_ablation_A0_vs_v68.png"
img.save(out)
print(f"  -> {out}  ({img.size[0]}x{img.size[1]})")
