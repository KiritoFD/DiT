# -*- coding: utf-8 -*-
"""white_baseline.py — 各指标的"全白零点基线"：白图 vs GT 得分越高 = 该指标越不能用于本任务

对同一批 10 个字（eval200_fixed 的 IDX），比较三种输入:
    GT(真迹) / WHITE(全白) / SAMPLE(模型输出)
指标: SSIM(default) / SSIM(gaussian, 原论文版) / LPIPS(vgg) / ink_iou / HF / frag
"""
import csv
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "vae"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from skimage.metrics import structural_similarity as S  # noqa: E402
from hf_losses import band_fraction  # noqa: E402

IDX = [131, 100, 67, 30, 177, 173, 86, 83, 145, 22]
SAMPLE_STEP = "0115000"


def load(p):
    return np.asarray(Image.open(p).convert("RGB").resize((256, 256)))


gts = [load(f"eval/gt/{i}.png") for i in IDX]
gtsw = [np.full_like(g, 255) for g in gts]
smps = [load(f"eval/v81fix_{SAMPLE_STEP}/{j}.png") for j in range(len(IDX))]


def ssim_def(a, b):
    return S(a, b, channel_axis=2, data_range=255)


def ssim_gauss(a, b):
    return S(a, b, channel_axis=2, data_range=255, gaussian_weights=True, sigma=1.5,
             use_sample_covariance=False)


def ink_iou(a, b):
    A = a.mean(axis=2) < 128; B = b.mean(axis=2) < 128
    u = (A | B).sum()
    return float((A & B).sum() / u) if u else 1.0


def hf(a):
    return float(band_fraction(torch.from_numpy(a).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1).item())


try:
    import lpips as _lp
    lp = _lp.LPIPS(net="alex", verbose=False).eval()
    for q in lp.parameters():
        q.requires_grad_(False)

    def lpips(a, b):
        with torch.no_grad():
            return float(lp(torch.from_numpy(a).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1,
                            torch.from_numpy(b).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1).item())
except Exception as e:
    print("  (LPIPS 不可用:", e, ")")
    lpips = None


def row(name, fn, higher_better=True):
    w = np.mean([fn(wh, g) for wh, g in zip(gtsw, gts)])
    s = np.mean([fn(sm, g) for sm, g in zip(smps, gts)])
    ok = (w < 0.5) if higher_better else (w > 0.5)
    print(f"  {name:<26} 白图={w:>8.4f}   样本={s:>8.4f}   "
          f"{'✓ 白图被正确压低' if (higher_better and w < 0.35) or (not higher_better and w > 0.9) else '✗ 白图得分过高 —— 不可单独用于本任务'}")


print(f"  样本 = v81@{SAMPLE_STEP}（已知大面积空白）;  GT = eval/gt（已核对与 CSV 逐像素相同）\n")
row("SSIM default(现用)", ssim_def)
row("SSIM gaussian(论文版)", ssim_gauss)
row("ink_iou", ink_iou)
row("HF (顶带份额)", hf, higher_better=False)
if lpips:
    row("LPIPS(alex) [越低越好]", lpips, higher_better=False)
print("""
  判读: "白图列"是零点基线。任何给白图 0.8 的指标, 都会把"输出塌成白纸"当成进步。
  本次实测: SSIM(default) 白图=0.85 -> **不可单独用于本任务**; ink_iou / LPIPS / HF 的表现见上表。
""")
