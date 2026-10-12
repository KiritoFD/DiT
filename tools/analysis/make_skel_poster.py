# -*- coding: utf-8 -*-
"""make_skel_poster.py — skel / std_soft 空间条件实验 poster

四块：
  A 视觉对照: GT | v68(无空间条件) | v70(旧 stdskel: 1px→膨胀→VAE latent) | v81(新 std_soft: TTF→64→blur, 不经 VAE)
  B 训练曲线: v81 vs v68（同 batch288/同 200k 调度/同样本, 单变量 = 只差空间条件）
  C 机制证据: 1px 骨架穿过四种 VAE 往返后的保真（我实测）
  D 结论
"""
import csv
import os
import sys

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

COLS = [("gt", "GT"), ("v68", "v68\n(no spatial cond)"),
        ("v70", "v70 (old stdskel)\n1px→dilate→VAE latent"),
        ("v81_poster", "v81 (new std_soft)\nTTF→64→blur, no VAE")]

# A: 6 个条件（覆盖 见/未见 × 简/繁）
IDX = [131, 100, 67, 30, 177, 173]
meta = {int(r["idx"]): r for r in csv.DictReader(
    open("expert_pack/callig_dit_pack_20261011/02_sample_grids/grid_index.csv", encoding="utf-8"))}

# B: v81 vs v68（同 batch 288；取自 eval_auto JSON，训练侧 vgg-LPIPS 口径）


def load_csv(p):
    d = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        try:
            d[int(r["step"])] = {k: (float(r[k]) if r[k] not in ("", None) else None)
                                 for k in ("ssim", "lpips", "frag")}
        except Exception:
            pass
    return d


V68 = load_csv("_local_ckpt/grid48/v68.csv")
V81 = load_csv("_local_ckpt/v81.csv")

# C: 1px 骨架穿过 VAE 往返（skel_branch_probe.py 实测）
PROBE = [("SD-VAE", 0.7935, 13.99), ("Calli-DINO", 0.9019, 4.79),
         ("Calli-topband", 0.9100, 5.64), ("Calli-physics", 0.9515, 3.97)]

fig = plt.figure(figsize=(17, 21))
gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1.0], hspace=0.06, wspace=0.12,
                      left=0.03, right=0.97, top=0.955, bottom=0.03)
axA = fig.add_subplot(gs[0, :]); axA.axis("off")
axA.set_title("A. Spatial-condition comparison  (GT | no cond | old stdskel | new std_soft)",
              fontsize=15, pad=12)
for j, (d, lab) in enumerate(COLS):
    axA.text(0.02 + j * 0.245, 1.005, lab, transform=axA.transAxes, ha="center", va="bottom",
             fontsize=10.5, weight="bold")
for i, idx in enumerate(IDX):
    y = 1 - (i + 1) * (1.0 / len(IDX))
    m = meta.get(idx, {})
    tag = "SEEN" if m.get("combo_seen") in ("True", True) else "UNSEEN"
    axA.text(0.012, y + 0.5 / len(IDX), f"#{idx}  {m.get('calligrapher','')}\n{m.get('script','')}/"
             f"{m.get('character','')}\n{tag}  ink={m.get('gt_ink','')}",
             transform=axA.transAxes, ha="left", va="center", fontsize=9)
    for j, (d, _) in enumerate(COLS):
        p = f"eval/{d}/{idx}.png"
        if not os.path.exists(p):
            continue
        im = Image.open(p).convert("RGB").resize((224, 224))
        axA.imshow(np.asarray(im), extent=(0.09 + j * 0.225, 0.09 + j * 0.225 + 0.215,
                                           y, y + 0.985 / len(IDX)), aspect="auto")
axA.set_xlim(0, 1); axA.set_ylim(0, 1)

axB = fig.add_subplot(gs[1, 0])
axC = fig.add_subplot(gs[1, 1])

steps = sorted(set(V68) & set(V81))
axB.plot(steps, [V68[s]["ssim"] for s in steps], "o-", label="v68 SSIM (no cond)", ms=3)
axB.plot(steps, [V81[s]["ssim"] for s in steps], "s-", label="v81 SSIM (+std_soft)", ms=3)
ax2 = axB.twinx()
ax2.plot(steps, [V68[s]["frag"] or 0 for s in steps], "--", color="gray", label="v68 frag")
ax2.plot(steps, [V81[s]["frag"] or 0 for s in steps], "--", color="orange", label="v81 frag")
axB.set_xlabel("step  (both batch=288, 200k cosine ⇒ matched samples)")
axB.set_ylabel("SSIM"); ax2.set_ylabel("frag")
axB.set_title("B. v81 vs v68 — single variable = spatial condition only\n"
              "ΔSSIM avg +0.0030 (peak +0.0089 @70k) | ΔLPIPS avg −0.0023 | frag: v81 worse",
              fontsize=10)
h1, l1 = axB.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
axB.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")

names = [p[0] for p in PROBE]
x = np.arange(len(names))
axC.bar(x - 0.2, [p[1] for p in PROBE], 0.4, label="skeleton IoU (glyph round-trip)")
axC.bar(x + 0.2, [p[2] / 20 for p in PROBE], 0.4, label="connectivity blow-up ÷20 (1px skel round-trip)")
for i, p in enumerate(PROBE):
    axC.text(i - 0.2, p[1] + 0.01, f"{p[1]:.3f}", ha="center", fontsize=8)
    axC.text(i + 0.2, p[2] / 20 + 0.01, f"{p[2]:.2f}×", ha="center", fontsize=8)
axC.set_xticks(x); axC.set_xticklabels(names, fontsize=8)
axC.set_title("C. Why 'skeleton' can't survive the 4ch/8x latent (measured)\n"
              "1px skeleton in → decode: connectivity 11.5 → 46~161 pieces", fontsize=10)
axC.legend(fontsize=8)

fig.suptitle("Sketch/soft spatial-condition experiment — Callig-DiT (2026-10-12)  |  "
             "v81 mid-run @110k, batch 288 vs v68 batch 288", fontsize=15, y=0.975)
out = "ANALYSIS/skel_experiment_poster_20261012.png"
fig.savefig(out, dpi=105)
print(f"  -> {out}")
