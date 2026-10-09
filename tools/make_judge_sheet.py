# -*- coding: utf-8 -*-
"""make_judge_sheet.py — 生成"评审用"大图: 每格按**原始 256px** 贴, 便于人工视觉打分。

海报里每格只有 190px(还要被整体缩放), 细节看不清。本工具按任意模型子集出图,
行 = 模型, 列 = 同一批样本(与海报同一套等距选样), 供肉眼比较笔锋/断笔/糊块/浮噪。

用法:
  python tools/make_judge_sheet.py --tier mid --models v68,v_b_aug_route_60k,v66,moyi_4ch --n 4
"""
import argparse
import csv
import os
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SPLIT = {"best": "exp-std/csv/eval200_split_top.csv",
         "mid": "exp-std/csv/eval200_split_mid.csv",
         "worst": "exp-std/csv/eval200_split_worst.csv"}
PER_SAMPLE = "assets/eval200fix_models_per_sample.csv"


def font(sz, bold=True):
    for p in ("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
              "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="mid", choices=list(SPLIT))
    ap.add_argument("--models", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--cell", type=int, default=256)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    ids = {}
    with open("exp-std/csv/eval200_fixed.csv", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            ids[r["image_path"]] = i
    with open(SPLIT[a.tier], encoding="utf-8") as f:
        tier_idx = [ids[r["image_path"]] for r in csv.DictReader(f)]

    per = {int(r["idx"]): r for r in csv.DictReader(open(PER_SAMPLE, encoding="utf-8"))
           if r["model_name"] == "v68"}
    ranked = sorted(tier_idx, key=lambda i: -float(per[i]["ssim"]))
    n = len(ranked)
    pos = sorted({round(k * (n - 1) / max(1, a.n - 1)) for k in range(a.n)})
    cols = [ranked[p] for p in pos]

    models = ["gt"] + [m.strip() for m in a.models.split(",") if m.strip()]
    C, PAD = a.cell, 10
    LBL = 190
    W = LBL + len(cols) * (C + PAD) + PAD
    H = 60 + len(models) * (C + PAD) + PAD
    img = Image.new("RGB", (W, H), "#0b0f16")
    d = ImageDraw.Draw(img)
    f_t = font(24)
    f_l = font(17)
    f_s = font(14)

    d.text((PAD, 12), f"评审图 · {a.tier} 档 · 每格原图 {C}px (未缩放) · 每格下方=该样本SSIM",
           font=f_t, fill="#f8fafc")
    for ci, idx in enumerate(cols):
        x = LBL + ci * (C + PAD)
        m = per[idx]
        d.text((x, 42), f"#{idx:03d} {m['calligrapher']}·{m['script']}·{m['char']}",
               font=f_s, fill="#94a3b8")

    for ri, mname in enumerate(models):
        y = 60 + ri * (C + PAD)
        d.text((PAD, y + C // 2 - 24), mname, font=f_l, fill="#facc15" if mname == "gt" else "#f8fafc")
        for ci, idx in enumerate(cols):
            x = LBL + ci * (C + PAD)
            p = os.path.join("eval", mname, f"{idx}.png")
            d.rectangle([(x, y), (x + C, y + C)], outline="#1e293b")
            if os.path.exists(p):
                img.paste(Image.open(p).convert("RGB"), (x, y))
        # 逐格 SSIM
    # 逐格数值另起一行标在格下不方便, 改为在右侧标签里标该模型本档均值
    sm = {r["model"]: r for r in csv.DictReader(
        open("assets/eval200_splits_exact_metrics.csv", encoding="utf-8"))}
    for ri, mname in enumerate(models):
        if mname in sm:
            r = sm[mname]
            y = 60 + ri * (C + PAD)
            d.text((PAD, y + C // 2 + 2), f"SSIM {float(r['all_ssim']):.4f}",
                   font=f_s, fill="#38bdf8")
            d.text((PAD, y + C // 2 + 20), f"IoU  {float(r['all_iou']):.4f}",
                   font=f_s, fill="#f59e0b")

    out = a.out or f"_sync_work/judge_{a.tier}.png"
    img.save(out)
    print(f"✓ {out}  ({W}×{H})")


if __name__ == "__main__":
    main()
