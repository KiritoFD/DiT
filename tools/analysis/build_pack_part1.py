# -*- coding: utf-8 -*-
"""build_pack_part1.py — 专家包 第1项：真实生成结果对比图（GT / v68 / v79）

覆盖：
  * 书体：楷 / 行 / 隶
  * 组合：训练**见过** vs **未见过**（用 train_top10_noaug.csv 判定）
  * 难度：用 GT 墨量占比作"简单/复杂"代理（低墨量=简, 高墨量=繁）
每个书体出一张海报（每行 = 一个样本, 列 = GT / v68 / v79），并列 index CSV。

⚠ 种子说明: 评测缓存的初始噪声是 `torch.Generator().manual_seed(0)` **硬编码**
  （src/eval/inference.py:609）→ 所有模型/所有行共用同一条噪声。
  这对**模型间对比**是优点（隔离模型差异），但**无法给"3 个不同种子"** —— 已如实标注。
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PACK = "expert_pack/callig_dit_pack_20261011"
GDIR = os.path.join(PACK, "02_sample_grids")
os.makedirs(GDIR, exist_ok=True)

MODELS = [("GT", "eval/gt"), ("v68", "eval/v68"), ("v79", "eval/v79")]
ROW_H = 256
LBL_W = 190
HDR_H = 34


def font(sz):
    for f in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
        if os.path.exists(f):
            try:
                return ImageFont.truetype(f, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def ink_frac(p):
    a = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
    return float((a < 128).mean())


def main():
    rows = list(csv.DictReader(open("exp-std/csv/eval200_fixed.csv", encoding="utf-8")))
    tpath = next(p for p in ("exp-std/csv/train_top10_noaug.csv", "exp-std/csv/train.csv")
                 if os.path.exists(p))
    train = list(csv.DictReader(open(tpath, encoding="utf-8")))
    print(f"  训练 CSV = {tpath}")
    seen = {(r["calligrapher"], r["script"], r["character"]) for r in train}
    print(f"  评测 {len(rows)} 行; 训练(noaug) {len(train)} 行; 训练组合数 {len(seen)}")

    by_script = defaultdict(list)
    for i, r in enumerate(rows):
        p = f"eval/gt/{i}.png"
        if not os.path.exists(p):
            continue
        r["_idx"] = i
        r["_ink"] = ink_frac(p)
        r["_seen"] = (r["calligrapher"], r["script"], r["character"]) in seen
        by_script[r["script"]].append(r)

    index = []
    for script, lst in sorted(by_script.items(), key=lambda kv: -len(kv[1])):
        picked = []
        for sn in (True, False):
            grp = sorted([r for r in lst if r["_seen"] == sn], key=lambda r: r["_ink"])
            if not grp:
                continue
            # 每档取 最简 2 + 最繁 2（覆盖难度两端）
            picked += [grp[0], grp[1], grp[-2], grp[-1]]
        if not picked:
            continue
        n = len(picked)
        W = LBL_W + 3 * 256
        canvas = Image.new("RGB", (W, HDR_H + n * ROW_H), (255, 255, 255))
        d = ImageDraw.Draw(canvas)
        f_h, f_s = font(17), font(15)
        for c, (name, _) in enumerate(MODELS):
            d.text((LBL_W + c * 256 + 8, 9), name, fill=(0, 0, 0), font=f_h)
        for j, r in enumerate(picked):
            y = HDR_H + j * ROW_H
            tag = "SEEN" if r["_seen"] else "UNSEEN"
            d.text((8, y + 8), f"#{r['_idx']} id={r['img_id']}", fill=(0, 0, 0), font=f_s)
            d.text((8, y + 28), f"{r['calligrapher']}", fill=(0, 0, 0), font=f_s)
            d.text((8, y + 48), f"{r['script']} / {r['character']}", fill=(0, 0, 0), font=f_s)
            d.text((8, y + 72), f"{tag}  ink={r['_ink']:.3f}", fill=(180, 0, 0), font=f_s)
            for c, (name, dd) in enumerate(MODELS):
                p = f"{dd}/{r['_idx']}.png"
                if os.path.exists(p):
                    canvas.paste(Image.open(p).convert("RGB").resize((256, 256)), (LBL_W + c * 256, y))
            d.line([(0, y), (W, y)], fill=(200, 200, 200))
            index.append(dict(grid=f"grid_{script}.png", row=j, idx=r["_idx"], img_id=r["img_id"],
                              calligrapher=r["calligrapher"], script=r["script"], character=r["character"],
                              combo_seen=r["_seen"], gt_ink=round(r["_ink"], 4)))
        out = os.path.join(GDIR, f"grid_{script}.png")
        canvas.save(out)
        print(f"  {out}  ({n} 行 × 3 列)")

    with open(os.path.join(PACK, "02_sample_grids", "grid_index.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(index[0].keys()))
        w.writeheader(); w.writerows(index)
    print(f"  index -> 02_sample_grids/grid_index.csv ({len(index)} 行)")


if __name__ == "__main__":
    main()
