# -*- coding: utf-8 -*-
"""build_std_skel_px60.py — 重建 fame-kxl-tj-px60 的标准字 g (骨架形态, 严格复刻配方).

背景 (2026-09-15, 严重问题):
  上一版把 g 建成了**填充字形位图** (ink≈0.185), 但配方要的是**标准字骨架**:
    tools/gen_base_images.py:render()  ->  字号 200 -> 居中(anchor="mm", **不做 bbox 归一化**)
      -> skeletonize(a<127) -> binary_dilation(8邻域, iterations=1)  [≈3px 白底黑线]
  实测旧标准字骨架 ink≈0.033, 与填充位图差 5.6 倍 —— 条件域完全不同。
  另: 旧版还额外做了 bbox 裁剪+缩放到 0.88*256, 几何也与配方不一致。

本脚本产出:
  data/fame-kxl-tj-px60/std/{train_img_id}.png        训练 g (覆盖旧的填充版)
  data/fame-kxl-tj-px60/std_eval/{eval_img_id}.png    评测 g (seen + strict, 按各自 img_id)
  data/fame-kxl-tj-px60/_preview/std_skel_cmp.png     填充 vs 骨架 目视对照

用法: python tools/build_std_skel_px60.py [--apply]
"""
import argparse
import csv
import multiprocessing as mp
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, generate_binary_structure

try:
    from skimage.morphology import skeletonize
except ImportError:                                   # 与配方同款 fallback
    from scipy.ndimage import binary_erosion

    def skeletonize(b):
        skel = np.zeros_like(b)
        img = b.copy()
        st = generate_binary_structure(2, 2)
        while img.any():
            er = binary_erosion(img, structure=st)
            skel |= img & ~er
            img = er
        return skel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NAME = "fame-kxl-tj-px60"
TRAIN_CSV = f"assets/train_{NAME}.csv"
EVAL_CSVS = ("assets/eval_seen_v10.csv", "assets/eval_fame3_strict_clean_v9.csv")
TRAIN_STD = f"data/{NAME}/std"
EVAL_STD = f"data/{NAME}/std_eval"
PREVIEW = f"data/{NAME}/_preview/std_skel_cmp.png"

FONT_DIR = "tools/fonts"
FONT_SIZE = 200
ST = generate_binary_structure(2, 2)
SCRIPT_FONT = {
    "楷": ["simkai.ttf", "STKAITI.TTF", "NotoSerifSC-VF.ttf"],
    "行": ["STXINGKA.TTF", "FZSTK.TTF"],
    "隶": ["SIMLI.TTF", "STLITI.TTF"],
}
_FC = {}


def render_skel(ch, script):
    """严格复刻 tools/gen_base_images.py:render() —— 渲染 -> 骨架 -> 3px."""
    cands = list(SCRIPT_FONT.get(script, SCRIPT_FONT["楷"])) + ["simkai.ttf", "simhei.ttf"]
    seen = set()
    for f in cands:
        if f in seen:
            continue
        seen.add(f)
        fp = os.path.join(FONT_DIR, f)
        if not os.path.isfile(fp):
            continue
        try:
            font = _FC.get(f) or ImageFont.truetype(fp, FONT_SIZE)
            _FC[f] = font
        except Exception:
            continue
        img = Image.new("L", (256, 256), 255)
        ImageDraw.Draw(img).text((128, 128), ch, font=font, fill=0, anchor="mm")
        a = np.asarray(img)
        if (a < 250).sum() < 10:
            continue
        sk = skeletonize(a < 127)
        sk = binary_dilation(sk, ST, iterations=1)
        return np.where(sk, 0, 255).astype(np.uint8)
    return None


def _w(key):
    return key, render_skel(key[1], key[0])


def render_old_filled(ch, script):
    """上一版的错误实现 (填充字形 + bbox 归一化), 仅用于目视对照."""
    for f in list(SCRIPT_FONT.get(script, SCRIPT_FONT["楷"])) + ["simkai.ttf", "simhei.ttf"]:
        fp = os.path.join(FONT_DIR, f)
        if not os.path.isfile(fp):
            continue
        try:
            font = ImageFont.truetype(fp, 200)
        except Exception:
            continue
        img = Image.new("L", (256, 256), 255)
        ImageDraw.Draw(img).text((128, 128), ch, font=font, fill=0, anchor="mm")
        a = np.asarray(img)
        if (a < 250).sum() < 10:
            continue
        m = a < 250
        ys, xs = np.where(m)
        crop = Image.fromarray(a).crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        t = 0.88 * 256
        s = t / max(crop.size)
        tw, th = max(int(crop.size[0] * s), 1), max(int(crop.size[1] * s), 1)
        crop = crop.resize((tw, th), Image.LANCZOS)
        cv = Image.new("L", (256, 256), 255)
        cv.paste(crop, ((256 - tw) // 2, (256 - th) // 2))
        return np.asarray(cv)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    train = list(csv.DictReader(open(TRAIN_CSV, encoding="utf-8")))
    evals = {p: list(csv.DictReader(open(p, encoding="utf-8"))) for p in EVAL_CSVS}
    print(f"[{NAME}] train {len(train)} 行; "
          + "; ".join(f"{os.path.basename(p)} {len(v)}" for p, v in evals.items()))

    keys = sorted({(r["script"], r["character"]) for r in train})
    for rs in evals.values():
        keys = sorted(set(keys) | {(r["script"], r["character"]) for r in rs})
    print(f"  需要渲染 (script,char): {len(keys)}  (train "
          f"{len({(r['script'], r['character']) for r in train})}, "
          f"eval 新增 {len(keys) - len({(r['script'], r['character']) for r in train})})")

    with mp.Pool(32) as pool:
        res = pool.map(_w, keys, chunksize=16)
    k2a = {k: v for k, v in res if v is not None}
    miss = [k for k, v in res if v is None]
    print(f"  渲染成功 {len(k2a)}, 失败 {len(miss)} {miss[:10]}")

    def ids_missing(rows):
        return [r for r in rows
                if (r["script"], r["character"]) not in k2a]

    mt = ids_missing(train)
    print(f"  train 无 g 的行: {len(mt)}")
    for p, rs in evals.items():
        mm_ = ids_missing(rs)
        print(f"  {os.path.basename(p)} 无 g 的行: {len(mm_)}"
              + (f"  例: {[(r['script'], r['character']) for r in mm_[:5]]}" if mm_ else ""))

    inks = np.array([float((v < 128).mean()) for v in k2a.values()])
    print(f"\n  ★ 新 g 的 ink: p1={np.percentile(inks,1):.4f} p50={np.percentile(inks,50):.4f} "
          f"p99={np.percentile(inks,99):.4f}  (配方实测 ≈0.033, 旧填充版 ≈0.185)")

    # 目视对照: 填充 vs 骨架
    os.makedirs(os.path.dirname(PREVIEW), exist_ok=True)
    samp = keys[:12]
    cell = 150
    cv = Image.new("RGB", (cell * 2 + 12, (cell + 24) * len(samp)), (255, 255, 255))
    d = ImageDraw.Draw(cv)
    from PIL import ImageFont as _IF
    f = _IF.truetype("tools/fonts/simhei.ttf", 18)
    for i, k in enumerate(samp):
        y = i * (cell + 24)
        old = render_old_filled(k[1], k[0])
        if old is not None:
            cv.paste(Image.fromarray(old).resize((cell, cell)).convert("RGB"), (0, y))
        cv.paste(Image.fromarray(k2a[k]).resize((cell, cell)).convert("RGB"), (cell + 12, y))
        d.text((4, y + cell + 2), f"{k[1]}  填充(旧/错)", fill=(150, 0, 0), font=f)
        d.text((cell + 16, y + cell + 2), f"{k[1]}  骨架(新/对)", fill=(0, 110, 0), font=f)
    cv.save(PREVIEW)
    print(f"  对照图 -> {PREVIEW}")

    if not a.apply:
        print("\n[DRY-RUN] 加 --apply 执行。")
        return

    os.makedirs(TRAIN_STD, exist_ok=True)
    os.makedirs(EVAL_STD, exist_ok=True)
    for rows, out in ((train, TRAIN_STD),
                      *((v, EVAL_STD) for v in evals.values())):
        n = 0
        for r in rows:
            arr = k2a.get((r["script"], r["character"]))
            if arr is None:
                continue
            iid = os.path.basename(r["image_path"])[:-4]
            Image.fromarray(arr, "L").save(f"{out}/{iid}.png")
            n += 1
        print(f"[apply] {out}: {n} png")
    print(f"[apply] 统计 {dict(Counter(os.path.basename(p) for p in EVAL_CSVS))}")


if __name__ == "__main__":
    main()
