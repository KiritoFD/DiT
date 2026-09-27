#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/analysis/test_salvage_skel_render.py — 测试 MCCD 4421 张待打捞样本的标准骨架渲染通过率"""
import os
import sys
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize
from scipy.ndimage import binary_dilation, generate_binary_structure

sys.stdout.reconfigure(encoding="utf-8")

CAND_CSV = "assets/mccd_salvage_candidates.csv"
FONT_DIR = "tools/fonts"
CAND_FONT_DIR = "tools/fonts/candidate_fonts"
SIZE = 256
FONT_SIZE = 200
ST = generate_binary_structure(2, 2)

FONTS = {
    "楷": [
        os.path.join(FONT_DIR, "simkai.ttf"),
        os.path.join(CAND_FONT_DIR, "liugongquan_kaishu.ttf"),
        os.path.join(CAND_FONT_DIR, "yanzhenqing_kaishu.ttf"),
        os.path.join(FONT_DIR, "SimHei.ttf")
    ],
    "行": [
        os.path.join(FONT_DIR, "STXINGKA.TTF"),
        os.path.join(CAND_FONT_DIR, "ZhiMangXing-Regular.ttf"),
        os.path.join(CAND_FONT_DIR, "LongCang-Regular.ttf"),
        os.path.join(CAND_FONT_DIR, "FZSTK.TTF"),
        os.path.join(FONT_DIR, "simkai.ttf")
    ],
    "隶": [
        os.path.join(FONT_DIR, "SIMLI.TTF"),
        os.path.join(CAND_FONT_DIR, "SIMLI.TTF"),
        os.path.join(FONT_DIR, "simkai.ttf")
    ]
}

_fc = {}

def get_font(path):
    if path not in _fc:
        if os.path.exists(path):
            try:
                _fc[path] = ImageFont.truetype(path, FONT_SIZE)
            except Exception:
                _fc[path] = None
        else:
            _fc[path] = None
    return _fc[path]

def render_skel(ch, script):
    cands = FONTS.get(script, [])
    for fp in cands:
        font = get_font(fp)
        if font is None:
            continue
        im = Image.new("L", (SIZE, SIZE), 255)
        try:
            ImageDraw.Draw(im).text((SIZE // 2, SIZE // 2), ch, font=font, fill=0, anchor="mm")
        except Exception:
            continue
        arr = np.asarray(im)
        if (arr < 250).sum() < 10:
            continue
        sk = skeletonize(arr < 127)
        if not sk.any():
            continue
        sk = binary_dilation(sk, ST, iterations=1)
        return True, os.path.basename(fp)
    return False, None

def main():
    df = pd.read_csv(CAND_CSV)
    print(f"开始测试 {len(df)} 张待打捞样本的骨架可渲染性...")
    
    # 按 (script, character) 去重测试
    unique_pairs = df[["script", "character"]].drop_duplicates()
    print(f"独立 (书体, 汉字) 对: {len(unique_pairs)} 个")

    ok_map = {}
    font_used = {}
    for idx, row in unique_pairs.iterrows():
        s, ch = row["script"], row["character"]
        ok, fn = render_skel(ch, s)
        ok_map[(s, ch)] = ok
        if ok:
            font_used[(s, ch)] = fn

    success_cnt = sum(1 for _, row in df.iterrows() if ok_map.get((row["script"], row["character"]), False))
    print(f"\n【骨架渲染测试结果】:")
    print(f"  - 总候选样本: {len(df)} 张")
    print(f"  - 成功渲染骨架: {success_cnt} 张 ({success_cnt/len(df)*100:.2f}%)")
    print(f"  - 失败样本数: {len(df) - success_cnt} 张")

    print("\n分书体成功率:")
    for s in ["楷", "行", "隶"]:
        sub = df[df["script"] == s]
        s_ok = sum(1 for _, row in sub.iterrows() if ok_map.get((row["script"], row["character"]), False))
        print(f"  - {s}书: {s_ok}/{len(sub)} ({s_ok/len(sub)*100:.1f}%)")

if __name__ == "__main__":
    main()
