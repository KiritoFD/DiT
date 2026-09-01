# -*- coding: utf-8 -*-
"""build_std_glyph_bbox.py — 预计算标准字形 kai 墨 bbox 映射。

对 std_skeleton_d3/kai/ 下每张 codepoint 图, 计算其前景(墨) bbox,
保存为 codepoint -> (y0, y1, x0, x1) 的 dict, 用于方案 B 清洗时
判断"真迹墨是否在字形 bbox 之外"（= 脏污候选）。

输出: _sync_work/std_glyph_bbox.json
      { codepoint_str: [y0, y1, x0, x1, w, h], ... }
"""
import os
import sys
import json
import glob

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")

import numpy as np
from PIL import Image

SRC = "std_skeleton_d3/kai"
OUT = "_sync_work/std_glyph_bbox.json"


def main():
    files = sorted(glob.glob(os.path.join(SRC, "U+*.png")))
    print(f"found {len(files)} glyphs in {SRC}")
    bbox = {}
    for f in files:
        cp_str = os.path.basename(f)[2:].replace(".png", "")
        try:
            cp = int(cp_str, 16)
        except ValueError:
            continue
        try:
            a = np.asarray(Image.open(f).convert("L"), dtype=np.uint8)
        except Exception:
            continue
        ink = a < 128
        if not ink.any():
            continue
        ys, xs = np.where(ink)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        # key 用十进制 codepoint 字符串, 便于 ord(character) 直接查
        bbox[str(cp)] = [y0, y1, x0, x1, a.shape[1], a.shape[0]]  # x1..w, y1..h

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(bbox, f, ensure_ascii=False)
    print(f"saved {len(bbox)} bboxes -> {OUT}")


if __name__ == "__main__":
    main()
