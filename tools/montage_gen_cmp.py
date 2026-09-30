# -*- coding: utf-8 -*-
"""montage_gen_cmp.py — 生成图三列对照: predskel条件 / GT骨架(oracle) / GT真迹。

纯目视, 回答"diffusion 跑出来是不是白的"。
用法: python tools/montage_gen_cmp.py --n 8 --out /tmp/gen_cmp.png
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ink(p):
    a = np.asarray(Image.open(p).convert("L"))
    return float((a < 128).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="assets/results/_calib_skel/eval_samples_ctrl/step0090001/strict_pred")
    ap.add_argument("--oracle", default="assets/results/_calib_skel/eval_samples_ctrl/step0010001/strict")
    ap.add_argument("--out", default="/tmp/gen_cmp.png")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--cell", type=int, default=160)
    a = ap.parse_args()

    n = a.n
    C = a.cell
    lab = 22
    cv = Image.new("RGB", (C * 3 + 24, 30 + n * (C + lab)), (245, 245, 245))
    d = ImageDraw.Draw(cv)
    d.text((6, 5), "左=predskel条件  中=GT骨架(oracle)  右=GT真迹", fill=(0, 0, 0))
    for i in range(n):
        y = 30 + i * (C + lab)
        cols = [(a.pred, f"g{i}.png"), (a.oracle, f"g{i}.png"), (a.oracle, f"gt{i}.png")]
        for j, (base, fn) in enumerate(cols):
            p = os.path.join(base, fn)
            x = 6 + j * (C + 6)
            if os.path.exists(p):
                cv.paste(Image.open(p).convert("RGB").resize((C, C)), (x, y))
                d.text((x, y + C + 3), f"ink={ink(p):.4f}", fill=(80, 80, 80))
            else:
                d.text((x, y + C // 2), f"缺 {fn}", fill=(200, 0, 0))
    cv.save(a.out)
    print("->", a.out)


if __name__ == "__main__":
    main()
