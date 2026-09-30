# -*- coding: utf-8 -*-
"""montage_input_g.py — 把「喂给主干的骨架」本身拼成对照图 (pred vs GT)。

目的: 直接看 predskel 是不是碎的 —— 不用任何指标, 纯目视。
用法: python tools/montage_input_g.py --out /tmp/in_g_cmp.png --n 12
"""
import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="assets/results/_calib_skel/eval_samples_ctrl/strict_pred_input_g")
    ap.add_argument("--gt", default="assets/results/_calib_skel/eval_samples_ctrl/strict_input_g")
    ap.add_argument("--out", default="/tmp/in_g_cmp.png")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--cell", type=int, default=150)
    a = ap.parse_args()

    pf = sorted(glob.glob(os.path.join(a.pred, "*.png")))
    gf = sorted(glob.glob(os.path.join(a.gt, "*.png")))
    print(f"pred {len(pf)} 张, gt {len(gf)} 张")
    n = min(a.n, len(pf), len(gf))
    if n == 0:
        print("无文件"); return
    C = a.cell
    cv = Image.new("RGB", (C * 2 + 16, n * (C + 22) + 30), (245, 245, 245))
    d = ImageDraw.Draw(cv)
    d.text((6, 4), "输入骨架: 左=SkelNet预测(pred)   右=GT骨架", fill=(0, 0, 0))
    for i in range(n):
        y = 26 + i * (C + 22)
        cv.paste(Image.open(pf[i]).convert("RGB").resize((C, C)), (4, y))
        cv.paste(Image.open(gf[i]).convert("RGB").resize((C, C)), (C + 12, y))
        # 前景占比 (墨=暗)
        for j, p in enumerate((pf[i], gf[i])):
            arr = np.asarray(Image.open(p).convert("L"))
            fg = float((arr < 128).mean())
            d.text((4 + j * (C + 12), y + C + 3), f"ink={fg:.4f}", fill=(90, 90, 90))
    cv.save(a.out)
    print("->", a.out)


if __name__ == "__main__":
    main()
