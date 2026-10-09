# -*- coding: utf-8 -*-
"""place_48_eval187.py — 把 48 导出的 187 张图整理进本地 eval/<tag>/{idx}.png。

48 的导出命名是零填充的 pred/000.png..186.png, 海报/指标脚本用的是 {idx}.png,
所以这里做一次重命名, 并顺手用我们自己的 SSIM 做**协议验证**
(若 Heun/t*1000 写错, SSIM 会掉到 0.2 量级; 正常应在 0.5~0.7)。

用法:
    python tools/place_48_eval187.py --src <解包后的 eval187 目录> --tags v_b_aug_route_60k,v_l_aug_route_50k
"""
import argparse
import os
import shutil
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

N = 187


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--tags", default="")
    ap.add_argument("--validate", action="store_true", default=True)
    a = ap.parse_args()

    tags = [t.strip() for t in a.tags.split(",") if t.strip()] or sorted(
        d for d in os.listdir(a.src) if os.path.isdir(os.path.join(a.src, d)))

    from src.eval.metrics import ssim as our_ssim

    for tag in tags:
        src = os.path.join(a.src, tag, "pred")
        if not os.path.isdir(src):
            print(f"  ⚠ 跳过 {tag}: 无 {src}")
            continue
        dst = os.path.join("eval", tag)
        os.makedirs(dst, exist_ok=True)
        n_ok = 0
        for i in range(N):
            p = os.path.join(src, f"{i:03d}.png")
            if not os.path.exists(p):
                p = os.path.join(src, f"{i}.png")
            if not os.path.exists(p):
                continue
            shutil.copyfile(p, os.path.join(dst, f"{i}.png"))
            n_ok += 1
        print(f"  {tag}: 放入 {n_ok}/{N} 张 -> {dst}")

        if a.validate and n_ok:
            vals, miss = [], 0
            for i in range(N):
                pp = os.path.join(dst, f"{i}.png")
                gp = os.path.join("eval", "gt", f"{i}.png")
                if not (os.path.exists(pp) and os.path.exists(gp)):
                    miss += 1
                    continue
                x = np.asarray(Image.open(pp).convert("RGB"), np.float32) / 255.0
                y = np.asarray(Image.open(gp).convert("RGB"), np.float32) / 255.0
                vals.append(float(our_ssim(x, y)))
            m = float(np.mean(vals))
            flag = "✓ 协议正常" if 0.45 < m < 0.75 else "✗ 异常! 检查 t 缩放 / Heun"
            print(f"    [验证] 与本地 gt 的均值 SSIM = {m:.4f}  (缺 {miss})  {flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
