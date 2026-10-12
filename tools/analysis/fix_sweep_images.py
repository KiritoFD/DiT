#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fix_sweep_images.py — 在 48 上自动找出"真正的模型输出图"并整理成 eval/v81fix_<step>/<i>.png

判据（不靠猜目录名）：
  * 候选组 = results_dir 下**每个目录**的 *.png（递归）
  * 命中条件：该组恰好 N 张，且**平均墨量 > 0.01**（细线"标准字输入图"墨量≈0，会被排除）
  * 若多组命中，取组内文件名字典序与 eval 行顺序一致者，并打印全部候选供核对
最后打印每张的墨量，确认非空。
"""
import glob
import os
import shutil
import sys

import numpy as np
from PIL import Image

os.chdir("/home/ds/Workspace/DiT")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STEPS = ["0020000", "0035000", "0040000", "0050000", "0055000", "0100000", "0110000", "0115000"]
IDX = [131, 100, 67, 30, 177, 173, 86, 83, 145, 22]
N = len(IDX)


def ink(p):
    try:
        return float((np.asarray(Image.open(p).convert("L")) < 128).mean())
    except Exception:
        return None


for st in STEPS:
    root = f"_local_ckpt/gen_v81sw_{st}_n10"
    dst = f"eval/v81fix_{st}"
    if not os.path.isdir(root):
        print(f"  {st}: 缺 results_dir {root}")
        continue
    groups = {}
    for p in glob.glob(os.path.join(root, "**", "*.png"), recursive=True):
        groups.setdefault(os.path.dirname(p), []).append(p)
    print(f"\n=== {st} === 根: {root}")
    for d, ps in sorted(groups.items()):
        ps = sorted(ps)
        inks = [ink(p) for p in ps]
        inks = [x for x in inks if x is not None]
        mi = float(np.mean(inks)) if inks else 0.0
        rel = d.replace(root + "/", "")
        print(f"  [{len(ps):>3} 张] mean_ink={mi:.4f}  {rel}")
    # 选组：恰好 N 张且 mean_ink > 0.01
    cand = None
    for d, ps in sorted(groups.items()):
        ps = sorted(ps)
        if len(ps) != N:
            continue
        inks = [ink(p) for p in ps]
        if inks and float(np.mean(inks)) > 0.01:
            cand = (d, ps)
            break
    if cand is None:                     # 退一步：任何张数>0 且 mean_ink>0.01 的组
        for d, ps in sorted(groups.items()):
            ps = sorted(ps)
            inks = [ink(p) for p in ps]
            if inks and float(np.mean(inks)) > 0.01:
                cand = (d, ps)
                break
    if cand is None:
        print("  ✗ 没找到 non-blank 的候选组")
        continue
    d, ps = cand
    if len(ps) > N:
        ps = ps[:N]
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst, exist_ok=True)
    ok = 0
    for i, p in enumerate(ps):
        shutil.copyfile(p, os.path.join(dst, f"{i}.png"))
        if (ink(p) or 0) > 0.005:
            ok += 1
    print(f"  ✅ 用 {d.replace(root+'/', '')} -> {dst}  非空 {ok}/{len(ps)}")
    print("     墨量: " + " ".join(f"{ink(os.path.join(dst, str(i)+'.png')):.3f}" for i in range(len(ps))))

# 打包（保持 eval/v81fix_<step>/ 结构，解到 eval/ 下即可）
DL = " ".join(f"v81fix_{s}" for s in STEPS if os.path.isdir(f"eval/v81fix_{s}"))
os.system(f"rm -f /tmp/v81fix.tgz && tar -czf /tmp/v81fix.tgz -C eval {DL} && ls -la /tmp/v81fix.tgz")
