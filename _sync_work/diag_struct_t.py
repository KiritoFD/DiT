#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断: 结构 loss (对 x0_pred 的 Dice) 在不同 t 下有多"免费"。

x0_pred = x_t - t*v。若 v=0 (模型什么都不学), x0_pred = x_t = (1-t)x0 + t*eps。
t 越小 -> x_t 越接近 x0 -> Dice 越接近 1 -> loss 越没有信息量。
本脚本扫 t, 报 "v=0 时的 Dice" 和 "v 为真值时的 Dice(=1)" 作对照。
"""
import os
import sys

import numpy as np

os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8")
th = None
z = np.load("data/top10_style23/skel64/val.npz")
x0 = (z["gt_skel"][:64].astype(np.float32) / 255.0)          # 1=墨
x0 = 1.0 - 2.0 * x0                                          # [-1,1], 墨=-1
x0 = x0[:, None]                                             # (N,1,64,64)
n = x0.shape[0]
print(f"n={n}  墨占比={((1-x0)/2).mean():.4f}")


def dice_p(pin, tin, eps=1e-6):
    num = 2 * (pin * tin).sum() + eps
    den = pin.sum() + tin.sum() + eps
    return float(num / den)


rng = np.random.default_rng(0)
print(f"\n{'t':>6} {'v=0 Dice':>10} {'|v=0 二值Dice':>12} {'|正确v Dice':>12}  说明")
for t in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.98):
    eps_ = rng.standard_normal(x0.shape).astype(np.float32)
    xt = (1 - t) * x0 + t * eps_
    tin = (1 - x0) * 0.5
    # v=0 (模型未学习)
    x0p0 = xt
    pin0 = np.clip((1 - x0p0) * 0.5, 1e-4, 1 - 1e-4)
    d0 = dice_p(pin0, tin)
    b0 = dice_p((pin0 > 0.5).astype(np.float32), (tin > 0.5).astype(np.float32))
    # v=真值 (完全学会)
    v_true = eps_ - x0
    x0p1 = xt - t * v_true
    d1 = dice_p(np.clip((1 - x0p1) * 0.5, 1e-4, 1 - 1e-4), tin)
    tag = "← 免费(无信号)" if d0 > 0.95 else ("← 有信号" if d0 < 0.85 else "")
    print(f"{t:6.2f} {d0:10.4f} {b0:12.4f} {d1:12.4f}  {tag}")
