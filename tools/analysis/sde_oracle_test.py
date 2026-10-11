# -*- coding: utf-8 -*-
"""sde_oracle_test.py — 一维高斯 oracle 单测：判定 SDE 漂移式是否保持边缘分布（无需训练）

模型（精确可解）:
    x0 ~ N(0, s0^2),  eps ~ N(0,1) 独立,  x_t = (1-t) x0 + t eps
    D(t)  = (1-t)^2 s0^2 + t^2
    v(x,t)= (t - (1-t) s0^2) / D(t) * x          (精确速度场, 假设 DiT 完美)
    score = -x / D(t)

采样: t 从 1 积分到 0, N 步, h = 1/N, dt = -h, xi ~ N(0,1)

    (A) 现状(我们的代码):  x += dt*(1 + γ² t) v                     + γ sqrt(2 t (1-t) h) xi
    (B) 正确(含 score 修正): x += dt*( v + γ²(1-t)(x + (1-t)v) )     + γ sqrt(2 t (1-t) h) xi
        —— 由 Tweedie: ∇log p_t = -(x+(1-t)v)/t, 与 g²=2γ²t(1-t) 代入 b = v - (g²/2)∇log p_t

理论终点方差 = s0² (t=0 时 x_0 的方差)。若某式给出 ≠ s0², 该式不保持边缘分布。
"""
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

N = 50          # 采样步数（与主线一致 heun-50；这里用 Euler 便于对照）
GAMMA = 0.5
NS = 200_000


def run(s0, mode, seed=0, n_steps=N, gamma=GAMMA):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(NS)                    # t=1: x_1 = eps ~ N(0,1)
    h = 1.0 / n_steps
    dt = -h
    for i in range(n_steps):
        t = 1.0 - i * h                            # 当前时刻
        D = (1.0 - t) ** 2 * s0 ** 2 + t ** 2
        v = (t - (1.0 - t) * s0 ** 2) / D * x      # 精确速度场
        if mode == "current":
            drift = (1.0 + gamma ** 2 * t) * v
        elif mode == "correct":
            a = 1.0 - t
            drift = v + gamma ** 2 * a * (x + a * v)
        else:                                      # 纯 ODE (γ=0 的等价检查)
            drift = v
        noise = gamma * np.sqrt(max(2.0 * t * (1.0 - t) * h, 0.0)) * rng.standard_normal(NS)
        x = x + dt * drift + noise
    return float(x.var())


print(f"  N={N} 步, γ={GAMMA}, MC 样本={NS:,}")
print(f"  {'x0 标准差':>10} | {'理论终点方差':>12} | {'现状式(A)':>10} | {'修正式(B)':>10} | {'纯ODE':>8}")
for s0 in (1.0, 0.2):
    th = s0 ** 2
    a = run(s0, "current")
    b = run(s0, "correct")
    c = run(s0, "ode")
    print(f"  {s0:>10.2f} | {th:>12.4f} | {a:>10.4f} | {b:>10.4f} | {c:>8.4f}")

print("""
  判读:
   * 若"纯ODE"≈理论 且 "现状式(A)"系统偏离理论 而 "修正式(B)"≈理论
     -> **现状 SDE 漂移不保持边缘分布 = 实现错误已确认**（在完美的速度场下都能偏）。
   * 偏离量随 s0 如何变化，可用于判断"尺度修好后是否就没事了"。
""")
