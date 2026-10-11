# -*- coding: utf-8 -*-
"""sde_oracle_test3.py — 收敛版单测（修正版）: 同时验 现状 / 专家加号版 / 减号版

线性 SDE  dx = a(t)x dt + g(t)dW  的方差满足精确 ODE:  dV/dt = 2a(t)V - g(t)^2,  V(1)=1
高精度 RK4 积分到 t=0。理论终点方差 = s0^2（因 D(0)=s0^2）。

插值: x_t=(1-t)x0+t eps, x0~N(0,s0^2), eps~N(0,1)
      D(t)=(1-t)^2 s0^2+t^2,  c(t)=v/x=(t-(1-t)s0^2)/D(t),  g(t)^2=2γ^2 t(1-t)

漂移式（a = 1-t）:
  ODE 基准(γ=0):      a = c                                   （必须精确 = s0², 用于验证积分器）
  现状(A):            b = (1+γ²t) v                 -> a = (1+γ²t) c
  专家(加号, B+):     b = v + γ²a(x + a v)          -> a = (1+γ²a²) c + γ²a
  推演(减号, B-):     b = v - γ²a(x + a v)          -> a = (1-γ²a²) c - γ²a
"""
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def a_of(t, s0, gamma, mode):
    D = (1.0 - t) ** 2 * s0 ** 2 + t ** 2
    c = (t - (1.0 - t) * s0 ** 2) / D
    a = 1.0 - t
    if mode == "ode":
        return c, 0.0
    if mode == "current":
        return (1.0 + gamma ** 2 * t) * c, 2.0 * gamma ** 2 * t * (1.0 - t)
    if mode == "plus":
        return (1.0 + gamma ** 2 * a * a) * c + gamma ** 2 * a, 2.0 * gamma ** 2 * t * (1.0 - t)
    if mode == "minus":
        return (1.0 - gamma ** 2 * a * a) * c - gamma ** 2 * a, 2.0 * gamma ** 2 * t * (1.0 - t)
    raise ValueError(mode)


def final_var(s0, gamma, mode, n=400000):
    V = 1.0
    h = -1.0 / n
    for i in range(n):
        ti = 1.0 + i * h

        def f(tt, VV):
            at, g2 = a_of(tt, s0, gamma, mode)
            return 2.0 * at * VV - g2

        k1 = f(ti, V); k2 = f(ti + h / 2, V + h / 2 * k1)
        k3 = f(ti + h / 2, V + h / 2 * k2); k4 = f(ti + h, V + h * k3)
        V = V + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return V


print("【A】积分器自检：γ=0 的 ODE 必须精确 = s0²")
print(f"  {'s0':>6} | {'理论':>9} | {'实测':>9} | 判定")
for s0 in (1.0, 0.5, 0.2, 0.1):
    v = final_var(s0, 0.0, "ode")
    print(f"  {s0:>6.2f} | {s0**2:>9.4f} | {v:>9.4f} | {'✓' if abs(v - s0**2) < 1e-6 else '✗'}")

G = 0.5
print(f"\n【B】γ={G} 下三种漂移的终点方差（理论 = s0²）")
print(f"  {'s0':>6} | {'理论':>9} | {'现状(A)':>10} | {'专家加号(B+)':>12} | {'减号(B-)':>10}")
for s0 in (1.0, 0.5, 0.2, 0.1):
    print(f"  {s0:>6.2f} | {s0**2:>9.4f} | {final_var(s0, G, 'current'):>10.4f} | "
          f"{final_var(s0, G, 'plus'):>12.4f} | {final_var(s0, G, 'minus'):>10.4f}")

print(f"\n【C】γ 依赖（s0=1.0）")
print(f"  {'γ':>5} | {'理论':>7} | {'现状(A)':>9} | {'加号(B+)':>10} | {'减号(B-)':>10}")
for g in (0.0, 0.1, 0.2, 0.5, 1.0):
    print(f"  {g:>5.2f} | {1.0:>7.4f} | {final_var(1.0, g, 'current'):>9.4f} | "
          f"{final_var(1.0, g, 'plus'):>10.4f} | {final_var(1.0, g, 'minus'):>10.4f}")

print("""
判读:
  * 【A】必须全 ✓（否则积分器问题, 测试无效）。
  * 【B/C】只有一列能恒等于理论 -> 那就是自洽的漂移式。
  * 若"现状(A)"偏离 -> **当前实现不保持边缘分布**（与步长、与 latent 尺度都无关的数学事实）。
""")
