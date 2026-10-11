# -*- coding: utf-8 -*-
"""sde_oracle_test2.py — 收敛版单测：把"漂移公式错误"与"离散化误差"彻底分开

原理: 对线性 SDE  dx = a(t)·x·dt + g(t)·dW，方差满足精确 ODE
         dV/dt = 2 a(t) V + g(t)^2 ,   V(t=1) = 1
      用高精度积分（RK4 + 细分）解 V(t=0)，即**连续时间**下该漂移式的终点方差。
      理论终点方差 = s0^2。凡不等的漂移式 = 不保持边缘分布（与步长无关）。

插值: x_t = (1-t) x0 + t eps,  x0~N(0,s0^2), eps~N(0,1)
      D(t) = (1-t)^2 s0^2 + t^2 ,  v(x,t) = c(t) x ,  c(t) = (t-(1-t)s0^2)/D(t)
      score = -x/D(t) ,  g(t)^2 = 2 γ^2 t (1-t)

三种漂移:
  现状(A):  b = (1 + γ^2 t) v                    -> a_A = (1+γ²t)·c
  修正(B):  b = v + γ^2(1-t)(x + (1-t)v)         -> a_B = (1+γ²(1-t)²)·c + γ²(1-t)
  纯ODE  :  b = v                                -> a_O = c                (应精确 = s0²)
"""
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def coeffs(t, s0, gamma, mode):
    D = (1.0 - t) ** 2 * s0 ** 2 + t ** 2
    c = (t - (1.0 - t) * s0 ** 2) / D
    a = 1.0 - t
    if mode == "ode":
        return c
    if mode == "current":
        return (1.0 + gamma ** 2 * t) * c
    if mode == "correct":
        return (1.0 + gamma ** 2 * a * a) * c + gamma ** 2 * a
    raise ValueError(mode)


def final_var(s0, gamma, mode, n=200000):
    """从 t=1 积分到 t=0，解 dV/dt = 2 a(t) V + g(t)^2（t 递减, dt<0）。RK4。"""
    t, V = 1.0, 1.0
    h = -1.0 / n
    for i in range(n):
        ti = 1.0 + i * h

        def f(tt, VV):
            g2 = 2.0 * gamma ** 2 * tt * (1.0 - tt)
            return 2.0 * coeffs(tt, s0, gamma, mode) * VV + g2

        k1 = f(ti, V)
        k2 = f(ti + h / 2, V + h / 2 * k1)
        k3 = f(ti + h / 2, V + h / 2 * k2)
        k4 = f(ti + h, V + h * k3)
        V = V + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = ti + h
    return V


GAMMA = 0.5
print(f"  连续时间下的终点方差（RK4, 200k 细分步）；理论值 = s0²；γ={GAMMA}")
print(f"  {'s0':>6} | {'理论 s0²':>9} | {'纯ODE':>10} | {'现状(A)':>10} | {'修正(B)':>10}")
for s0 in (1.0, 0.5, 0.2, 0.1):
    th = s0 ** 2
    print(f"  {s0:>6.2f} | {th:>9.4f} | {final_var(s0, GAMMA, 'ode'):>10.4f} | "
          f"{final_var(s0, GAMMA, 'current'):>10.4f} | {final_var(s0, GAMMA, 'correct'):>10.4f}")

print("\n  γ 依赖（s0=1.0）:")
print(f"  {'γ':>6} | {'理论':>9} | {'现状(A)':>10} | {'修正(B)':>10}")
for g in (0.0, 0.1, 0.2, 0.5, 1.0):
    print(f"  {g:>6.2f} | {1.0:>9.4f} | {final_var(1.0, g, 'current'):>10.4f} | {final_var(1.0, g, 'correct'):>10.4f}")

print("""
  判读要点:
   * 纯ODE 列必须精确 = 理论值（若不然说明积分器本身有问题, 本次测试无效）。
   * 现状(A) 偏离理论 -> **当前漂移式不保持边缘分布**（与步长/尺度无关的数学事实）。
   * 修正(B) 若 = 理论 -> 该式是自洽解，可直接替换。
   * 看偏离是否随 s0（尺度）变化 -> 判断"修好归一化是否就能恢复 SDE"。
""")
