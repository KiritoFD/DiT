#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_c2ot_drop_interaction.py — C2OT 与 **per-factor 条件 dropout** 的交互实测。

为什么需要它
-----------
C2OT 的"零先验漂移"证明依赖一个前提：**分组键 = 模型真正看到的（有效）条件**。
本仓库里 per-factor dropout 是在 DiT **forward 内部**才把条件置 null 的，
而 `FlowMatching._extract_c2ot_keys` 取 `model_kwargs` 里的**名义** y_callig/y_script。
于是"同条件子集"里混进了有效条件已被 null 的样本（callig/script/char 各 0.08，all 0.05）
-> 严格证明降为**近似**。本测试把这个近似量出来。

估计量（为什么是"符号加权平均"）
------------------------------
朴素 OT 的偏置是**方向性**的：它最大化 <x0, n>，于是噪声被系统性对齐到样本漂移方向。
要在"有效条件类"上测它：
  1. 构一条**共享漂移轴** u（各类均值都在 u 上有符号偏移 α_c + β_s）——否则方向性偏置
     在类内平均后自相抵消，指标失明（第一版实测基准只有 0.0018，而 Test 2 同轴反向为 0.33）；
  2. 每类算 proj_e = mean_{i∈e} <n_i, u>；
  3. 用**已知的类偏移符号**做加权平均:  shift = Σ_e μ_e · proj_e / Σ_e |μ_e|
     —— 无偏耦合下正负相互抵消、期望为 0；有偏时同号累加。
     （直接取 max|proj_e| 是不行的：类小的时候它测的是有限样本噪声。）

对照
----
  naive      朴素全局 OT          (基准: 应有显著正偏)
  nominal    C2OT + 名义键        (当前代码行为)      <- 关心的数
  effective  C2OT + 有效键        (理论正确分组)
  nodrop     C2OT + 名义键, 且不抽 drop (对照, 应 ≈ nominal 的噪声地板)
  randperm   随机耦合             (噪声地板: 给出估计量的零分布)

判读: |nominal| 落在 randperm 的噪声地板抖动范围内、且远小于 |naive| => 现有实现可用；
      若 nominal 显著 > effective 且超出噪声地板数倍 => 需把耦合键改成"有效条件"。

== 实测结论 (2026-10-06, 机器 4090, 纯 CPU) ==
  按类偏移符号加权, 40 次试验(均值标准误 ≈ std/√40 ≈ 0.01):
    朴素全局 OT            +0.2303
    C2OT + 名义键          +0.0044     <- 当前代码行为, 即 0
    C2OT + 有效键          +0.0126
    不抽 drop (Test 2 复现) +0.0182
    随机耦合(噪声地板)      -0.0006 ± 0.065
  **功效检验** (drop 拉到 6 倍: all .2 / callig .5 / script .5):
    朴素 +0.2463 | 名义 -0.0202 | 有效 -0.0103  -> 仍为噪声级
  => **per-factor dropout 不会破坏 C2OT 的零先验漂移**, 无需把 drop mask 提到耦合之前。

  机制: C2OT 组内所有样本**共享同一均值** μ, OT 代价里的 <μ, n> 项对所有候选是常数,
        从分配目标中整体消掉 -> 组内 OT 只匹配**个体偏差**(这也正是它拉直轨迹的来源),
        因此噪声投影的期望恒为 0、与组的均值无关; drop 只把若干组混合, 混不出方向性偏置。

  注: 噪声地板 ≈0.065 => 本测试能分辨 ≳0.1 的偏置(3σ)或 ≳0.03 的均值偏移(3·SE)；
      0.23 的朴素基准说明估计量确实灵敏。

⚠ 受控合成测试（非真实训练 batch）：它检验"分组键错配"这一机制本身；端到端仍需 v67 实跑。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch as th
from scipy.optimize import linear_sum_assignment

from src.loss.flow_matching import FlowMatching

N_TRIALS = 40
N_CALLIG, N_SCRIPT = 4, 2
PER_SLOT = 64                      # B = 512
ALPHA = [-3., 0., 3., 6.]          # 书家在共享轴上的有符号偏移
BETA = [-2., 2.]                   # 书体在共享轴上的有符号偏移
NULL = 99
DROP = dict(all=0.05, callig=0.08, script=0.08)   # 与 v66/v67 配置一致

# 可选: 用命令行覆盖 drop 概率, 做**功效检验** (一个测不出问题的测试没有价值)
#   python tests/test_c2ot_drop_interaction.py [drop_all] [drop_callig] [drop_script]
if len(sys.argv) >= 4:
    DROP = dict(all=float(sys.argv[1]), callig=float(sys.argv[2]),
                script=float(sys.argv[3]))
    print('[power-check] drop 覆盖为 all=%.2f callig=%.2f script=%.2f'
          % (DROP['all'], DROP['callig'], DROP['script']))


def _unit(shape):
    v = th.randn(*shape)
    return v / v.norm()


def _shift_estimator(noise_assigned, eff_key, mu_e_per_sample, u):
    """按类偏移符号加权平均的先验投影偏差; 同时返回有效类数。"""
    proj = noise_assigned.reshape(noise_assigned.shape[0], -1) @ u
    num = den = 0.0
    n_cls = 0
    for k in th.unique(eff_key).tolist():
        m = (eff_key == k)
        n = int(m.sum())
        if n < 8:
            continue
        mu = float(mu_e_per_sample[m].mean())      # 该类的漂移量(带符号)
        p = float(proj[m].mean())
        num += mu * p
        den += abs(mu)
        n_cls += 1
    return (num / den if den > 0 else float('nan')), n_cls


def main():
    fm = FlowMatching(use_c2ot=True, c2ot_mode="slot")
    B = N_CALLIG * N_SCRIPT * PER_SLOT
    y_callig = th.repeat_interleave(th.arange(N_CALLIG), N_SCRIPT * PER_SLOT)
    y_script = th.arange(N_SCRIPT).repeat(N_CALLIG * PER_SLOT)
    nominal_key = y_callig * 100 + y_script

    u = _unit((4, 16, 16)).reshape(-1)
    alpha = th.tensor(ALPHA)
    beta = th.tensor(BETA)
    slot_off = th.stack([alpha[c] + beta[s]
                         for c in range(N_CALLIG) for s in range(N_SCRIPT)
                         for _ in range(PER_SLOT)]).reshape(B, 1, 1, 1)
    m_alpha, m_beta = float(alpha.mean()), float(beta.mean())

    res = {k: [] for k in ('naive', 'nominal', 'effective', 'nodrop', 'randperm')}
    ncls = []
    for _ in range(N_TRIALS):
        x0 = slot_off * u.reshape(1, -1).reshape(1, 4, 16, 16) + th.randn(B, 4, 16, 16)
        noise = th.randn(B, 4, 16, 16)

        # ---- drop 抽样 (与 dit.py 一致: 各因子独立, 再 OR 上 all) ----
        _all = th.rand(B) < DROP['all']
        c_drop = (th.rand(B) < DROP['callig']) | _all
        s_drop = (th.rand(B) < DROP['script']) | _all
        eff_c = th.where(c_drop, th.full_like(y_callig, NULL), y_callig)
        eff_s = th.where(s_drop, th.full_like(y_script, NULL), y_script)
        eff_key = eff_c * 100 + eff_s
        # 有效条件下该类在 u 上的漂移量: 被 drop 的因子 -> 取其总体均值
        keep_c = (~c_drop).float()
        keep_s = (~s_drop).float()
        mu_eff = (keep_c * alpha[y_callig] + (1 - keep_c) * m_alpha
                  + keep_s * beta[y_script] + (1 - keep_s) * m_beta)

        nominal_kwargs = {"y_callig": y_callig, "y_script": y_script}
        effective_kwargs = dict(nominal_kwargs, c2ot_key=eff_key)

        with th.no_grad():
            cost = th.cdist(x0.reshape(B, -1).float(),
                            noise.reshape(B, -1).float(), p=2).pow(2)
            _, cl = linear_sum_assignment(cost.cpu().numpy())
            n_naive = noise[th.from_numpy(cl)]
            n_perm = noise[th.randperm(B)]

        variants = [
            ('naive', n_naive, eff_key),
            ('randperm', n_perm, eff_key),
            ('nominal', fm._apply_c2ot(x0, noise, nominal_kwargs), eff_key),
            ('effective', fm._apply_c2ot(x0, noise, effective_kwargs), eff_key),
            # 对照: 不抽 drop 时, 名义键==有效键 (Test 2 的复现)
            ('nodrop', fm._apply_c2ot(x0, noise, nominal_kwargs), nominal_key),
        ]
        for tag, n, keys in variants:
            mu_vec = mu_eff if keys is eff_key else slot_off.reshape(B)
            v, n_c = _shift_estimator(n, keys, mu_vec, u)
            res[tag].append(v)
        ncls.append(n_c)

    print('\n' + '=' * 86)
    print('C2OT × per-factor dropout 交互实测 (按类偏移符号加权的先验投影偏差, 无偏应为 0)')
    print('  drop: callig=%.2f script=%.2f all=%.2f | B=%d 槽位=%dx%d 有效类≈%d 试验=%d'
          % (DROP['callig'], DROP['script'], DROP['all'], B, N_CALLIG, N_SCRIPT,
             int(np.mean(ncls)), N_TRIALS))
    print('-' * 86)
    desc = [('naive', '朴素全局 OT (基准, 应有显著正偏)'),
            ('nominal', 'C2OT + 名义键 (当前代码行为)   ← 关心的数'),
            ('effective', 'C2OT + 有效键 (理论正确分组)'),
            ('nodrop', 'C2OT + 名义键, 不抽 drop (Test 2 复现)'),
            ('randperm', '随机耦合 (噪声地板: 估计量零分布)')]
    for tag, d in desc:
        v = np.array(res[tag], dtype=float)
        print('  %-10s %-38s mean=%+.4f  std=%.4f' % (tag, d, float(np.mean(v)),
                                                      float(np.std(v))))
    print('-' * 86)
    m_naive, m_nom, m_eff = (float(np.mean(res['naive'])),
                             float(np.mean(res['nominal'])),
                             float(np.mean(res['effective'])))
    floor = float(np.std(res['randperm']))
    thresh = 0.05
    if abs(m_nom) < thresh and abs(m_nom) <= 3 * floor:
        verdict = ('✓ 名义键偏差 %.4f 已 <0.05 且落在噪声地板(3σ=%.4f)内 -> 现有实现可直接用'
                   % (m_nom, 3 * floor))
    elif abs(m_nom) < 0.5 * abs(m_naive):
        verdict = ('△ 名义键 %.4f 明显小于朴素 %.4f, 但超出噪声地板 -> 记录在案, 建议核 5k 步'
                   % (m_nom, m_naive))
    else:
        verdict = '✗ 偏差偏大 -> 应把耦合键改成"有效条件"'
    print('  结论: 名义 %+.4f | 有效 %+.4f | 朴素 %+.4f | 噪声地板(randperm std) %.4f'
          % (m_nom, m_eff, m_naive, floor))
    print('        %s' % verdict)
    print('=' * 86)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
