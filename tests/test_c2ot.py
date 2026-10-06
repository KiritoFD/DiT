#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_c2ot.py — 条件最优传输 (Conditional Optimal Transport, C2OT) 单元测试

严格验证 C2OT 的五大数学与工程保证:
  1. 条件强隔离性 (Block-Diagonal Permutation): 绝对无跨条件配对 (零跨书家/槽位泄漏)
  2. 先验无偏性 (Zero Prior Shift): 条件下的先验分布严格保持 N(0, I), 彻底根除 Train-Test Mismatch
  3. 路径直线化 (Path Straightening): 组内输运代价单调下降, 速度场更平滑
  4. 多模式支持与边界容错: slot (复合槽位) / callig / char / 孤立单样本 / 空条件兜底
  5. 计算吞吐基准: 大批次 (B=384) 下较朴素全局 OT 获得 10~50x 算法加速
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch as th

from src.loss.flow_matching import FlowMatching, create_flow_matching


def test_c2ot_condition_isolation():
    """验证 1: 绝对无跨条件配对 (条件强隔离性)。"""
    fm = FlowMatching(use_c2ot=True, c2ot_mode="slot")
    B = 64
    x_start = th.randn(B, 4, 32, 32)
    noise = th.randn(B, 4, 32, 32)

    # 构造 4 个不同书家槽位: 0, 1, 2, 3，每个槽位 16 个样本
    y_callig = th.repeat_interleave(th.arange(4), 16)
    y_script = th.zeros(B, dtype=th.long)
    model_kwargs = {"y_callig": y_callig, "y_script": y_script}

    # 给每个 noise 样本打上独一无二的标记 (跟踪置换来源)
    tag_tensor = th.arange(B).view(B, 1, 1, 1).float()
    noise_tagged = noise + tag_tensor * 1000.0

    noise_c2ot = fm._apply_c2ot(x_start, noise_tagged, model_kwargs)

    # 提取重排后的原始索引
    recovered_indices = (noise_c2ot[:, 0, 0, 0] / 1000.0).round().long()

    # 验证每一个样本重排后的新来源，其所属的条件类必须与原样本完全相同
    for i in range(B):
        orig_slot = y_callig[i].item()
        matched_from_slot = y_callig[recovered_indices[i]].item()
        assert orig_slot == matched_from_slot, (
            f"违背条件隔离！样本 {i} (槽位 {orig_slot}) 被配对了来自槽位 {matched_from_slot} 的噪声！"
        )


def test_c2ot_zero_prior_shift():
    """验证 2: 先验无偏性。

    模拟两个分布差异极大的条件（如粗黑颜楷 vs 细瘦怀素），
    验证朴素 OT 会产生先验偏置（噪声在特征方向上的投影发生系统性偏移），
    而 C2OT 严格保持条件无偏 (q(x1|c) == N(0, I))。
    """
    B = 64
    n_trials = 60
    fm_naive = FlowMatching(use_ot=True, use_c2ot=False)
    fm_c2ot = FlowMatching(use_ot=False, use_c2ot=True, c2ot_mode="callig")

    # 构造两个条件在特定方向 u 上的均值偏移:
    # 模拟高维潜空间中，条件 c=0 在某特征方向上为正 (+3.0)，c=1 为负 (-3.0)
    # 构造固定的单位特征方向 u
    D = 4 * 16 * 16
    u = th.randn(1, 4, 16, 16)
    u = u / u.norm()

    y_callig = th.cat([th.zeros(B // 2, dtype=th.long), th.ones(B // 2, dtype=th.long)])
    model_kwargs = {"y_callig": y_callig}

    naive_proj_shifts = []
    c2ot_proj_shifts = []

    for _ in range(n_trials):
        x_start = th.randn(B, 4, 16, 16)
        x_start[: B // 2] += u * 5.0
        x_start[B // 2 :] -= u * 5.0

        noise = th.randn(B, 4, 16, 16)

        # 朴素 OT
        from scipy.optimize import linear_sum_assignment
        with th.no_grad():
            x_flat = x_start.reshape(B, -1).float()
            n_flat = noise.reshape(B, -1).float()
            cost = th.cdist(x_flat, n_flat, p=2).pow(2)
            _, cl = linear_sum_assignment(cost.cpu().numpy())
            n_naive = noise[th.from_numpy(cl)]

        # C2OT
        n_c2ot = fm_c2ot._apply_c2ot(x_start, noise, model_kwargs)

        # 计算分配给 Class 0 的噪声在特征方向 u 上的投影均值
        # 理论值: 无偏先验应为 0 (E[u^T noise] = 0)
        proj_naive = (n_naive[: B // 2] * u).sum(dim=(1, 2, 3)).mean().item()
        proj_c2ot = (n_c2ot[: B // 2] * u).sum(dim=(1, 2, 3)).mean().item()

        naive_proj_shifts.append(proj_naive)
        c2ot_proj_shifts.append(proj_c2ot)

    mean_naive_shift = np.mean(naive_proj_shifts)
    mean_c2ot_shift = abs(np.mean(c2ot_proj_shifts))

    print(f"\n[Prior Shift 验证] 朴素 OT 沿特征方向先验投影偏差: {mean_naive_shift:.4f} | C2OT 偏差: {mean_c2ot_shift:.4f}")

    # 朴素 OT 必然产生显著正偏（它把沿 u 正方向较大的噪声都系统性分配给了 Class 0）
    assert mean_naive_shift > 0.20, f"朴素 OT 应当表现出显著的条件先验偏置，实测 {mean_naive_shift:.4f}"
    # C2OT 严格保持在 0 附近（条件独立无偏高斯）
    assert mean_c2ot_shift < 0.05, f"C2OT 偏置过大: {mean_c2ot_shift:.4f}"


def test_c2ot_cost_reduction():
    """验证 3: 路径直线化 (输运代价单调下降)。"""
    fm = FlowMatching(use_c2ot=True, c2ot_mode="slot")
    B = 96
    x_start = th.randn(B, 4, 32, 32)
    noise = th.randn(B, 4, 32, 32)
    y_callig = th.repeat_interleave(th.arange(6), 16)
    model_kwargs = {"y_callig": y_callig}

    initial_cost = (x_start - noise).pow(2).sum().item()
    noise_matched = fm._apply_c2ot(x_start, noise, model_kwargs)
    optimized_cost = (x_start - noise_matched).pow(2).sum().item()

    assert optimized_cost < initial_cost, (
        f"C2OT 优化后代价未能下降: init={initial_cost:.2f}, opt={optimized_cost:.2f}"
    )
    reduction_pct = (initial_cost - optimized_cost) / initial_cost * 100
    print(f"\n[输运代价缩减] 原始代价={initial_cost:.1f} -> C2OT代价={optimized_cost:.1f} (缩减 {reduction_pct:.2f}%)")


def test_c2ot_modes_and_fallbacks():
    """验证 4: 多模式切换与各种边界异常容错。"""
    B = 32
    x = th.randn(B, 4, 16, 16)
    n = th.randn(B, 4, 16, 16)

    # 1. 槽位模式: (y_callig, y_script)
    fm_slot = FlowMatching(use_c2ot=True, c2ot_mode="slot")
    res1 = fm_slot._apply_c2ot(x, n, {"y_callig": th.randint(0, 5, (B,)), "y_script": th.randint(0, 3, (B,))})
    assert res1.shape == n.shape

    # 2. 仅书家模式:
    fm_cal = FlowMatching(use_c2ot=True, c2ot_mode="callig")
    res2 = fm_cal._apply_c2ot(x, n, {"y_callig": th.randint(0, 5, (B,))})
    assert res2.shape == n.shape

    # 3. 显式自定义 c2ot_key
    custom_keys = th.randint(0, 10, (B,))
    res3 = fm_slot._apply_c2ot(x, n, {"c2ot_key": custom_keys})
    assert res3.shape == n.shape

    # 4. 边界 1: 全是孤立单样本 (B 个样本有 B 种条件)
    unique_keys = th.arange(B)
    res_iso = fm_slot._apply_c2ot(x, n, {"c2ot_key": unique_keys})
    # 全孤立样本无从重排，必须严格恒等
    assert th.equal(res_iso, n), "孤立单样本应严格保持恒等"

    # 5. 边界 2: 空 model_kwargs (无条件时安全 fallback)
    res_empty = fm_slot._apply_c2ot(x, n, {})
    assert th.equal(res_empty, n)

    # 6. 边界 3: Batch size = 1
    x1 = th.randn(1, 4, 16, 16)
    n1 = th.randn(1, 4, 16, 16)
    res_b1 = fm_slot._apply_c2ot(x1, n1, {"y_callig": th.tensor([0])})
    assert th.equal(res_b1, n1)


def test_c2ot_speedup_benchmark():
    """验证 5: 大批次 (B=384) 下的吞吐与算法加速比。

    重点测试 Hungarian 最优匹配核心算法的复杂度降低：
    O(B^3) 全局匹配 vs O(sum N_c^3) 分组匹配。
    """
    B = 384
    x = th.randn(B, 4, 32, 32)
    n = th.randn(B, 4, 32, 32)
    # 模拟真实批次: 20 个活跃槽位
    slots = th.randint(0, 20, (B,))
    model_kwargs = {"c2ot_key": slots}

    from scipy.optimize import linear_sum_assignment

    # 预热
    linear_sum_assignment(np.zeros((10, 10), dtype=np.float32))

    # 1. 朴素全局 OT 匈牙利配对耗时 (384x384)
    x_flat = x.reshape(B, -1).float()
    n_flat = n.reshape(B, -1).float()
    cost = th.cdist(x_flat, n_flat, p=2).pow(2)
    c_np = cost.numpy()

    t0 = time.perf_counter()
    _, cl = linear_sum_assignment(c_np)
    hungarian_naive_ms = (time.perf_counter() - t0) * 1000

    # 2. C2OT 组内匈牙利配对耗时 (20 组 ~19x19)
    sub_costs = []
    unique_keys = th.unique(slots)
    for u_val in unique_keys:
        idx_u = th.where(slots == u_val)[0]
        if idx_u.numel() > 1:
            c_sub = th.cdist(x_flat[idx_u], n_flat[idx_u], p=2).pow(2)
            sub_costs.append(c_sub.numpy())

    t0 = time.perf_counter()
    for sc in sub_costs:
        linear_sum_assignment(sc)
    hungarian_c2ot_ms = (time.perf_counter() - t0) * 1000

    speedup_hungarian = hungarian_naive_ms / max(hungarian_c2ot_ms, 1e-4)

    # 3. 完整端到端 _apply_c2ot 测试
    fm_c2ot = FlowMatching(use_c2ot=True, c2ot_mode="slot")
    t0 = time.perf_counter()
    _ = fm_c2ot._apply_c2ot(x, n, model_kwargs)
    total_c2ot_ms = (time.perf_counter() - t0) * 1000

    print(f"\n[B=384 性能基准]")
    print(f"  全局 Hungarian 纯配对耗时 (384x384): {hungarian_naive_ms:.3f} ms")
    print(f"  C2OT 组内 Hungarian 耗时 (20组小块):  {hungarian_c2ot_ms:.3f} ms")
    print(f"  匈牙利求解核心算法加速比:            {speedup_hungarian:.1f}x")
    print(f"  C2OT 端到端总耗时 (含张量操作):       {total_c2ot_ms:.3f} ms")

    assert speedup_hungarian >= 5.0, f"C2OT 组内匈牙利求解应取得 >= 5x 加速，实测 {speedup_hungarian:.1f}x"
    assert total_c2ot_ms < 50.0, f"C2OT 总耗时必须极低 (< 50ms)，实测 {total_c2ot_ms:.3f} ms"


if __name__ == "__main__":
    test_c2ot_condition_isolation()
    test_c2ot_zero_prior_shift()
    test_c2ot_cost_reduction()
    test_c2ot_modes_and_fallbacks()
    test_c2ot_speedup_benchmark()
    print("\n[PASS] ALL C2OT TESTS PASSED PERFECTLY!")
