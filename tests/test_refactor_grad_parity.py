# -*- coding: utf-8 -*-
"""test_refactor_grad_parity.py —— **梯度等价黄金**（审查 PR0 ✓ 2026-10-14 ✓）

审查 PR0 原文:
  "**梯度等价测试** —— 在固定 FP32 小 batch 上，检查**主干和条件层**的梯度，
   避免'前向看起来正常，实际训练方向变了'。" ✓

★ 本文件是那条要求的**自动化落地** ✓:
  · 前向黄金（`tools/verify_forward_parity.py` ✓）是 `no_grad` ✗ ⇒ **管不到反传** ✗
  · 而 PR4 的搬移大量在**反传与更新**路径 ✓（backward 策略 ✓ 损失装配顺序 ✓ clip/EMA ✓）
  ⇒ 必须有梯度黄金 ✓

★ 覆盖范围与**灵敏度**（已实测 ✓）:
  · 记录**全参数梯度**按固定顺序拼接后的 sha256 ✓（**逐位** ✓）
  · 记录按角色分组的梯度范数 ✓（backbone / condition ✓ 便于定位 ✓）
  · **阴性对照** ✓: 单个权重微扰 `1e-6` ⇒ sha 变化 ✓（`4270a845…` → `0b7325f2…` ✓）
    ⇒ 黄金**确实灵敏** ✓ 不是"永远通过"的假测试 ✓
"""
from __future__ import annotations

import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CKPT = "_local_ckpt/0055000.pt"
GOLDEN = "assets/_audit/grad_v84A55k.json"

pytestmark = pytest.mark.skipif(
    not (os.path.exists(CKPT) and os.path.exists(GOLDEN)),
    reason="缺少本地 ckpt 或梯度黄金（二者都不入库 ✓）")


def _run():
    from tools.verify_grad_parity import compute
    return compute(CKPT)


def test_grad_fingerprint_is_bit_exact():
    """★ 全参数梯度 sha256 必须与黄金**逐位一致** ✓"""
    gold = json.load(io.open(GOLDEN, encoding="utf-8"))
    res = _run()
    assert res["grad_sha256"] == gold["grad_sha256"], (
        "梯度不一致 ✗ ⇒ **训练方向已变** ✗\n"
        f"  golden {gold['grad_sha256']}\n  now    {res['grad_sha256']}")


def test_grad_fingerprint_is_deterministic():
    """★ 同一进程内跑两次必须**完全相同** ✓（否则黄金无意义 ✗）"""
    a, b = _run(), _run()
    assert a["grad_sha256"] == b["grad_sha256"]
    assert a["group_norms"] == b["group_norms"]


def test_group_norms_match_golden():
    """★ 分组范数一致 ✓（比 sha 更可读: 能直接看出**哪一组**变了 ✓）"""
    gold = json.load(io.open(GOLDEN, encoding="utf-8"))
    res = _run()
    assert set(res["group_norms"]) == set(gold["group_norms"]), \
        "参数分组发生变化 ✗（模型结构被改动?）"
    for k, v0 in gold["group_norms"].items():
        v1 = res["group_norms"][k]
        assert abs(v0 - v1) <= 1e-9 * max(1.0, abs(v0)), f"{k}: {v0} -> {v1}"


def test_covers_backbone_and_conditioning():
    """★ 审查要求"主干**和条件层**" ✓ —— 必须两组都**有梯度** ✓"""
    res = _run()
    assert res["group_norms"].get("backbone", 0.0) > 0, "主干无梯度 ✗"
    assert res["group_norms"].get("condition", 0.0) > 0, "条件层无梯度 ✗"
    # 参数量应与审查的 P≈59.9M 吻合 ✓（交叉核对 ✓）
    assert 5.5e7 < res["n_grad_elements"] < 6.5e7, res["n_grad_elements"]


def test_golden_is_sensitive_to_tiny_perturbation(monkeypatch):
    """★★ **阴性对照** ✓ —— 证明这份黄金**能失败** ✓（不是假测试 ✓）

    微扰**单个**权重 `1e-6` ⇒ sha 必须变化 ✓（已实测 ✓）
    """
    from tools import verify_grad_parity as V
    gold = json.load(io.open(GOLDEN, encoding="utf-8"))
    orig = V.load
    touched = []

    def _patched(ckpt, device="cpu"):
        m, a = orig(ckpt, device)
        import torch
        with torch.no_grad():
            for n, p in m.named_parameters():
                if n.endswith("weight") and p.dim() == 2:
                    p[0, 0] += 1e-6
                    touched.append(n)
                    break
        return m, a

    monkeypatch.setattr(V, "load", _patched)
    res = V.compute(CKPT)
    assert touched, "未找到可微扰的 2D 权重 ✗（测试自身失效 ✗）"
    assert res["grad_sha256"] != gold["grad_sha256"], \
        "微扰 1e-6 后 sha 竟然不变 ✗ ⇒ 黄金**不灵敏** ✗（假测试 ✗）"
