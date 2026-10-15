# -*- coding: utf-8 -*-
"""test_refactor_pr4d.py —— 2026-10-14 [重构 PR4] **ObjectiveComposer** 的验收测试 ✓

审查 §4.2: "`Trainer` 控制循环，`TrainStep` 完成一次更新，**`ObjectiveComposer` 组合损失**，
  `CheckpointManager` 负责持久化，各自保持小而清晰" ✓
审查原则 C: "实验插件可以增加 Loss 或条件路径，但**不能修改核心训练生命周期**…
  只需**注册相应模块和损失**" ✓
审查 P1-6: "应该返回 detached GPU tensor，**仅在日志周期批量转换为 CPU 数值**" ✓

本组测试锁定:
  ① ★ `total` 与**手写累加链逐位一致** ✓（浮点加法不满足结合律 ⇒ 顺序是语义 ✓）
  ② ★ **顺序敏感** ✓（乱序注册 ⇒ 结果不同 ✓ —— 这正是"不要重排"的证明 ✓）
  ③ raw 一律是 **detached tensor** ✓（唯一转换点是 `as_floats()` ✓ 无每步同步 ✗）
  ④ 零权重项**照样加** ✓（与原链一致 ✓ NaN 不会被静默吞掉 ✓）
  ⑤ 无副作用 ✓（不碰 optimizer / checkpoint / 参数 ✓）
"""
from __future__ import annotations

import io
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train.objectives import ObjectiveComposer, compose                  # noqa: E402


# ─────────────── ① 与手写链逐位一致 ───────────────
def test_total_matches_handwritten_chain_bitwise():
    """★★ 核心 ✓: `total` 必须与手写 `a + w1*b + w2*c + …` **逐位**一致 ✓"""
    for seed in range(20):
        g = torch.Generator().manual_seed(seed)
        base = torch.randn((), generator=g)
        terms = [(("t%d" % i), torch.randn((), generator=g),
                  float(torch.randn((), generator=g).abs())) for i in range(5)]
        ref = base
        for _n, loss, w in terms:
            ref = ref + w * loss                    # 手写链（同顺序 ✓）
        obj = compose(base, terms)
        assert torch.equal(ref, obj.total), f"seed={seed} 不逐位一致 ✗"


def test_total_matches_with_unnormalized_weights():
    """权重为 0 / 1 / 大值 等边界也要逐位一致 ✓"""
    g = torch.Generator().manual_seed(7)
    base = torch.randn((), generator=g)
    terms = [("a", torch.randn((), generator=g), 0.0),
             ("b", torch.randn((), generator=g), 1.0),
             ("c", torch.randn((), generator=g), 1e6),
             ("d", torch.randn((), generator=g), -0.5)]
    ref = base
    for _n, loss, w in terms:
        ref = ref + w * loss
    assert torch.equal(ref, compose(base, terms).total)


# ─────────────── ② 顺序敏感（为何不得重排 ✓）───────────────
def test_order_matters_and_is_preserved():
    """★ 浮点加法**不满足结合律** ✗ ⇒ 顺序变了结果就可能不同 ✓

    本测试构造一组**已知会因顺序而变**的项 ✓ 以证明"注册顺序 = 语义" ✓
    """
    base = torch.tensor(1.0)
    big = torch.tensor(1e16)
    small = torch.tensor(1.0)
    neg = torch.tensor(-1e16)
    # 顺序 A: (base + big) + small + neg ⇒ small 被吞掉 ✓
    oa = compose(base, [("big", big, 1.0), ("small", small, 1.0), ("neg", neg, 1.0)])
    # 顺序 B: (base + big) + neg + small ⇒ small 存活 ✓
    ob = compose(base, [("big", big, 1.0), ("neg", neg, 1.0), ("small", small, 1.0)])
    assert not torch.equal(oa.total, ob.total), \
        "该构造下两种顺序必须给出不同结果 ✓（证明顺序是语义 ✓）"
    assert float(ob.total) == 1.0 and float(oa.total) == 0.0, (float(oa.total), float(ob.total))


def test_registration_order_reported():
    obj = compose(torch.zeros(()), [("a", torch.zeros(()), 1.0),
                                    ("b", torch.zeros(()), 1.0)])
    assert obj.names() == ["fm", "a", "b"], obj.names()


# ─────────────── ③ raw 是 detached tensor（无每步同步 ✗）───────────────
def test_terms_raw_are_detached_tensors():
    x = torch.tensor(2.0, requires_grad=True)
    obj = ObjectiveComposer(x * 3.0, base_name="fm")
    obj.add("a", x * 4.0, 0.5)
    for t in obj.terms():
        assert isinstance(t["raw"], torch.Tensor), \
            "raw 必须是 **tensor** ✗（float 会每步同步 ✗ 违反 P1-6 ✓）"
        assert not t["raw"].requires_grad, "raw 必须 detached ✓"
    assert isinstance(obj.detach_total, torch.Tensor)


def test_as_floats_is_the_only_conversion_point():
    """★ 唯一转换点是 `as_floats()` / `weighted_floats()` / `describe()` ✓"""
    obj = compose(torch.tensor(1.0), [("a", torch.tensor(2.0), 0.5)])
    assert obj.as_floats() == {"fm": 1.0, "a": 2.0}
    assert obj.weighted_floats() == {"fm": 1.0, "a": 1.0}
    assert "a=1(w=0.5)" in obj.describe()
    # 数值读取不应改变图 ✓
    assert torch.equal(obj.total, torch.tensor(2.0))


def test_active_filters_zero_weight():
    obj = compose(torch.zeros(()), [("on", torch.ones(()), 1.0),
                                    ("off", torch.ones(()), 0.0)])
    assert obj.active() == ["fm", "on"], obj.active()


# ─────────────── ④ 零权重照样加（不静默吞 NaN ✓）───────────────
def test_zero_weight_still_adds_so_nan_surfaces():
    """★ 与原链一致 ✓: `0.0 * nan = nan` ⇒ **不会被静默吞掉** ✓

    （若改成"权重为 0 就跳过" ✗，一个坏样本在该项里产生的 NaN 就会**消失** ✗
      —— 那会掩盖真实的数值问题 ✗）
    """
    obj = compose(torch.tensor(1.0), [("bad", torch.tensor(float("nan")), 0.0)])
    assert math.isnan(float(obj.total)), "零权重项里的 NaN 必须**照实传播** ✓"


def test_none_loss_treated_as_zero():
    obj = compose(torch.tensor(5.0), [("missing", None, 0.3)])
    assert float(obj.total) == 5.0, "None ⇒ 按 0 处理 ✓（不崩 ✓）"
    assert obj.terms()[-1]["raw"].numel() == 0 or float(obj.terms()[-1]["raw"]) == 0.0


# ─────────────── ⑤ 无副作用 + 可反传 ───────────────
def test_total_is_differentiable():
    w = torch.tensor(2.0, requires_grad=True)
    obj = compose(w * 3.0, [("a", w * 5.0, 0.5)])
    obj.total.backward()
    assert float(w.grad) == 3.0 + 0.5 * 5.0, float(w.grad)


def test_module_has_no_optimizer_or_checkpoint_side_effects():
    """★ 原则 C 守卫 ✓: 本模块不得 import optimizer / checkpoint ✓"""
    src = io.open("src/train/objectives.py", encoding="utf-8").read()
    for bad in ("torch.optim", "import ckpt", "from src.train.ckpt",
                "save_checkpoint", ".step()", "zero_grad"):
        assert bad not in src, f"objectives.py 不得涉及 `{bad}` ✗（原则 C ✓）"


def test_compose_helper_equivalence():
    g = torch.Generator().manual_seed(3)
    base = torch.randn((), generator=g)
    ts = [("x", torch.randn((), generator=g), 0.7)]
    a = compose(base, ts)
    b = ObjectiveComposer(base, "fm")
    for nm, lo, w in ts:
        b.add(nm, lo, w)
    assert torch.equal(a.total, b.total)


# ─────────────── 接线守卫 ───────────────
def test_train_py_uses_composer_for_wired_terms():
    """★ 接线守卫 ✓: 已迁移的三类项必须走 composer ✓ 且顺序断言仍在 ✓"""
    t = io.open("src/train/train.py", encoding="utf-8").read()
    assert "_ObjComposer(" in t, "未接线到 ObjectiveComposer ✗"
    for name in ('_obj.add("repa"', '_obj.add("std_mid"', '_obj.add("latent_skel"',
                 '_obj.add("style_rank"', '_obj.add("critic_writer"',
                 '_obj.add("critic_away"'):
        assert name in t, f"缺注册项 {name} ✗"
    # 顺序: repa 必须在 std_mid 之前 ✓（浮点顺序是语义 ✓）
    i_repa = t.index('_obj.add("repa"')
    i_std = t.index('_obj.add("std_mid"')
    i_skel = t.index('_obj.add("latent_skel"')
    assert i_repa < i_std < i_skel, "注册顺序被改变 ✗（会改变训练轨迹 ✗）"
    i_cw = t.index('_obj.add("critic_writer"')
    i_ca = t.index('_obj.add("critic_away"')
    assert i_cw < i_ca, "critic 两项顺序被改变 ✗"
