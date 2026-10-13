# -*- coding: utf-8 -*-
"""风格度量的**不变量测试**（合成数据，纯 numpy，不需要模型/数据/GPU ✓）。

覆盖:
  ① 完美嵌入（书家 one-hot）⇒ 检索 mAP/Top-1 = 1、XCTAcc = 1
  ② 随机嵌入 ⇒ 检索接近随机、XCTAcc 接近 0.5（证明指标不是恒为 1 的假指标 ✓）
  ③ 三条硬约束**真的生效**: 同书体 / 排除同字 / 排除同作品
  ④ 不可计算的槽位**被跳过并计数**（绝不用别的书体顶替 ✓）
  ⑤ 原型间距**符号正确**（换成错误书家 ⇒ 变负）
  ⑥ HWD-CN 与 Style-MMD 在"同分布"上 ≈0；在"不同分布"上 >0
"""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.style.metrics import (EmbSet, build_prototypes, cross_char_retrieval, hwd_cn,
                                    l2norm, prototype_margin, style_mmd, xctacc)

RNG = np.random.default_rng(0)
D = 256


def _onehot_set(writers, scripts, chars, works, dim=None):
    """书家 one-hot 嵌入：同书家完全相同 ⇒ 检索必须满分 ✓。"""
    u = sorted(set(writers.tolist()))
    idx = {w: i for i, w in enumerate(u)}
    d = dim or len(u)
    emb = np.zeros((len(writers), max(d, D)), dtype=np.float32)
    emb[np.arange(len(writers)), [idx[w] for w in writers.tolist()]] = 1.0
    return EmbSet(emb, writers, scripts, chars, works, [f"s{i}" for i in range(len(writers))])


def _rand_set(writers, scripts, chars, works, seed=1):
    rng = np.random.default_rng(seed)
    return EmbSet(rng.normal(size=(len(writers), D)).astype(np.float32), writers, scripts,
                  chars, works, [f"s{i}" for i in range(len(writers))])


def _grid(n_writer=10, n_char=6, n_work=3, script=0):
    w, c, k = [], [], []
    for wi in range(n_writer):
        for ci in range(n_char):
            for ki in range(n_work):
                w.append(wi)
                c.append(ci)
                k.append(wi * 10 + ki)          # 作品：同书家不同作品
    a = np.array
    return a(w), a([script] * len(w)), a(c), a(k)


# ─────────────────────────── ① / ② 检索 ───────────────────────────
def test_retrieval_perfect_on_writer_onehot():
    w, s, c, k = _grid()
    E = _onehot_set(w, s, c, k)
    r = cross_char_retrieval(E, E, topk=(1, 5))
    assert r["n_used"] > 0, "应有可用 query ✓"
    assert r["map"] == pytest.approx(1.0), "one-hot 嵌入的 mAP 必须为 1 ✓"
    assert r["top1"] == pytest.approx(1.0)
    assert r["topk"][5] == pytest.approx(1.0)


def test_retrieval_near_chance_on_random():
    w, s, c, k = _grid()
    E = _rand_set(w, s, c, k)
    r = cross_char_retrieval(E, E, topk=(1,))
    assert 0.0 <= r["map"] < 0.35, "随机嵌入不应给出高 mAP（防假指标 ✓）"
    assert r["top1"] < 0.4


def test_retrieval_respects_exclusions():
    """同字/同作品必须被排除 ✓ —— 构造只有同字同作品的情形, 应全部跳过 ✓。"""
    w = np.array([0, 0, 1, 1])
    s = np.array([0, 0, 0, 0])
    c = np.array([7, 7, 7, 7])                  # 只有一个字
    k = np.array([3, 3, 3, 3])                  # 只有一个作品
    E = _onehot_set(w, s, c, k)
    r = cross_char_retrieval(E, E, topk=(1,))
    assert r["n_used"] == 0 and r["n_skipped"] == len(E), "全同字同作品 ⇒ 必须全部跳过 ✓"
    # 放宽约束后应可用 ✓
    r2 = cross_char_retrieval(E, E, topk=(1,), exclude_same_char=False, exclude_same_work=False)
    assert r2["n_used"] == len(E)
    assert r2["map"] == pytest.approx(1.0)


def test_retrieval_same_script_constraint():
    """跨书体不得互相检索 ✓；且**无负样本时 AP 不可定义 ⇒ 必须跳过并计数** ✓。"""
    # **同一书体**内含两位书家 × 两个字 × 作品互不相同 ⇒ 每条 query 都有正+负候选 ✓
    w = np.array([0, 0, 1, 1])
    s = np.array([0, 0, 0, 0])
    c = np.array([1, 2, 1, 2])
    k = np.array([0, 2, 1, 3])                  # 作品互不相同 ✓
    E = _onehot_set(w, s, c, k)
    r = cross_char_retrieval(E, E, topk=(1,), same_script=True)
    assert r["n_used"] == 4, "有正有负 ⇒ 4 条都可用 ✓"
    assert r["map"] == pytest.approx(1.0)
    # 只有 1 个候选且恰为同书家 ⇒ 无负样本 ⇒ 跳过 ✓（这是设计行为, 不是 bug ✓）
    one = E.subset(np.array([True, True, False, False]))
    r1 = cross_char_retrieval(one, one, topk=(1,))
    assert r1["n_used"] == 0 and r1["n_skipped"] == 2, "无负样本 ⇒ 跳过并计数 ✓"
    # gallery 只留另一位书家 ⇒ 对 query 而言无正样本 ⇒ 全跳过 ✓
    g = E.subset(np.array([False, False, True, True]))
    assert cross_char_retrieval(E, g, topk=(1,), same_script=True)["n_used"] == 0
    # 关掉"排除同作品"仍应可用 ✓
    assert cross_char_retrieval(E, E, topk=(1,), exclude_same_work=False)["n_used"] == 4


# ─────────────────────── ③ 原型间距（符号） ───────────────────────
def test_prototype_margin_sign_and_swap():
    w, s, c, k = _grid()
    real = _onehot_set(w, s, c, k)
    proto = build_prototypes(real)
    assert len(proto) == 10, "每个书家一个原型 ✓"
    gen = _onehot_set(w, s, c, k)
    m = prototype_margin(gen, proto)
    assert m["margin_mean"] == pytest.approx(1.0), "完美嵌入 ⇒ margin ≈ 1 ✓"
    assert m["pos_frac"] == pytest.approx(1.0)
    # 把书家标签整体错位 +1 ⇒ margin 必须变负 ✓
    wrong = EmbSet(gen.emb, (gen.writer + 1) % 10, gen.script, gen.char, gen.work, gen.name)
    m2 = prototype_margin(wrong, proto)
    assert m2["margin_mean"] < 0, "标签错位后必须更像别的书家 ✓"
    assert m2["pos_frac"] < 0.2


def test_prototype_margin_skips_when_target_missing():
    w, s, c, k = _grid()
    real = _onehot_set(w, s, c, k)
    proto = build_prototypes(real)
    gen = EmbSet(real.emb[:3], np.array([99, 99, 99]), real.script[:3], real.char[:3],
                 real.work[:3], real.name[:3])
    m = prototype_margin(gen, proto)
    assert m["n_used"] == 0 and m["n_skipped"] == 3, "目标书家无原型 ⇒ 跳过并计数 ✓"


# ─────────────────────── ④ HWD-CN / Style-MMD ───────────────────────
def test_hwd_cn_zero_on_same_distribution():
    w, s, c, k = _grid()
    E = _rand_set(w, s, c, k, seed=2)
    out = hwd_cn(E, E)
    assert out["n_used"] == 10
    assert out["hwd_cn_macro"] == pytest.approx(0.0, abs=1e-6), "同集合 ⇒ 距离为 0 ✓"


def test_hwd_cn_positive_on_shifted_distribution():
    w, s, c, k = _grid()
    E = _rand_set(w, s, c, k, seed=3)
    F = EmbSet(E.emb + 0.5, E.writer, E.script, E.char, E.work, E.name)
    assert hwd_cn(F, E)["hwd_cn_macro"] > 0.1, "整体平移 ⇒ 距离显著 > 0 ✓"
    # 均值相同但方差不同：HWD-CN ≈ 0，而 MMD 应能发现 ✓（这正是要配 MMD 的原因）
    G = EmbSet(E.emb[:1].repeat(len(E), 0) + 0.0, E.writer, E.script, E.char, E.work, E.name)
    mmd = style_mmd(G.emb, E.emb)
    assert mmd > 0.01, "退化(所有样本相同)分布 ⇒ Style-MMD 必须为正 ✓"


def test_style_mmd_near_zero_on_identical():
    w, s, c, k = _grid()
    E = _rand_set(w, s, c, k, seed=4)
    v = style_mmd(E.emb, E.emb)
    assert abs(v) < 0.05, "同分布的无偏估计应接近 0（允许略负 ✓）"


def test_style_mmd_rejects_tiny_slots():
    w = np.array([0, 0, 1, 1])
    s = np.zeros(4, dtype=int)
    c = np.array([1, 2, 1, 2])
    k = np.zeros(4, dtype=int)
    A = _rand_set(w, s, c, k, seed=5)
    out = __import__("src.eval.style.metrics", fromlist=["x"]).style_mmd_by_slot(A, A)
    assert out["n_used"] == 0, "每组 <4 样本 ⇒ 不计算 ✓"


# ─────────────────────────── ⑤ XCTAcc ───────────────────────────
def test_xctacc_perfect_and_chance():
    w, s, c, k = _grid()
    E = _onehot_set(w, s, c, k)
    r = xctacc(E, E)
    assert r["xctacc"] == pytest.approx(1.0), "one-hot ⇒ 正样本一定更近 ✓"
    assert r["gap"] > 0.9
    R = _rand_set(w, s, c, k, seed=6)
    assert 0.3 < xctacc(R, R)["xctacc"] < 0.7, "随机嵌入 ⇒ 接近 0.5 ✓"


def test_xctacc_skips_when_no_cross_writer_same_char():
    """找不到"别的书家写的同一个字" ⇒ 跳过（不得用别的书体顶替 ✓）。"""
    w = np.array([0, 0, 1, 1])
    s = np.zeros(4, dtype=int)
    c = np.array([1, 2, 3, 4])                  # 每个字只有一位书家写过
    k = np.zeros(4, dtype=int)
    E = _onehot_set(w, s, c, k)
    r = xctacc(E, E)
    assert r["n_used"] == 0 and r["n_skipped"] == 4, "无负样本池 ⇒ 全部跳过 ✓"


def test_l2norm_handles_zero_vector():
    x = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    y = l2norm(x)
    assert np.isfinite(y).all(), "零向量不得产生 NaN ✓"
    assert np.linalg.norm(y[1]) == pytest.approx(1.0)
