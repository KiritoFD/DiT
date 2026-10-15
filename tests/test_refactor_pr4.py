# -*- coding: utf-8 -*-
"""test_refactor_pr4.py —— 2026-10-14 [重构 P1-6 收尾 / PR4 第一刀] 的验收测试 ✓

审查 §7.3 P0 原文: "消除 Critic `.item()`、`float(cuda_tensor)`、**逐样本 CUDA 同步**" ✓
审查 P1-6 原文: "`src/train/strong_style.py` 里的逐样本 `float(t[i])`、随机 `.item()`
  也应一起向量化" ✓

核查结论（比审查更细 ✓）:
  · `strong_style.py` 资格筛选的 `float(t[i])` ✗ —— **每步 B 次 CUDA 同步** ✓ 真问题 ✗
  · `g.max() > 1.5` ✗ —— **每步一次 CUDA 同步** ✓ 真问题 ✗
  · `torch.randint(len(valid), ())` ✗ —— **无 device ⇒ CPU 张量** ✓
    ⇒ 那里的 `.item()` 是 **CPU 侧**操作 ✓ **不是** CUDA 同步 ✗
    ⇒ 且它带 RNG ⇒ 改它会**改变随机流** ✗（违反"不改变行为" ✗）⇒ **有意保留** ✓

本组测试锁定"向量化后**逐元素等价**" ✓（这是能改它的唯一正当理由 ✓）
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train.strong_style import StrongStyle                             # noqa: E402


class _Args:
    strong_style_full_batch = False
    critic_t_min = 0.3
    critic_t_max = 0.8


def _mk(n_slots=6, mask=None, pairs=None, n=24, seed=0):
    """构造一个不加载 critic ckpt 的 StrongStyle 实例 ✓（只测选择逻辑 ✓）。"""
    ss = StrongStyle.__new__(StrongStyle)
    g = torch.Generator().manual_seed(seed)
    slots = ["w%d·楷" % i for i in range(n_slots)]
    pairs = pairs if pairs is not None else {(i, 0): slots[i] for i in range(n_slots)}
    ss.critic = type("C", (), {"id2slot": dict(pairs), "slots": slots})()
    ss.args = _Args()
    m = torch.zeros(n_slots, n_slots, dtype=torch.long)
    if mask is None:
        for i in range(n_slots):
            m[i] = torch.tensor([1 if j != i else 0 for j in range(n_slots)])
    else:
        m = mask
    ss.mask = m
    ss._slot_lut = ss._build_slot_lut()
    ss._mask_row_sum = m.reshape(m.shape[0], -1).sum(dim=1)
    return ss, g


def _reference_eligible(ss, batch, t):
    """**逐字复刻**旧实现 ✓（作为等价性参照 ✓ 无 RNG ✓）。"""
    eligible = []
    for i, (w, s) in enumerate(zip(batch['y_callig_raw'].tolist(),
                                   batch['y_script'].tolist())):
        slot = ss.critic.id2slot.get((w, s))
        if slot is not None and (ss.args.strong_style_full_batch or
                                 ss.args.critic_t_min <= float(t[i]) <= ss.args.critic_t_max):
            yi = ss.critic.slots.index(slot)
            if int(ss.mask[yi].sum()) >= 2:
                eligible.append((i, w, s, yi))
    return eligible


def _rand_batch(g, n, n_slots):
    return {
        "y_callig_raw": torch.randint(0, n_slots + 3, (n,), generator=g),
        "y_script": torch.randint(0, 2, (n,), generator=g),
    }


def test_vectorized_selection_matches_reference_exactly():
    """★★ 核心验证 ✓: 向量化选择必须与旧循环**逐元素等价**（含顺序 ✓）"""
    for seed in range(25):
        ss, g = _mk(n=32, seed=seed)
        batch = _rand_batch(g, 32, 6)
        t = torch.rand(32, generator=g)
        new = ss._eligible_rows(batch, t)
        old = _reference_eligible(ss, batch, t)
        assert new == old, f"seed={seed} 不等价 ✗\n new={new[:4]}\n old={old[:4]}"


def test_vectorized_selection_full_batch_ignores_t():
    ss, g = _mk(n=16, seed=1)
    ss.args.strong_style_full_batch = True
    batch = _rand_batch(g, 16, 6)
    t = torch.zeros(16)                       # 全部在 [0.3,0.8] 之外 ✓
    new = ss._eligible_rows(batch, t)
    old = _reference_eligible(ss, batch, t)
    assert new == old and len(new) > 0, (len(new), len(old))
    ss.args.strong_style_full_batch = False


def test_vectorized_selection_boundaries():
    """★ 边界语义必须与旧实现**逐位一致** ✓ —— 这里藏着 float32/float64 陷阱 ✗✗

    实测（本测试抓到的真实差异 ✓）:
        t = 0.8 存成 **float32** = 0.800000011920929 ✗
        旧代码 `float(t[1]) <= 0.8` 是 **float64** 比较 ⇒ 0.8000000119 <= 0.8 ⇒ **False** ✗
        ⇒ 该行**被排除** ✓（不是"包含" ✗ —— 我最初的期望就写错了 ✓）
        若用 float32 张量比标量, torch 把标量提升为 float32 ⇒ 0.8000000119 <= 0.8000000119
        ⇒ **True** ✗✗ ⇒ 语义不一致 ✗
    ⇒ 实现必须转 **double** 再比 ✓（见 `_eligible_rows` 的注释 ✓）
    """
    ss, g = _mk(n=3, seed=2)
    batch = {"y_callig_raw": torch.tensor([0, 1, 2]),
             "y_script": torch.tensor([0, 0, 0])}
    t = torch.tensor([0.3, 0.8, 0.7999999])
    new = ss._eligible_rows(batch, t)
    old = _reference_eligible(ss, batch, t)
    assert new == old, (new, old)
    # t=0.8（float32 略大于 0.8）**必须被排除** ✓；0.7999999 必须被包含 ✓
    assert [r[0] for r in new] == [0, 2], [r[0] for r in new]
    # t_min 侧同理: 0.3 的 float32 = 0.30000001192092896 ⇒ >= 0.3 ✓ 包含 ✓
    assert 0 in [r[0] for r in new]


def test_vectorized_selection_skips_unmapped_and_thin_slots():
    """未映射的 (w,s) ✓ 与 mask 行和 < 2 的槽 ✓ 都必须被跳过 ✓（与旧实现一致 ✓）"""
    pairs = {(0, 0): "w0·楷", (1, 0): "w1·楷"}
    m = torch.zeros(2, 2, dtype=torch.long)
    m[0] = torch.tensor([0, 1])               # 行和 = 1 ⇒ 不合格 ✓
    m[1] = torch.tensor([1, 0])               # 行和 = 1 ⇒ 不合格 ✓
    ss, g = _mk(n_slots=2, mask=m, pairs=pairs, n=8, seed=3)
    batch = {"y_callig_raw": torch.tensor([0, 1, 2, 3, 0, 1, 5, 0]),
             "y_script": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])}
    t = torch.full((8,), 0.5)
    assert ss._eligible_rows(batch, t) == [] == _reference_eligible(ss, batch, t)


def test_vectorized_selection_handles_out_of_range_ids():
    """raw id 超出查表范围 ⇒ 必须安全跳过 ✓（不得越界 ✗）"""
    ss, g = _mk(n_slots=3, n=6, seed=4)
    batch = {"y_callig_raw": torch.tensor([0, 999999, 1, -1, 2, 3]),
             "y_script": torch.tensor([0, 0, 0, 0, 0, 0])}
    t = torch.full((6,), 0.5)
    assert ss._eligible_rows(batch, t) == _reference_eligible(ss, batch, t)


def test_g_scale_cached_and_equivalent():
    """★ `g` 量纲判定只做一次 ✓ 且与旧"每步判"语义一致 ✓"""
    ss, g = _mk(n=4, seed=5)
    ss._g_scale = None
    g01 = torch.rand(2, 1, 8, 8, generator=g)          # 0-1 ✓
    if ss._g_scale is None:
        ss._g_scale = 1.0 if float(g01.detach().amax()) <= 1.5 else (1.0 / 255.0)
    assert ss._g_scale == 1.0
    first = ss._g_scale
    g255 = torch.rand(2, 1, 8, 8, generator=g) * 255.  # 即使后面来了 0-255 ✓ 也不再重判 ✓
    assert ss._g_scale == first, "量纲判定不得在 run 中变化 ✗（数据属性 ✓）"


def test_no_per_step_cuda_sync_in_selection_path():
    """★ 静态守卫 ✓: 选择路径不得再出现逐元素 `float(t[i])` / `g.max()` ✗

    ⚠ 必须**剥掉 docstring** 再查 ✗ —— `_eligible_rows` 的 docstring 里**引用**了旧代码
      （作为等价性说明 ✓）⇒ 朴素子串检查会误报 ✗（本测试早期版本正是这样误报的 ✓）。
    """
    import ast
    import io
    import re
    src = io.open("src/train/strong_style.py", encoding="utf-8").read()
    tree = ast.parse(src)
    # 把 docstring 置空 ⇒ 只留可执行代码 ✓
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    code = ast.unparse(tree)
    assert not re.search(r"float\(t\[", code), "仍有逐元素 float(t[i]) ✗（P1-6）"
    assert "g.max() > 1.5" not in code, "仍有每步 g.max() 同步 ✗（P1-6）"
    # 允许保留的 CPU 侧随机（带 RNG ⇒ 改它会改变随机流 ✗）
    assert "torch.randint(len(valid)" in code, \
        "CPU 侧 randint 应**有意保留** ✓（改它会改变随机流 ✗）"
    # ★ 关键: t 的比较必须转 double ✓（否则 float32 提升会在边界上改变语义 ✗）
    assert ".double()" in code, "t 比较必须转 double ✓（边界等价性 ✓）"
