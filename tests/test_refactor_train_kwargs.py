# -*- coding: utf-8 -*-
"""test_refactor_train_kwargs.py —— **训练侧条件装配迁移** 的验收测试（审查 §4.1 ✓ 2026-10-14 ✓）

审查 §4.1: 把散落的 `g`/`y_char`/`y_callig`/`y_pair`/`y_script` 收进 `ConditionBatch` ✓

★ 本轮迁移 ✓: `train.py` 的 `model_kwargs` **初始装配** ⇒
  `src/train/step.build_train_model_kwargs(args, batch, device, *, cond_mode,
   std_soft_cond, y_callig, y_char, y_script)` ✓
  · **逐字搬移** ✓（抽成函数只为**可测** ✓）
  · 后续的**条件 dropout / 形变增强**仍由调用方在同一 `dict` 上突变 ✓ 一字未改 ✓

★★ 两条最容易"顺手统一"而改坏的契约 ✗✗:
  ① `y_script` 的三种来源是**并集** ✓✗ —— 原代码在 hier 块里**又设了一次** ✓
     ⇒ `3cond` **或** `use_script_cond` **或** `hier_style>0` ⇒ 任一成立就带上 ✓
  ② `g` 的三路来源**顺序不变** ✓: `std_soft` → `g` → `skel_latent` ✓（首个命中即止 ✓）
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train.step import build_train_model_kwargs as B                   # noqa: E402


def _read_train_src() -> str:
    """训练路径的源码 = `train.py` + `trainer.py` ✓（2026-10-14 循环已搬到 trainer.py ✓）。

    ⚠ 守卫不应**绑死文件位置** ✗ —— 它要守的是"**这段代码存在且形状正确**" ✓
      （审查 §3: 迁移可保留兼容层 ✓ 同一精神 ✓）
    """
    import io as _io
    import os as _os
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    out = []
    for f in ("src/train/train.py", "src/train/trainer.py"):
        p = _os.path.join(root, f)
        if _os.path.exists(p):
            out.append(_io.open(p, encoding="utf-8").read())
    return "\n".join(out)


DEV = "cpu"
BATCH_N = 3


class _A:
    """最小 args ✓（缺省全关 ✓ 与历史行为一致 ✓）。"""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _batch(with_std_soft=True, with_g=True, with_skel=True, with_hier=True):
    b = {}
    if with_std_soft:
        b["std_soft"] = torch.ones(BATCH_N, 1, 4, 4)
    if with_g:
        b["g"] = torch.ones(BATCH_N, 4, 4, 4) * 2
    if with_skel:
        b["skel_latent"] = torch.ones(BATCH_N, 4, 4, 4) * 3
    if with_hier:
        b["y_callig_raw"] = torch.tensor([1, 2, 3])
        b["y_pair"] = torch.tensor([4, 5, 6])
        b["y_script"] = torch.tensor([7, 8, 9])
    return b


def _old_inline(args, batch, cond_mode, std_soft_cond, y_callig, y_char, y_script,
                device="cpu"):
    """**逐字复刻**旧内联装配 ✓（迁移前的写法 ✓）。"""
    if cond_mode == "3cond":
        mk = dict(y_callig=y_callig, y_script=y_script, y_char=y_char)
    else:
        mk = (dict(y_callig=y_callig, y_script=y_script, y_char=y_char)
              if getattr(args, "use_script_cond", False)
              else dict(y_callig=y_callig, y_char=y_char))
    if int(getattr(args, "hier_style", 0)) > 0:
        mk["y_callig_raw"] = batch["y_callig_raw"].to(device, non_blocking=True)
        mk["y_pair"] = batch["y_pair"].to(device, non_blocking=True)
        mk["y_script"] = batch["y_script"].to(device, non_blocking=True)
    if std_soft_cond and "std_soft" in batch and batch["std_soft"].numel() > 0:
        mk["g"] = batch["std_soft"].to(device, non_blocking=True).float()
    elif getattr(args, "w_glyph_cond", False) and "g" in batch and batch["g"].numel() > 0:
        mk["g"] = batch["g"].to(device, non_blocking=True)
    elif (getattr(args, "skel_as_glyph_cond", False) and "skel_latent" in batch
          and batch["skel_latent"].numel() > 0):
        mk["g"] = batch["skel_latent"].to(device, non_blocking=True).float()
    return mk


def _run(args, batch, cond_mode, std_soft_cond, y_script=None):
    yc, ych = torch.arange(BATCH_N), torch.arange(BATCH_N) + 10
    ys = y_script if y_script is not None else torch.arange(BATCH_N) + 20
    old = _old_inline(args, batch, cond_mode, std_soft_cond, yc, ych, ys)
    new = B(args, batch, DEV, cond_mode=cond_mode, std_soft_cond=std_soft_cond,
            y_callig=yc, y_char=ych, y_script=ys)
    return old, new


def _same(old, new):
    if sorted(old) != sorted(new):
        return False, f"键集合不同: {sorted(new)} vs {sorted(old)}"
    for k in old:
        if not torch.equal(old[k], new[k]):
            return False, f"{k} 值不同"
    return True, ""


# ─────────────── ★★ 穷举: 三来源 × 三路 × hier ✓ ───────────────
def test_matches_old_inline_exhaustively():
    combos = 0
    for use_script_cond in (False, True):
        for cond_mode in ("2cond", "3cond"):
            for hier in (0, 1):
                for std_soft_cond in (False, True):
                    for w_glyph in (False, True):
                        for skel_as_glyph in (False, True):
                            args = _A(use_script_cond=use_script_cond,
                                      hier_style=hier, w_glyph_cond=w_glyph,
                                      skel_as_glyph_cond=skel_as_glyph)
                            b = _batch()
                            old, new = _run(args, b, cond_mode, std_soft_cond)
                            ok, why = _same(old, new)
                            assert ok, (f"script={use_script_cond} cond={cond_mode} "
                                        f"hier={hier} stdsoft={std_soft_cond} "
                                        f"w_glyph={w_glyph} skel={skel_as_glyph}: {why}")
                            combos += 1
    assert combos == 64, combos


# ─────────────── ★★ y_script 三来源**并集** ✓ ───────────────
def test_y_script_sources_are_union():
    """★★ 任一来源成立就带上 `y_script` ✓✗（原代码在 hier 块里又设了一次 ✓）"""
    b = _batch()
    # ① 仅 3cond ✓
    kw = B(_A(), b, DEV, cond_mode="3cond", std_soft_cond=False,
           y_callig=torch.arange(3), y_char=torch.arange(3),
           y_script=torch.tensor([7, 8, 9]))
    assert "y_script" in kw, "3cond 必须带 y_script ✓"
    # ② 仅 use_script_cond ✓
    kw = B(_A(use_script_cond=True), b, DEV, cond_mode="2cond", std_soft_cond=False,
           y_callig=torch.arange(3), y_char=torch.arange(3),
           y_script=torch.tensor([7, 8, 9]))
    assert "y_script" in kw, "use_script_cond 必须带 y_script ✓"
    # ③ 仅 hier ✓
    kw = B(_A(hier_style=1), b, DEV, cond_mode="2cond", std_soft_cond=False,
           y_callig=torch.arange(3), y_char=torch.arange(3), y_script=None)
    assert "y_script" in kw and kw["y_script"].tolist() == [7, 8, 9], \
        "hier 必须从 batch 带 y_script ✓（即使调用方没传 ✓）"
    # ④ 三者全关 ⇒ **不带** ✓
    kw = B(_A(), b, DEV, cond_mode="2cond", std_soft_cond=False,
           y_callig=torch.arange(3), y_char=torch.arange(3),
           y_script=torch.tensor([7, 8, 9]))
    assert "y_script" not in kw, "三者全关时**不得**带 y_script ✗"


def test_hier_labels_only_when_enabled():
    b = _batch()
    kw = B(_A(hier_style=0), b, DEV, cond_mode="2cond", std_soft_cond=False,
           y_callig=torch.arange(3), y_char=torch.arange(3))
    assert "y_callig_raw" not in kw and "y_pair" not in kw, \
        "hier_style=0 ⇒ 三个 S2 标签都不带 ✓（零副作用 ✓）"
    kw = B(_A(hier_style=1), b, DEV, cond_mode="2cond", std_soft_cond=False,
           y_callig=torch.arange(3), y_char=torch.arange(3))
    assert kw["y_pair"].tolist() == [4, 5, 6] and kw["y_callig_raw"].tolist() == [1, 2, 3]


# ─────────────── ★★ g 三路来源**顺序** ✓ ───────────────
def test_g_source_priority():
    b = _batch()
    yc = ych = torch.arange(BATCH_N)
    # ① std_soft 优先 ✓（三路都开时也走它 ✓）
    kw = B(_A(w_glyph_cond=True, skel_as_glyph_cond=True), b, DEV, cond_mode="2cond",
           std_soft_cond=True, y_callig=yc, y_char=ych)
    assert kw["g"].shape == (BATCH_N, 1, 4, 4), "std_soft 优先级最高 ✓"
    # ② 无 std_soft ⇒ 走 g ✓
    kw = B(_A(w_glyph_cond=True, skel_as_glyph_cond=True), b, DEV, cond_mode="2cond",
           std_soft_cond=False, y_callig=yc, y_char=ych)
    assert kw["g"].shape == (BATCH_N, 4, 4, 4) and float(kw["g"][0, 0, 0, 0]) == 2.0, \
        "第二优先是 batch['g'] ✓"
    # ③ 只有 skel ⇒ 走 skel_latent ✓
    kw = B(_A(skel_as_glyph_cond=True), b, DEV, cond_mode="2cond",
           std_soft_cond=False, y_callig=yc, y_char=ych)
    assert float(kw["g"][0, 0, 0, 0]) == 3.0, "第三是 skel_latent ✓"
    # ④ 全关 ⇒ **不带 g** ✓✓（这是"不能用 for_forward=True"的原因 ✓）
    kw = B(_A(), b, DEV, cond_mode="2cond", std_soft_cond=False,
           y_callig=yc, y_char=ych)
    assert "g" not in kw, "无字形条件时**不得**带 g ✗（更不得带 g=None ✗）"


def test_empty_tensors_are_ignored():
    """★ 空张量 ⇒ 视为"没有该条件" ✓（`numel() > 0` 判定 ✓ 与旧一致 ✓）"""
    b = _batch()
    b["std_soft"] = torch.zeros(0)
    b["g"] = torch.zeros(0)
    b["skel_latent"] = torch.zeros(0)
    kw = B(_A(w_glyph_cond=True, skel_as_glyph_cond=True), b, DEV, cond_mode="2cond",
           std_soft_cond=True, y_callig=torch.arange(3), y_char=torch.arange(3))
    assert "g" not in kw, "全为空张量 ⇒ 不带 g ✓"


def test_missing_batch_keys_are_safe():
    b = {}
    kw = B(_A(hier_style=1, w_glyph_cond=True, skel_as_glyph_cond=True), b, DEV,
           cond_mode="2cond", std_soft_cond=True,
           y_callig=torch.arange(3), y_char=torch.arange(3))
    assert set(kw) == {"y_callig", "y_char"}, f"batch 缺键时不应崩也不应乱加 ✗: {sorted(kw)}"


# ─────────────── ★ 接线守卫 ✓ ───────────────
def test_train_py_uses_the_helper():
    src = _read_train_src()
    assert "model_kwargs = _build_train_model_kwargs(" in src, "未接线 ✗"
    assert "std_soft_cond=_std_soft_cond," in src
    # 旧内联必须消失 ✓
    assert "model_kwargs = dict(y_callig=y_callig, y_script=y_script, y_char=y_char)" not in src, \
        "旧内联仍在 ✗"
    # 后续的条件 dropout 突变**必须保留** ✓（本次刻意不动 ✓）
    assert "model_kwargs['g'] = random_skeleton_deformation(" in src, \
        "后续形变增强被误删 ✗（本次只迁初始装配 ✓）"
