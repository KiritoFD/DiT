# -*- coding: utf-8 -*-
"""test_refactor_condition_batch.py —— **ConditionBatch 兼容层契约**（审查 §4.1 ✓ 2026-10-14 ✓）

审查 §4.1 原文:
  "尤其应统一: `ConditionBatch(std_mask=…, writer_id=…, script_id=…, keep_masks=…)`
   不再在几十个函数之间传递 `g`、`y_char`、`y_callig`、`y_pair`、`y_script`、
   各种 dropout 和路由标志。" ✓
  "迁移可以保留旧导入路径作为**短期兼容层**。" ✓

★ 本文件是 PR5（拆条件系统）的**安全网** ✓:
  `src/model/conditioning.py` 已经提供了打包/解包 ✓ —— 但要**逐个**迁移调用点 ✓
  ⇒ 迁移过程中最大的风险是**兼容映射悄悄错位** ✗（改了字段名/漏了一个标签 ✓）
  ⇒ 用**穷举往返** + **跨模块一致性**把它钉死 ✓

★★ 最微妙的一条契约 ✗（源码原文 ✓）:
  "显式传 None 与'不传'在 forward 里等价 ✓，但 **`g=None` 是"条件被丢掉"的语义** ✓，
   所以需要表达"明确不带 g"时必须走 `for_forward=False` 的完整映射 ✓。"
  ⇒ `batch_to_kwargs(..., for_forward=True)` **必须保留** `g`（即使它是 None ✓）
    而**其它** None 键应被丢弃 ✓ —— 这两条方向相反 ✓ 极易"顺手统一"改坏 ✗
"""
from __future__ import annotations

import inspect
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.conditioning import (ConditionBatch, batch_to_kwargs,      # noqa: E402
                                    build_batch, call_forward)
from src.model.dit_conditioning import forward_with_cfg                  # noqa: E402

# 六个历史关键字 ⇄ ConditionBatch 字段（**唯一的映射真相** ✓）
LEGACY = {
    "y_callig": "writer",
    "y_char": "char",
    "g": "glyph",
    "y_script": "script",
    "y_callig_raw": "writer_raw",
    "y_pair": "writer_script_pair",
}


def _val(k, i=0):
    if k == "g":
        return torch.full((2, 1, 4, 4), float(i))
    return torch.full((2,), i, dtype=torch.long)


# ─────────────── ① ★★ 穷举往返: 64 种组合全部无损 ✓ ───────────────
def test_roundtrip_is_lossless_exhaustively():
    """★★ 对 6 个标签的**全部 2^6 = 64 种**存在/缺失组合，往返必须无损 ✓"""
    keys = list(LEGACY)
    for mask in range(1 << len(keys)):
        kw = {k: (_val(k, i) if (mask >> i) & 1 else None)
              for i, k in enumerate(keys)}
        b = build_batch(**kw)
        back = b.to_kwargs()
        assert set(back) == set(LEGACY), f"映射键集合不对 ✗: {sorted(back)}"
        for k in keys:
            v0, v1 = kw[k], back[k]
            if v0 is None:
                assert v1 is None, f"mask={mask}: {k} 应从 None 回来仍是 None ✗"
            else:
                assert v1 is not None and torch.equal(v0, v1), \
                    f"mask={mask}: {k} 往返丢失/改变 ✗"


def test_field_mapping_is_exact():
    """★ 字段映射必须**逐个**对 ✓（不是"大概对" ✓）"""
    for legacy, field in LEGACY.items():
        b = build_batch(**{legacy: _val(legacy)})
        assert getattr(b, field) is not None, f"{legacy} 应映射到 {field} ✗"
        assert b.to_kwargs()[legacy] is not None
    # 反向: 给字段 ⇒ 只在对应 legacy 键出现 ✓
    b = ConditionBatch(writer=_val("y_callig"))
    kw = b.to_kwargs()
    assert kw["y_callig"] is not None
    for k in LEGACY:
        if k != "y_callig":
            assert kw[k] is None, f"未设置的 {k} 不应被填上 ✗"


def test_present_reflects_labels():
    b = build_batch(y_callig=_val("y_callig"), g=_val("g"))
    p = b.present()
    assert p == {"writer": True, "script": False, "char": False, "glyph": True,
                 "writer_raw": False, "writer_script_pair": False}, p


def test_content_mode_is_carried():
    assert build_batch(content_mode="std").content_mode == "std"
    assert build_batch().content_mode is None


# ─────────────── ② ★★ 跨模块一致性（防改名漂移 ✓）───────────────
def test_legacy_keys_match_forward_signature():
    """★★ `to_kwargs()` 的键必须**恰好**是 forward 接受的历史关键字 ✓

    ⇒ 任一侧改名/漏项 ⇒ 立刻失败 ✓（这类漂移是**静默**的 ✗ 只在运行到某分支才炸 ✗）
    """
    sig = inspect.signature(forward_with_cfg)
    accepted = {p.name for p in sig.parameters.values()
                if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    for k in LEGACY:
        assert k in accepted, f"`forward_with_cfg` 不再接受 `{k}` ✗（映射已失效 ✗）"
    # 且六个标签必须**都在** forward 签名里 ✓（不能少 ✗）
    assert set(LEGACY) <= accepted


def test_legacy_keys_match_dit_forward_signature():
    """★ 同上，对 `DiT_2Cond.forward` 也查一遍 ✓（两个入口都要一致 ✓）"""
    from src.model.dit import DiT_2Cond
    sig = inspect.signature(DiT_2Cond.forward)
    accepted = set(sig.parameters)
    for k in LEGACY:
        assert k in accepted, f"`DiT_2Cond.forward` 不再接受 `{k}` ✗"


# ─────────────── ③ ★★ for_forward 的方向相反的两条规则 ───────────────
def test_for_forward_keeps_g_even_when_none():
    """★★ `g=None` 是**"条件被丢掉"的语义** ✓ ⇒ 必须**保留** `g` 键 ✗

    源码原文: "需要表达'明确不带 g'时必须走 for_forward=False 的完整映射" ✓
    """
    b = build_batch(y_callig=_val("y_callig"), g=None)
    kw = batch_to_kwargs(b, for_forward=True)
    assert "g" in kw, "for_forward=True 时 `g` 键**必须保留** ✗（哪怕值是 None ✓）"
    assert kw["g"] is None


def test_for_forward_drops_other_none_keys():
    """★ 其它标签的 None 应被**丢弃** ✓（与 `g` 相反 ✓ 同一函数里两个方向 ✗）"""
    b = build_batch(y_callig=_val("y_callig"), y_char=None, y_script=None,
                    y_callig_raw=None, y_pair=None, g=_val("g"))
    kw = batch_to_kwargs(b, for_forward=True)
    assert set(kw) == {"y_callig", "g"}, f"应只留非 None + g ✗: {sorted(kw)}"


def test_for_forward_false_returns_full_mapping():
    """★ `for_forward=False` ⇒ **完整**六个键 ✓（表达"明确不带" ✓）"""
    b = build_batch()
    kw = batch_to_kwargs(b, for_forward=False)
    assert set(kw) == set(LEGACY), f"完整映射应含六个键 ✗: {sorted(kw)}"


# ─────────────── ④ call_forward ───────────────
def test_call_forward_passes_kwargs_and_extras():
    seen = {}

    def _model(x, t, **kw):
        seen.update(kw)
        seen["_x"] = x
        return "ok"

    x, t = torch.zeros(2, 4, 8, 8), torch.zeros(2)
    b = build_batch(y_callig=_val("y_callig"), g=_val("g"), content_mode="std")
    assert call_forward(_model, x, t, b, cfg_scale=2.0) == "ok"
    assert seen["y_callig"] is not None and seen["g"] is not None
    assert seen["cfg_scale"] == 2.0, "extra 必须透传 ✓"
    assert seen["_x"] is x
