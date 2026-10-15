# -*- coding: utf-8 -*-
"""test_refactor_sampler_kwargs.py —— **PR5 第一个调用点迁移** 的验收测试（2026-10-14 ✓）

审查 §4.1: 把散落的 `g`/`y_char`/`y_callig`/`y_pair`/`y_script` 收进 `ConditionBatch` ✓

★ 本轮迁移 ✓: `src/eval/sampler.py` 的内联装配 ⇒ **可测助手**
  `model_kwargs_for_slice(model, yc=…, yh=…, hier_slice=…, skel_slice=…, device=…)` ✓
  逻辑**逐字搬移** ✓（含 g/cond 包装器判定 ✓）—— 抽成函数只为**可测** ✓

★★ 必须用 `kwargs_for_call` ✗ —— **不能**用 `batch_to_kwargs(for_forward=True)` ✓
  后者**总是**保留 `g`（契约: `g=None` = "**条件被丢掉**" ✓）
  ⇒ 会把"没传 g"变成"明确传 g=None" ✗ ⇒ **语义改变** ✗
  （实测: `['y_callig','y_char']` → `['g','y_callig','y_char']` ✓ 见
    tests/test_refactor_migration_rule.py ✓）

★ `cond` 键（ControlNet 包装 ✓）**不在** `ConditionBatch` 里 ✗
  ⇒ 按契约**留在调用方**（本助手内 ✓）
"""
from __future__ import annotations

import io
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.sampler import model_kwargs_for_slice                     # noqa: E402

B = 3


class _Plain:
    """无 `use_glyph_cond` / 无 `std_soft_cond` ⇒ 判定走 **`cond`** ✓"""


class _Glyph:
    use_glyph_cond = True


def _old_inline(model, yc, yh, hier=None, skel=None, device="cpu"):
    """**逐字复刻**旧内联装配 ✓（迁移前的写法 ✓）。"""
    mk = dict(y_callig=yc, y_char=yh)
    if hier is not None:
        _hraw, _hpair, _hscript = hier
        mk["y_callig_raw"] = torch.tensor(_hraw, device=device, dtype=torch.long)
        mk["y_pair"] = torch.tensor(_hpair, device=device, dtype=torch.long)
        mk["y_script"] = torch.tensor(_hscript, device=device, dtype=torch.long)
    if skel is not None:
        _use_g = bool(getattr(model, "use_glyph_cond", False))
        mk["g" if _use_g else "cond"] = skel
    return mk


def _inputs():
    yc = torch.arange(B)
    yh = torch.arange(B) + 10
    skel = torch.ones(B, 1, 4, 4)
    # ⚠ 真实路径传的是 **list 切片** ✓（不是张量 ✓ 否则会触发 torch.tensor(tensor) 警告 ✗）
    hier = ([1, 2, 3], [4, 5, 6], [7, 8, 9])
    return yc, yh, skel, hier


# ─────────────── ★★ 与旧内联逐字等价（四种组合 ✓）───────────────
def test_equivalent_to_old_inline_all_combinations():
    yc, yh, skel, hier = _inputs()
    for tag, model in (("plain", _Plain()), ("glyph", _Glyph())):
        for use_hier in (False, True):
            for use_skel in (False, True):
                h = hier if use_hier else None
                s = skel if use_skel else None
                old = _old_inline(model, yc, yh, h, s)
                new = model_kwargs_for_slice(
                    model, yc=yc, yh=yh,
                    hier_slice=(h if use_hier else None),
                    skel_slice=(s if use_skel else None))
                assert sorted(new) == sorted(old), \
                    f"{tag}/hier={use_hier}/skel={use_skel}: 键集合不同 ✗ {sorted(new)} vs {sorted(old)}"
                for k in old:
                    assert torch.equal(new[k], old[k]), \
                        f"{tag}/hier={use_hier}/skel={use_skel}: {k} 值不同 ✗"


# ─────────────── ★★ 关键: 不传 skel ⇒ 不得出现 g/cond ✗ ───────────────
def test_no_skel_means_no_g_and_no_cond():
    """★★ 不传字形时**必须一个都不加** ✗ —— 这正是"不能直接用 for_forward=True"的原因 ✓"""
    yc, yh, _, _ = _inputs()
    for model in (_Plain(), _Glyph()):
        kw = model_kwargs_for_slice(model, yc=yc, yh=yh)
        assert set(kw) == {"y_callig", "y_char"}, \
            f"不传 skel 时不应出现 g/cond ✗: {sorted(kw)}"
        assert "g" not in kw and "cond" not in kw


def test_plain_model_gets_cond_not_g():
    yc, yh, skel, _ = _inputs()
    kw = model_kwargs_for_slice(_Plain(), yc=yc, yh=yh, skel_slice=skel)
    assert "cond" in kw and "g" not in kw, sorted(kw)


def test_glyph_model_gets_g_not_cond():
    yc, yh, skel, _ = _inputs()
    kw = model_kwargs_for_slice(_Glyph(), yc=yc, yh=yh, skel_slice=skel)
    assert "g" in kw and "cond" not in kw, sorted(kw)


def test_wrapper_with_main_attr_is_detected():
    """★ 包装器（`model.main.use_glyph_cond` ✓）也要被识别 ✓（历史 ckpt 兼容 ✓）"""
    class _Main:
        use_glyph_cond = True

    class _Wrap:
        main = _Main()

    yc, yh, skel, _ = _inputs()
    kw = model_kwargs_for_slice(_Wrap(), yc=yc, yh=yh, skel_slice=skel)
    assert "g" in kw, "包装器的 main.use_glyph_cond 未被识别 ✗"


def test_signature_accepts_cond_only_model():
    """★ 若 forward **只**接受 `g` 而不接受 `cond` ⇒ 判定用 `g` ✓（签名兜底 ✓）"""
    class _Sig:
        def forward_with_cfg(self, x, t, y_callig=None, y_char=None, g=None):
            return None

    yc, yh, skel, _ = _inputs()
    kw = model_kwargs_for_slice(_Sig(), yc=yc, yh=yh, skel_slice=skel)
    assert "g" in kw, "签名兜底（'g' in params and 'cond' not in params）未生效 ✗"


# ─────────────── ★ hier 三标签 ✓ ───────────────
def test_hier_labels_only_when_provided():
    yc, yh, _, hier = _inputs()
    kw = model_kwargs_for_slice(_Plain(), yc=yc, yh=yh, hier_slice=hier)
    assert set(kw) >= {"y_callig_raw", "y_pair", "y_script"}
    assert kw["y_pair"].dtype == torch.long and kw["y_pair"].tolist() == [4, 5, 6]
    assert kw["y_script"].tolist() == [7, 8, 9]


# ─────────────── ★ 接线守卫 ✓ ───────────────
def test_sampler_uses_the_helper():
    import re
    src = io.open("src/eval/sampler.py", encoding="utf-8").read()
    assert "mk = model_kwargs_for_slice(" in src, "循环体未改用助手 ✗"
    assert "def model_kwargs_for_slice(" in src
    assert "kwargs_for_call(_batch)" in src, "必须用 kwargs_for_call ✓（不是 for_forward=True ✗）"
    # ⚠ 必须**剥掉三引号块**再查 ✗ —— 助手文档里**引用**了 `batch_to_kwargs` 作为反例 ✓
    #   （"不能用它"的说明 ✓）⇒ 朴素子串检查会误报 ✗（实测如此 ✓）
    code = re.sub(r'""".*?"""', '""', src, flags=re.S)
    code = re.sub(r"'''.*?'''", "''", code, flags=re.S)
    assert "batch_to_kwargs" not in code, \
        "sampler **代码里**不应使用 batch_to_kwargs ✗（其 for_forward=True 会凭空加 g ✗）"
    assert 'mk = dict(y_callig=yc, y_char=yh)' not in code, "旧内联仍在 ✗"
