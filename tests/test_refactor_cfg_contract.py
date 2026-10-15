# -*- coding: utf-8 -*-
"""test_refactor_cfg_contract.py —— **CFG 条件复制 / 置 null 契约**（审查 §4.1 ✓ 2026-10-14 ✓）

审查 §4.1 原文:
  "尤其应统一: `ConditionBatch(std_mask=…, writer_id=…, script_id=…, keep_masks=…)`
   不再在几十个函数之间传递 `g`、`y_char`、`y_callig`、`y_pair`、`y_script`、各种 dropout
   和路由标志。
   **注意必须明确定义 `y_callig_raw` 与 `y_pair` 的不同索引空间**，
   避免训练和评测对书家标签的解释**再次不一致**。" ✓

★ 为什么这条特别重要 ✗: 源码里已有一处**真实事故**的记载 ✓
  `dit_conditioning.py`: "★ 2026-10-13 [审查 P0]: 原来**漏传** y_script / y_callig_raw / y_pair ✗
   —— 与本方法 docstring 自己的要求相矛盾" ✓
  以及更早的: "y_callig_raw -> [real, null] ← **不置 null 的话 uncond 半仍带真书家，
   CFG 直接失效**" ✓
  ⇒ 这类错误**静默** ✓ 不报错 ✓ 只是"CFG 白做" ✗ ⇒ 必须有测试钉死 ✓

★ 本文件锁定**五条轴**各自不同的处置 ✓（这是最容易"顺手统一"而改坏的地方 ✗）:

  | 轴 | cond 半 | uncond 半 | 理由 |
  |---|---|---|---|
  | `y_callig` | 真 | **null** ✓ | 经典 CFG 的风格轴 ✓ |
  | `y_char` | 真 | **null** ✓ | 内容轴 ✓ |
  | `g`（标准字形）| 真 | **真** ✓✗ | "字形内容是**正条件**，CFG 只强化 callig 风格" ✓ |
  | `y_script` | 真 | **真** ✓✗ | "结构轴，与 g 同源" ✓ |
  | `y_callig_raw` | 真 | **null** ✓ | 风格轴 ✓ 不置 null ⇒ **CFG 直接失效** ✗✗ |
  | `y_pair` | 真 | **null** ✓ | 同上 ✓ |
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.dit_conditioning import forward_with_cfg                  # noqa: E402

B = 3


class _Embed:
    def __init__(self, n):
        self.num_classes = n


class _Hier:
    def __init__(self, n):
        self.num_pairs = n


class _RecModel:
    """记录被传入的**全部条件** ✓（正是我们要断言的东西 ✓）。"""

    def __init__(self, n_char=11, n_callig=7, n_pairs=5, image_channels=4):
        self.use_char_cond = True
        self.y_char_embedder = _Embed(n_char)
        self.y_callig_embedder = _Embed(n_callig)
        self.style_hier = _Hier(n_pairs)
        self.image_channels = image_channels
        self.seen: dict = {}
        self.delegated: dict = {}

    def forward(self, x, t, y_callig, y_char, g=None, y_script=None,
                y_callig_raw=None, y_pair=None):
        self.seen = dict(x=x, t=t, y_callig=y_callig, y_char=y_char, g=g,
                         y_script=y_script, y_callig_raw=y_callig_raw, y_pair=y_pair)
        # 输出 (2B, C)：前 B 行 = cond，后 B 行 = uncond ✓（数值上取 cond 即可 ✓）
        return torch.zeros(x.shape[0], self.image_channels)

    def forward_with_2axis_cfg(self, x, t, y_callig, y_char, **kw):
        self.delegated = dict(x=x, t=t, y_callig=y_callig, y_char=y_char, **kw)
        return torch.zeros(x.shape[0], self.image_channels)


def _inputs():
    x = torch.randn(B, 4, 8, 8)
    t = torch.rand(B)
    y_callig = torch.arange(B) % 7
    y_char = torch.arange(B) % 11
    g = torch.zeros(B, 1, 64, 64)
    y_script = torch.zeros(B, dtype=torch.long)
    y_callig_raw = torch.arange(B) % 7
    y_pair = torch.arange(B) % 5
    return x, t, y_callig, y_char, g, y_script, y_callig_raw, y_pair


def _run(**over):
    m = _RecModel()
    x, t, yc, ych, g, ys, ycr, yp = _inputs()
    kw = dict(g=g, y_script=ys, y_callig_raw=ycr, y_pair=yp, cfg_glyph=None)
    kw.update(over)
    forward_with_cfg(m, x, t, yc, ych, 2.0, **kw)
    return m


# ─────────────── ★ 五条轴各自的处置 ───────────────
def test_all_condition_axes_are_duplicated_to_2b():
    """★ 所有条件都必须复制成 **2B** ✓（否则 batch 不一致 ⇒ 后面 cat 报模糊的 size 错 ✗）"""
    m = _run()
    s = m.seen
    assert s["x"].shape[0] == 2 * B and s["t"].shape[0] == 2 * B
    for k in ("y_callig", "y_char", "g", "y_script", "y_callig_raw", "y_pair"):
        assert s[k] is not None and s[k].shape[0] == 2 * B, f"{k} 未复制成 2B ✗"


def test_style_axes_are_nulled_in_uncond_half():
    """★★ 风格轴（`y_callig` / `y_callig_raw` / `y_pair`）**必须**在 uncond 半置 null ✗✗

    源码原文: "不置 null 的话 uncond 半仍带真书家，**CFG 直接失效**" ✓
    """
    m = _run()
    s = m.seen
    for k, null_id in (("y_callig", m.y_callig_embedder.num_classes),
                       ("y_callig_raw", m.y_callig_embedder.num_classes),
                       ("y_pair", m.style_hier.num_pairs)):
        v = s[k]
        assert torch.equal(v[:B], _inputs()[{"y_callig": 2, "y_callig_raw": 6, "y_pair": 7}[k]]), \
            f"{k} 的 cond 半必须保留真值 ✓"
        assert bool((v[B:] == null_id).all()), \
            f"{k} 的 uncond 半**没有**置 null({null_id}) ✗✗ ⇒ CFG 直接失效 ✗"


def test_char_axis_is_nulled_in_uncond_half():
    """★ 内容轴 `y_char` 的 uncond 半置 null ✓（`num_classes` ✓）"""
    m = _run()
    v = m.seen["y_char"]
    assert bool((v[B:] == m.y_char_embedder.num_classes).all()), "y_char uncond 半必须置 null ✓"


def test_glyph_and_script_axes_are_NOT_nulled():
    """★★ `g` 与 `y_script` 的 uncond 半**必须保留真值** ✓✗（与上面相反 ✓）

    源码原文: "标准字形条件 g 始终全给(两半都用真实 g): **字形内容是正条件,
    CFG 只强化 callig 风格**" ✓ / "y_script -> 两半都给真实书体（结构轴，与 g 同源）" ✓

    ⚠ 这是**刻意的不对称** ✗ —— 若"顺手统一"成"所有轴都置 null" ✗
      ⇒ 内容轴被引入 CFG ✗ ⇒ 生成语义改变 ✗（且 doc54 的 "xattn strict≈0" 结论会失真 ✗）
    """
    m = _run()
    s = m.seen
    assert torch.equal(s["g"][:B], s["g"][B:]), "g 的两半必须**完全相同** ✓（不置 null ✓）"
    assert torch.equal(s["y_script"][:B], s["y_script"][B:]), \
        "y_script 的两半必须**完全相同** ✓（结构轴 ✓）"


def test_no_axis_left_as_none_when_provided():
    """★ 传了就**不能**在复制中变成 None ✗（2026-10-13 事故的形态 ✓）"""
    m = _run()
    for k in ("g", "y_script", "y_callig_raw", "y_pair"):
        assert m.seen[k] is not None, f"传入的 {k} 在 2 路路径里变成了 None ✗"


def test_none_axes_stay_none():
    """★ 没传就保持 None ✓（不得凭空造张量 ✗）"""
    m = _run(g=None, y_script=None, y_callig_raw=None, y_pair=None)
    for k in ("g", "y_script", "y_callig_raw", "y_pair"):
        assert m.seen[k] is None, f"未传入的 {k} 不应被创建 ✗"


# ─────────────── ★ 委托路径（2026-10-13 事故的**直接回归** ✓）───────────────
def test_delegation_passes_all_s2_labels():
    """★★ 走双轴 CFG 时，S2 三个标签**必须全部透传** ✓✗

    源码原文: "★ 2026-10-13 [审查 P0]: 原来**漏传** y_script / y_callig_raw / y_pair ✗
      —— 与本方法 docstring 自己的要求相矛盾（S2 条件必须随 y_callig 一起复制）" ✓
    ⇒ 本测试是那处修复的**回归护栏** ✓
    """
    m = _run(cfg_glyph=1.5)
    d = m.delegated
    assert d, "cfg_glyph 非 None 时必须**委托**给 forward_with_2axis_cfg ✓"
    for k in ("y_script", "y_callig_raw", "y_pair"):
        assert k in d, f"委托时**漏传** `{k}` ✗✗（2026-10-13 事故复发 ✗）"
        assert d[k] is not None, f"委托时 `{k}` 为 None ✗"
    assert d.get("cfg_glyph") == 1.5 and d.get("cfg_callig") == 2.0
    assert d.get("g") is not None


# ─────────────── ★ 入口 batch 一致性守卫 ───────────────
def test_batch_mismatch_raises_with_clear_message():
    """★ 入口 batch 不一致必须**明确报错** ✗（而不是留到 cat 处报模糊 size 错 ✗）"""
    m = _RecModel()
    x, t, yc, ych, g, ys, ycr, yp = _inputs()
    bad_t = torch.rand(B + 1)
    with pytest.raises(RuntimeError, match="入口 batch 不一致"):
        forward_with_cfg(m, x, bad_t, yc, ych, 2.0, g=g, cfg_glyph=None)
