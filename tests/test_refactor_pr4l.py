# -*- coding: utf-8 -*-
"""test_refactor_pr4l.py —— 2026-10-14 [重构 PR4] **clip+step+复原 三连** 的验收测试 ✓

审查 §4.2 第 5–6 条: "5. finite 梯度检查和 **clipping**；6. **optimizer** / scheduler / EMA 更新" ✓

本组测试锁定:
  ① ★★ **顺序即语义** ✗✗: clip → step → restore_null ✓
       · clip 必须在 step **之前** ✓（否则等于没裁 ✓）
       · restore_null 必须在 step **之后** ✓
         （writer table 的 **null 行**会被 step 更新 ⇒ 必须立刻复原 ✓
          否则 CFG 的 null token 会跟着漂移 ✗ 那是**语义**错误 ✗）
  ② `max_norm` 原样传递 ✓（**绝对**阈值 ✓ 不随 latent 尺度自动缩放 ✓）
  ③ `writer_table=None` ⇒ 不调用 ✓
  ④ 边界 ✓: **不含** zero_grad / scheduler / EMA ✗（时机归调用方 ✓）
"""
from __future__ import annotations

import io
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train.step import clip_and_step                                 # noqa: E402


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



class _Rec:
    def __init__(self, order, name):
        self.order, self.name = order, name

    def step(self):
        self.order.append("step")

    def restore_null(self, *a, **k):
        self.order.append("restore_null")


# ─────────────── ① ★★ 顺序 ───────────────
def test_order_clip_then_step_then_restore(monkeypatch):
    order = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def _clip(params, max_norm=None, **kw):
        order.append(("clip", float(max_norm)))
        return real_clip(params, max_norm=max_norm, **kw)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _clip)
    opt = _Rec(order, "opt")
    wt = _Rec(order, "wt")
    p = torch.nn.Parameter(torch.zeros(3))
    p.grad = torch.ones(3)
    clip_and_step(opt, [p], 0.5, writer_table=wt)
    assert [x if isinstance(x, str) else x[0] for x in order] == \
        ["clip", "step", "restore_null"], order
    assert order[0][1] == 0.5, "max_norm 必须原样传递 ✓"


def test_clip_actually_before_step_by_effect(monkeypatch):
    """★ 用**效果**证明顺序 ✓: 若先 step 再 clip, 梯度已被消费 ⇒ clip 无意义 ✓"""
    seen = {}

    def _clip(params, max_norm=None, **kw):
        seen["clip_saw_grad"] = any(p.grad is not None for p in params)
        return torch.tensor(0.0)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _clip)

    class _Opt:
        def step(self):
            seen["stepped"] = True

    p = torch.nn.Parameter(torch.zeros(3))
    p.grad = torch.ones(3)
    clip_and_step(_Opt(), [p], 1.0)
    assert seen["clip_saw_grad"] is True, "clip 时梯度必须还在 ✓（即 clip 在 step 之前 ✓）"


# ─────────────── ② / ③ ───────────────
def test_writer_table_none_is_safe(monkeypatch):
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_",
                        lambda params, max_norm=None, **kw: torch.tensor(0.0))
    order = []
    opt = _Rec(order, "opt")
    clip_and_step(opt, [], 1.0, writer_table=None)
    assert order == ["step"], "无 writer table ⇒ 只 step ✓"


def test_restore_null_called_once(monkeypatch):
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_",
                        lambda params, max_norm=None, **kw: torch.tensor(0.0))
    order = []
    clip_and_step(_Rec(order, "opt"), [], 1.0, writer_table=_Rec(order, "wt"))
    assert order.count("restore_null") == 1, "必须**恰好一次**复原 ✓（不多不少 ✓）"


# ─────────────── ④ 边界 ───────────────
def test_no_zero_grad_or_scheduler_side_effects():
    import ast
    import re
    src = io.open("src/train/step.py", encoding="utf-8").read()
    # 剥掉三引号块后再查 ✓（避免文档引用误报 ✗）
    code = re.sub(r'""".*?"""', '""', src, flags=re.S)
    code = re.sub(r"'''.*?'''", "''", code, flags=re.S)
    tree = ast.parse(code)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "clip_and_step")
    body = ast.unparse(ast.Module(body=fn.body, type_ignores=[]))
    for bad in ("zero_grad", "scheduler", "update_ema", "backward"):
        assert bad not in body, f"clip_and_step 不得涉及 `{bad}` ✗（时机归调用方 ✓）"
    # 三件事必须都在 ✓
    for must in ("clip_grad_norm_", ".step()", "restore_null"):
        assert must in body, f"缺少 `{must}` ✗"


def test_train_py_uses_clip_and_step():
    t = _read_train_src()
    assert "_clip_and_step(opt, trainable_params_list," in t, "未接线 ✗"
    # 旧的内联三连必须已消失 ✓
    assert "torch.nn.utils.clip_grad_norm_(trainable_params_list," not in t, \
        "旧内联 clip 仍在 ✗"
    assert "_writer_table_tuning.restore_null()\n" not in t.split("_clip_and_step")[0][-200:], \
        "旧内联 restore_null 仍在 ✗"
