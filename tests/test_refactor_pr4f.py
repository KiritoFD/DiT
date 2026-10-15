# -*- coding: utf-8 -*-
"""test_refactor_pr4f.py —— 2026-10-14 [重构 PR4] **optimizer 组件** 的验收测试 ✓

审查 §4.2 把 optimizer 列为需要"明确状态与生命周期"的组件之一 ✓

本组测试锁定:
  ① `fused_supported()` ≡ 旧的**内联**判定（逐字 ✓）
  ② `build_adamw` ≡ 旧的**内联**构造 ✓（含 TypeError 回退 ✓）
  ③ ★★ `build_adamw_groups` **必须保留各组自带的 lr/wd** ✗✗
     —— 这是 VAE 双轨的**全部意义** ✓（DiT/REPA/VAE 三组 lr 不同 ✓）
        我在写补丁时差点让工厂统一传 lr/wd ✗ 那会把三组覆盖成同一个值 ✗✗
        ⇒ 用测试把它钉死 ✓
  ④ `build_muon` ≡ 旧构造 ✓（含导入失败的报错文本 ✓）
  ⑤ 身份摘要 ✓ / 接线守卫 ✓ / 无顺序副作用 ✓
"""
from __future__ import annotations

import io
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import optimizer as O                                       # noqa: E402


class _Args:
    def __init__(self, **kw):
        self.optimizer = kw.pop("optimizer", "adamw")
        self.lr = kw.pop("lr", 5e-5)
        self.weight_decay = kw.pop("weight_decay", 0.01)
        for k, v in kw.items():
            setattr(self, k, v)


# ─────────────── ① fused 判定逐字等价 ───────────────
def test_fused_supported_matches_old_inline():
    old = bool(torch.cuda.is_available()) and float(torch.__version__[:3]) >= 2.0
    assert O.fused_supported() is bool(old)


def test_fused_supported_is_single_source_of_truth():
    """★ 守卫 ✓: train.py 里**不得**再有内联 fused 判定 ✗（旧代码写了两遍 ✗）"""
    t = io.open("src/train/train.py", encoding="utf-8").read()
    assert "float(torch.__version__[:3]) >= 2.0" not in t, \
        "train.py 仍有内联 fused 判定 ✗（应统一走 optimizer.fused_supported ✓）"


# ─────────────── ② build_adamw ≡ 旧构造 ───────────────
def test_build_adamw_matches_old_construction():
    ps = [torch.nn.Parameter(torch.zeros(2))]
    opt, fused = O.build_adamw(ps, lr=1e-4, weight_decay=0.02)
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == 1e-4
    assert opt.param_groups[0]["weight_decay"] == 0.02
    assert fused is O.fused_supported()


def test_build_adamw_typeerror_fallback(monkeypatch):
    """★ 老 torch 不认 `fused=` ⇒ 必须回退且把 `fused` 置 False ✓（与旧代码一致 ✓）"""
    real = torch.optim.AdamW
    calls = []

    def _fake(params, **kw):
        calls.append(kw)
        if "fused" in kw:
            raise TypeError("unexpected keyword argument 'fused'")
        return real(params, **kw)

    monkeypatch.setattr(torch.optim, "AdamW", _fake)
    ps = [torch.nn.Parameter(torch.zeros(2))]
    opt, fused = O.build_adamw(ps, lr=1e-3, weight_decay=0.0)
    assert fused is False, "回退后 fused 必须为 False ✓"
    assert len(calls) == 2 and "fused" in calls[0] and "fused" not in calls[1]
    assert isinstance(opt, real)


# ─────────────── ③ ★★ 组内 lr/wd 必须保留 ───────────────
def test_build_adamw_groups_preserves_per_group_lr_and_wd():
    """★★ VAE 双轨的**全部意义** ✓: 三组各自 lr/wd 不得被统一覆盖 ✗✗

    （我在写补丁时差点让工厂传 `lr=` / `weight_decay=` ✗
      ⇒ 那会把 DiT/REPA/VAE 三组覆盖成同一个 lr ✗✗ ⇒ 用测试钉死 ✓）
    """
    p = [torch.nn.Parameter(torch.zeros(2)) for _ in range(3)]
    groups = [
        {"params": [p[0]], "lr": 5e-5, "weight_decay": 0.01},   # DiT ✓
        {"params": [p[1]], "lr": 5e-5, "weight_decay": 0.0},    # REPA ✓
        {"params": [p[2]], "lr": 2e-6, "weight_decay": 0.0},    # VAE ✓
    ]
    opt, _f = O.build_adamw_groups(groups)
    assert [g["lr"] for g in opt.param_groups] == [5e-5, 5e-5, 2e-6], \
        "各组 lr 被覆盖 ✗✗（VAE 双轨失效 ✗）"
    assert [g["weight_decay"] for g in opt.param_groups] == [0.01, 0.0, 0.0], \
        "各组 weight_decay 被覆盖 ✗✗"


def test_build_adamw_groups_signature_has_no_lr_kwargs():
    """★ 结构性守卫 ✓: `build_adamw_groups` **不得**接受 lr / weight_decay ✗"""
    import inspect
    sig = inspect.signature(O.build_adamw_groups)
    assert "lr" not in sig.parameters, "不得暴露 lr 参数 ✗（会诱导覆盖各组 ✗）"
    assert "weight_decay" not in sig.parameters, "不得暴露 weight_decay 参数 ✗"


# ─────────────── ④ build_muon ≡ 旧构造 ───────────────
def test_build_muon_delegates_with_old_defaults():
    try:
        from src.optim.muon import Muon
    except Exception:                                                  # noqa: BLE001
        pytest.skip("src.optim.muon 不可用")
    ps = [torch.nn.Parameter(torch.randn(4, 4))]
    a = _Args(optimizer="muon", muon_lr=0.03, lr=1e-4, weight_decay=0.02)
    opt = O.build_muon(ps, a)
    assert isinstance(opt, Muon)


def test_build_muon_error_text_matches_old():
    src = io.open("src/train/optimizer.py", encoding="utf-8").read()
    assert 'f"src.optim.muon import failed: {e}"' in src, \
        "导入失败的报错文本必须与旧实现逐字一致 ✓（便于检索历史日志 ✓）"


# ─────────────── ⑤ 身份与接线 ───────────────
def test_identity_and_describe():
    a = _Args(optimizer="adamw", lr=5e-5, weight_decay=0.01, train_vae_encoder=True, vae_lr=2e-6)
    d = O.identity(a)
    assert d["name"] == "adamw" and d["lr"] == 5e-5 and d["weight_decay"] == 0.01
    assert d["train_vae_encoder"] is True and d["vae_lr"] == 2e-6
    s = O.describe(a)
    for tok in ("name=adamw", "lr=5e-05", "wd=0.01", "fused=", "vae_lr=2e-06"):
        assert tok in s, (tok, s)


def test_resolve_name_defaults_and_lowercase():
    class _B:
        pass
    assert O.resolve_name(_B()) == "adamw", "缺省必须 adamw ✓"
    assert O.resolve_name(_Args(optimizer="Muon")) == "muon", "必须小写归一 ✓"


def test_train_py_uses_factories():
    t = io.open("src/train/train.py", encoding="utf-8").read()
    assert "_opt.build_adamw_groups(optim_groups)" in t, "VAE 双轨未接线 ✗"
    assert "_opt.build_adamw(trainable_params_list" in t, "普通 AdamW 未接线 ✗"
    assert "_opt.build_muon(" in t, "Muon 未接线 ✗"
    assert "[optim-identity]" in t, "启动日志必须打印优化器身份 ✓（审查 P1-7 ✓）"
    assert "= _Muon(" not in t, "train.py 不应再直接构造 Muon ✗"


def test_module_has_no_ordering_side_effects():
    """★ 边界 ✓: 本模块**只**做构造 ✓ 不做组拆分/模型突变/调度器联动 ✗
    （那些有**顺序约束** ✓ 必须留在调用方 ✓）

    ⚠ 必须**剥掉 docstring** 再查 ✗ —— 模块文档里**引用**了这些名字
      （作为"不在本模块做"的说明 ✓）⇒ 朴素子串检查会误报 ✗（实测正是如此 ✓）
    """
    import ast
    src = io.open("src/train/optimizer.py", encoding="utf-8").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    code = ast.unparse(tree)
    for bad in ("requires_grad_", "param_groups", "base_lrs", "load_state_dict",
                "split_writer_optimizer_group", "scheduler"):
        assert bad not in code, f"optimizer.py 不得涉及 `{bad}` ✗（顺序约束留在调用方 ✓）"
