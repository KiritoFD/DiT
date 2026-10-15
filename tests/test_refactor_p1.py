# -*- coding: utf-8 -*-
"""test_refactor_p1.py —— 2026-10-14 重构 P1 修复的**验收测试** ✓

覆盖:
  P1-1 模型生命周期: 构造后**禁止**在前向中创建/重建参数 ✗（必须报错 ✓）
  P1-3a checkpoint 严格模式: 写盘/校验失败 ⇒ **抛错**（调用方非零退出 ✓）
  P1-4 注释与实现一致（静态: 见 test_no_empty_preload_in_latent_dataset ✓）
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import ckpt as CK                                          # noqa: E402


# ───────────────────────── P1-1 参数生命周期 ─────────────────────────
class _FakeStyleHost:
    """只带 `_style_out` 需要的三个属性 ✓（不建真模型 ✓ 秒级 ✓）。"""

    def __init__(self, d_in, hidden=8, locked=False, proj_dim=None):
        self.style_ln_mod = torch.nn.LayerNorm(hidden)
        if proj_dim is None:
            self.style_in_proj = None
        elif proj_dim == hidden:
            self.style_in_proj = torch.nn.Identity()
        else:
            self.style_in_proj = torch.nn.Linear(proj_dim, hidden)
        self._style_proj_locked = locked
        self._d_in = d_in

    def _style_out(self, e):
        from src.model.dit import DiT_2Cond
        return DiT_2Cond._style_out(self, e)


def test_style_out_builds_during_construction():
    """构造期（未锁定 ✓）允许按真实维度定型 ✓"""
    h = _FakeStyleHost(d_in=5, hidden=8, locked=False, proj_dim=None)
    out = h._style_out(torch.zeros(2, 5))
    assert tuple(out.shape) == (2, 8)
    assert isinstance(h.style_in_proj, torch.nn.Linear)
    assert h.style_in_proj.in_features == 5


def test_style_out_raises_after_lock():
    """★ P1-1 ✓: 锁定后维度不符 ⇒ **报错** ✗ 绝不静默造参数 ✗"""
    h = _FakeStyleHost(d_in=5, hidden=8, locked=True, proj_dim=None)
    with pytest.raises(RuntimeError, match="禁止在前向中创建"):
        h._style_out(torch.zeros(2, 5))
    assert h.style_in_proj is None, "锁定后不得创建任何参数 ✗"


def test_style_out_raises_on_dim_change_after_lock():
    h = _FakeStyleHost(d_in=5, hidden=8, locked=True, proj_dim=5)   # 建好 5 -> 8 ✓
    assert tuple(h._style_out(torch.zeros(2, 5)).shape) == (2, 8)   # 匹配 ⇒ 正常 ✓
    with pytest.raises(RuntimeError, match="禁止在前向中创建"):
        h._style_out(torch.zeros(2, 7))                            # 维度变了 ⇒ 报错 ✗


def test_dit_locks_style_proj_after_init():
    """★ P1-1 ✓: `DiT_2Cond.__init__` 末尾必须设 `_style_proj_locked=True` ✓"""
    import io as _io
    src = _io.open("src/model/dit.py", encoding="utf-8").read()
    assert "self._style_proj_locked = True" in src, \
        "DiT_2Cond.__init__ 末尾必须锁定 style_in_proj ✗（P1-1）"


# ───────────────────────── P1-3a ckpt 严格模式 ─────────────────────────
def test_ckpt_strict_default_env(monkeypatch):
    monkeypatch.delenv("DSH_CKPT_STRICT", raising=False)
    assert CK._ckpt_strict_default() is True, "默认必须严格 ✓"
    for v in ("0", "false", "NO", "off"):
        monkeypatch.setenv("DSH_CKPT_STRICT", v)
        assert CK._ckpt_strict_default() is False, v
    monkeypatch.setenv("DSH_CKPT_STRICT", "1")
    assert CK._ckpt_strict_default() is True


class _FakeWriter:
    def __init__(self, errs):
        self._errs = errs

    def drain(self, timeout=None):
        return 0, list(self._errs)


def test_drain_ckpt_raises_in_strict(monkeypatch):
    """★ P1-3a ✓: 有失败 ⇒ 严格模式必须**抛错** ✗（否则"训练完成但模型没存" ✗✗）"""
    monkeypatch.setattr(CK, "WRITER", _FakeWriter([("/tmp/x.pt", "OSError: disk full")]))
    monkeypatch.delenv("DSH_CKPT_STRICT", raising=False)
    with pytest.raises(RuntimeError, match="写盘或校验失败"):
        CK.drain_ckpt(logger=None, timeout=1)


def test_drain_ckpt_returns_errors_when_not_strict(monkeypatch):
    monkeypatch.setattr(CK, "WRITER", _FakeWriter([("/tmp/x.pt", "boom")]))
    n, errs = CK.drain_ckpt(logger=None, timeout=1, strict=False)
    assert n == 0 and len(errs) == 1, errs


def test_drain_ckpt_ok_when_no_errors(monkeypatch):
    monkeypatch.setattr(CK, "WRITER", _FakeWriter([]))
    n, errs = CK.drain_ckpt(logger=None, timeout=1)
    assert n == 0 and errs == []


# ───────────────────────── P0-2 静态守卫（防回归 ✓）─────────────────────────
def test_no_np_empty_in_preload_paths():
    """★ P0-2 ✓: latent_dataset 的预加载路径**不得**再用 np.empty ✗
    （未初始化内存绝不允许进 batch ✓）"""
    import io as _io
    src = _io.open("src/utils/latent_dataset.py", encoding="utf-8").read()
    code = [ln for ln in src.split("\n") if not ln.strip().startswith("#")]
    hits = [ln.strip() for ln in code if "np.empty(" in ln]
    assert not hits, f"仍有 np.empty 预加载: {hits[:3]} ✗（P0-2 回归）"


def test_protocol_png_and_manifest_required():
    """★ P0-1 ✓: write_done 的必需产物检查存在（静态守卫防回归 ✓）"""
    import io as _io
    src = _io.open("src/eval/protocol.py", encoding="utf-8").read()
    assert "样本 manifest 缺失" in src and "PNG 数" in src
    assert "_png_inventory" in src
