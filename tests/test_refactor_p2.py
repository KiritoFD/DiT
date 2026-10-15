# -*- coding: utf-8 -*-
"""test_refactor_p2.py —— 2026-10-14 重构 P1-2 的**验收测试** ✓

P1-2: checkpoint 必须保存**全部随机源** ✓ ⇒ 恢复后**逐步、逐样本可重放** ✓
     （审查: 旧 ckpt 只存 model/opt/ema/scheduler/args/step ✗ ⇒ 只能"近似续训" ✗
       而本项目经常比较 5k–10k 的**微小增益** ⇒ 这种差别很重要 ✓）
"""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import ckpt as CK                                          # noqa: E402


def _draw():
    """从四个随机源各取一个数 ✓（复现性的直接观测量 ✓）。"""
    return (random.random(),
            float(np.random.rand()),
            float(torch.rand(1).item()))


# ─────────────────────── capture / restore 往返 ───────────────────────
def test_capture_has_all_sources():
    st = CK.capture_rng()
    assert "python" in st, "缺 Python random ✗"
    assert "numpy" in st, "缺 NumPy RNG ✗"
    assert "torch_cpu" in st, "缺 torch CPU RNG ✗"
    if torch.cuda.is_available():
        assert "torch_cuda" in st, "缺 CUDA RNG ✗"


def test_restore_makes_next_draw_identical():
    """★ P1-2 的核心性质 ✓: 捕获 -> 消耗 -> 恢复 -> **下一次抽样与原来一致** ✓✓"""
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    st = CK.capture_rng()
    a = _draw()                                     # 捕获后的第一次抽样 ✓
    _draw()                                         # 再消耗一些随机性 ✓
    _draw()
    assert CK.restore_rng(st) is True, "完整状态必须恢复成功 ✓"
    b = _draw()
    assert a == b, f"恢复后抽样不一致 ✗ {a} != {b}（P1-2 未生效 ✗）"


def test_restore_empty_returns_false():
    assert CK.restore_rng({}) is False, "空状态必须显式返回 False ✓（不得假装成功 ✗）"


def test_restore_partial_reports_incomplete():
    st = {"python": random.getstate()}              # 只有 python ✓ 其余缺失 ✗
    assert CK.restore_rng(st) is False, "不完整状态必须返回 False ✓"


# ─────────────────────── build_checkpoint 契约 ───────────────────────
class _Args:
    epoch_steps = 5000
    ckpt_every = 5000


class _ArgsMisaligned:
    epoch_steps = 5000
    ckpt_every = 2500


def _mk_model_opt():
    m = torch.nn.Linear(4, 3)
    o = torch.optim.AdamW(m.parameters(), lr=1e-3)
    return m, o


def test_build_checkpoint_contains_rng_and_meta():
    m, o = _mk_model_opt()
    ck = CK.build_checkpoint(m, o, None, None, _Args(), 5000)
    assert "rng" in ck, "ckpt 必须含 rng ✓（P1-2）"
    assert {"python", "numpy", "torch_cpu"} <= set(ck["rng"]), ck["rng"].keys()
    meta = ck.get("rng_meta") or {}
    assert meta.get("resume_kind") == "exact", meta
    assert meta.get("epoch_aligned") is True, meta
    assert meta.get("epoch_steps") == 5000 and meta.get("ckpt_every") == 5000


def test_build_checkpoint_flags_misaligned_phase():
    """★ epoch_steps != ckpt_every ⇒ 必须**显式标记相位不对齐** ✗（不得沉默 ✓）"""
    m, o = _mk_model_opt()
    ck = CK.build_checkpoint(m, o, None, None, _ArgsMisaligned(), 2500)
    assert (ck.get("rng_meta") or {}).get("epoch_aligned") is False, ck.get("rng_meta")


def test_build_checkpoint_tolerates_missing_epoch_steps():
    """args 没有 epoch_steps/ckpt_every 时不得抛错 ✓（旧配置兼容 ✓）"""
    m, o = _mk_model_opt()

    class _Bare:
        pass
    ck = CK.build_checkpoint(m, o, None, None, _Bare(), 10)
    meta = ck.get("rng_meta") or {}
    assert meta.get("epoch_steps") is None and meta.get("epoch_aligned") is False


def test_checkpoint_roundtrip_through_torch_save(tmp_path):
    """真实落盘往返 ✓: save -> load -> restore_rng ⇒ 抽样一致 ✓"""
    m, o = _mk_model_opt()
    torch.manual_seed(7)
    random.seed(7)
    ck = CK.build_checkpoint(m, o, None, None, _Args(), 42)
    p = os.path.join(str(tmp_path), "x.pt")
    torch.save(ck, p)
    a = _draw()
    _draw()
    back = torch.load(p, map_location="cpu", weights_only=False)
    assert CK.restore_rng(back["rng"]) is True
    assert _draw() == a, "落盘往返后抽样必须一致 ✓（P1-2 端到端 ✓）"
