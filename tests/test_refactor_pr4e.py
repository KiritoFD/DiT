# -*- coding: utf-8 -*-
"""test_refactor_pr4e.py —— 2026-10-14 [重构 PR4] **CheckpointManager** 的验收测试 ✓

审查 §4.2: "`Trainer` 控制循环，`TrainStep` 完成一次更新，`ObjectiveComposer` 组合损失，
  **`CheckpointManager` 负责持久化**，各自保持小而清晰" ✓
审查 P1-3a: "对正式训练使用**严格模式**：最后一次 checkpoint 未成功写入或校验失败，
  进程必须**非零退出**" ✓

本组测试锁定:
  ① `save` 与旧的两行调用（`save_checkpoint` + `prune_checkpoints`）**参数与顺序完全一致** ✓
  ② `finalize` ≡ `drain_ckpt` ✓ 且**默认严格** ✓（strict=None ⇒ 走默认 ✓）
  ③ `should_save` ≡ 旧条件 ✓（`>0` 那条留在调用方 ✓）
  ④ 边界: 只碰持久化 ✓（不碰 optimizer / 评测 ✓）
"""
from __future__ import annotations

import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import ckpt as CK                                           # noqa: E402
from src.train.checkpoint import CheckpointManager                         # noqa: E402


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



# ─────────────── ① save ≡ 旧两行（参数与顺序 ✓）───────────────
def test_save_calls_same_functions_with_same_args(monkeypatch, tmp_path):
    """★★ 核心 ✓: 必须**先** save_checkpoint（同参数同顺序 ✓）**再** prune_checkpoints ✓"""
    calls = []

    def _fake_save(model, opt, ema_model, scheduler, args, step, cdir, logger):
        calls.append(("save", model, opt, ema_model, scheduler, args, step, cdir, logger))
        return os.path.join(cdir, "%07d.pt" % step)

    def _fake_prune(cdir, keep, logger):
        calls.append(("prune", cdir, keep, logger))
        return 0

    monkeypatch.setattr(CK, "save_checkpoint", _fake_save)
    monkeypatch.setattr(CK, "prune_checkpoints", _fake_prune)

    m = object()
    o = object()
    e = object()
    sch = object()
    a = object()
    log = object()
    mgr = CheckpointManager(str(tmp_path), keep=3, logger=log)
    path = mgr.save(m, o, e, sch, a, 5000)

    assert [c[0] for c in calls] == ["save", "prune"], calls
    assert calls[0][1:] == (m, o, e, sch, a, 5000, str(tmp_path), log), \
        "save_checkpoint 的参数与顺序必须与旧调用**逐字一致** ✗"
    assert calls[1][1:] == (str(tmp_path), 3, log), \
        "prune_checkpoints 的参数必须与旧调用一致 ✗"
    assert path.endswith("0005000.pt")
    assert mgr.last_saved == 5000 and mgr.n_saved == 1


def test_keep_defaults_to_zero(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(CK, "save_checkpoint",
                        lambda *a, **k: "p")
    monkeypatch.setattr(CK, "prune_checkpoints",
                        lambda cdir, keep, logger: seen.update(keep=keep))
    CheckpointManager(str(tmp_path), keep=0).save(object(), object(), None, None, None, 10)
    assert seen["keep"] == 0


def test_disabled_manager_is_noop(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(CK, "save_checkpoint", lambda *a, **k: called.append(1))
    mgr = CheckpointManager(str(tmp_path), enabled=False)
    assert mgr.save(object(), object(), None, None, None, 1) is None
    assert not called, "enabled=False 时不得存盘 ✗"


# ─────────────── ② finalize ≡ drain_ckpt（默认严格 ✓）───────────────
def test_finalize_delegates_with_default_strict(monkeypatch):
    seen = {}

    def _fake_drain(logger=None, timeout=600, strict=None):
        seen.update(logger=logger, timeout=timeout, strict=strict)
        return 7, []

    monkeypatch.setattr(CK, "drain_ckpt", _fake_drain)
    log = object()
    n, errs = CheckpointManager("/tmp/x", logger=log).finalize(timeout=600)
    assert (n, errs) == (7, [])
    assert seen["logger"] is log and seen["timeout"] == 600
    assert seen["strict"] is None, \
        "默认必须传 strict=None ✓ ⇒ 走 ckpt.py 的**严格**默认 ✓（P1-3a ✓）"


def test_finalize_propagates_explicit_strict(monkeypatch):
    seen = {}
    monkeypatch.setattr(CK, "drain_ckpt",
                        lambda logger=None, timeout=600, strict=None: seen.update(s=strict) or (0, []))
    CheckpointManager("/tmp/x", strict=False).finalize()
    assert seen["s"] is False, "显式 strict=False 必须透传 ✓"


def test_finalize_raises_in_strict_mode(monkeypatch):
    """★ P1-3a 的语义必须仍然成立 ✓: 有失败 ⇒ 严格模式**抛错** ✗"""
    monkeypatch.setattr(CK, "WRITER", type("W", (), {"drain": lambda self, timeout=None: (0, [("p", "boom")])})())
    monkeypatch.delenv("DSH_CKPT_STRICT", raising=False)
    with pytest.raises(RuntimeError, match="写盘或校验失败"):
        CheckpointManager("/tmp/x").finalize(timeout=1)


# ─────────────── ③ should_save ≡ 旧条件 ───────────────
@pytest.mark.parametrize("every,step,expect", [
    (5000, 0, True), (5000, 5000, True), (5000, 4999, False),
    (0, 100, False), (-1, 100, False), (1, 7, True),
])
def test_should_save_matches_old_condition(every, step, expect):
    mgr = CheckpointManager("/tmp/x")
    old = (every > 0 and step % every == 0)
    assert mgr.should_save(step, every) is bool(old) is expect


def test_step_gt_zero_stays_in_caller():
    """★ `step > 0` 属**训练语义** ✓ 仍留在调用方 ✓（本模块只管"刻度对齐" ✓）"""
    t = _read_train_src()
    assert "if _save_ckpt and train_steps > 0:" in t, \
        "`train_steps > 0` 必须仍留在调用方 ✓（不得挪进 manager ✗）"


# ─────────────── ④ 边界（原则 B ✓）───────────────
def test_module_has_no_optimizer_or_eval_side_effects():
    src = io.open("src/train/checkpoint.py", encoding="utf-8").read()
    for bad in ("opt.step", "zero_grad", "scheduler.step", "in_mem_eval",
                "maybe_run_in_training", "loss.backward"):
        assert bad not in src, f"checkpoint.py 不得涉及 `{bad}` ✗（原则 B ✓）"


def test_train_py_uses_manager_and_dropped_old_calls():
    """★ 接线守卫 ✓: 旧的三处直接调用必须已消失 ✗ 且改走 manager ✓"""
    t = _read_train_src()
    assert "_ckpt.save(model, opt, ema_model, scheduler, args, train_steps)" in t, "未接线 ✗"
    assert "_ckpt.finalize(timeout=600)" in t, "收尾未接线 ✗"
    assert "[ckpt]" in t, "启动日志必须打印存盘策略身份 ✓"
    import re
    for name in ("save_checkpoint", "prune_checkpoints", "drain_ckpt"):
        assert not re.search(r"^\s+%s\(" % name, t, re.M), \
            f"train.py 仍有直接调用 {name}( ✗（应走 CheckpointManager ✓）"


def test_describe_reports_identity():
    d = CheckpointManager("/a/b/run_42", keep=2).describe()
    for token in ("ckpt_dir=run_42", "keep=2", "strict=default(严格)", "enabled=True"):
        assert token in d, (token, d)
