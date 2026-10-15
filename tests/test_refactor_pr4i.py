# -*- coding: utf-8 -*-
"""test_refactor_pr4i.py —— 2026-10-14 [重构 PR4] **窗口均值汇总** 的验收测试 ✓

审查 §4.2: `Trainer.run()` 只管 "…if should_log(step): **log_metrics(metrics)**" ✓
  ⇒ 指标汇总是**纯计算** ✓ 应与日志输出、状态重置分离 ✓

本组测试锁定:
  ① ★ `average_window` 与旧内联算法**逐位一致** ✓（含 all_reduce 路径 ✓）
  ② `divisor` 兜底（`max(log_steps,1)` ✓ 防"整窗被跳过"时除零 ✓）
  ③ 12ch 分组: image / canny / skel 的识别 ✓（无 canny/skel ⇒ 0 ✓）
  ④ `steps_per_sec` 公式 ✓ / 接线守卫 ✓
"""
from __future__ import annotations

import io
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import telemetry as TEL                                     # noqa: E402


def _running(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {k: (torch.rand((), generator=g, dtype=torch.float64) * 3.0)
            for k in ("loss", "diff", "repa", "std_mid", "skel")}


# ─────────────── ① 与旧内联算法逐位一致 ───────────────
def test_average_window_matches_inline_single_rank():
    for seed in range(15):
        r = _running(seed)
        d = 1 + seed % 7
        # 内联版（逐字复刻 ✓）
        ref = {k: (r[k] / d).item() for k in ("loss", "diff", "repa", "std_mid")}
        ref["skel"] = (r["skel"] / d).item()
        w = TEL.average_window(r, d, 1)
        for k in ref:
            assert abs(getattr(w, k) - ref[k]) < 1e-15, (seed, k, getattr(w, k), ref[k])


def test_average_window_all_reduce_path():
    r = _running(1)
    seen = []

    def _ar(v, op=None):
        seen.append(("ar", op))
        v.mul_(2.0)                          # 模拟"另一个 rank 的贡献" ✓

    w = TEL.average_window(r, 2, 2, all_reduce=_ar)
    assert len(seen) == 4, "四个标量各归约一次 ✓（与旧内联一致 ✓）"
    # 归约后除以 world_size ✓
    expect = (r["loss"] / 2).mul(2.0).item() / 2
    assert abs(w.loss - expect) < 1e-15, (w.loss, expect)


def test_divisor_guard():
    """★ `divisor` 兜底 ✓: 传 0 ⇒ 按 1 处理（旧代码由调用方 `max(log_steps,1)` 保证 ✓）"""
    r = _running(2)
    w0 = TEL.average_window(r, 0, 1)
    w1 = TEL.average_window(r, 1, 1)
    assert w0.loss == w1.loss, "divisor=0 必须按 1 处理 ✓（不得除零 ✗）"


# ─────────────── ③ 12ch 分组 ───────────────
def test_c12_group_split():
    r = _running(3)
    acc = torch.arange(12, dtype=torch.float64)
    d = 2
    w = TEL.average_window(r, d, 1, acc_c12=acc,
                           aux_dirs=("aug_canny_v1", "aug_skel_v1"))
    cm = acc / d
    assert abs(w.c12_img - float(cm[:4].mean())) < 1e-15
    assert abs(w.c12_canny - float(cm[4:8].mean())) < 1e-15, "canny 组必须被识别 ✓"
    assert abs(w.c12_skel - float(cm[8:12].mean())) < 1e-15, "skel 组必须被识别 ✓"


def test_c12_without_matching_names_stays_zero():
    r = _running(4)
    w = TEL.average_window(r, 1, 1, acc_c12=torch.ones(12, dtype=torch.float64),
                           aux_dirs=("aug_foo", "aug_bar"))
    assert w.c12_canny == 0.0 and w.c12_skel == 0.0, \
        "名字里无 canny/skel ⇒ 两组保持 0 ✓（与旧内联一致 ✓）"
    assert w.c12_img == 1.0, "image 组始终有 ✓"


def test_c12_none_gives_zeros():
    r = _running(5)
    w = TEL.average_window(r, 1, 1, acc_c12=None)
    assert (w.c12_img, w.c12_canny, w.c12_skel) == (0.0, 0.0, 0.0)


# ─────────────── ④ steps_per_sec 与接线 ───────────────
def test_steps_per_sec_formula():
    r = _running(6)
    w = TEL.average_window(r, 1, 1, steps=10, seconds=2.0)
    assert abs(w.steps_per_sec - 5.0) < 1e-12
    w2 = TEL.average_window(r, 1, 1, steps=0, seconds=0.0)
    assert w2.steps_per_sec == 0.0, "0 步 ⇒ 0 ✓（不得除零 ✗）"


def test_train_py_uses_average_window():
    t = io.open("src/train/train.py", encoding="utf-8").read()
    assert "_wa = _tel.average_window(" in t, "未接线 ✗"
    assert "avg_l, avg_d, avg_r = _wa.loss, _wa.diff, _wa.repa" in t, "下游变量名必须保持 ✓"
    assert "steps_per_sec = _wa.steps_per_sec" in t
    # 旧内联的归约代码必须已消失 ✓
    assert "dist.all_reduce(avg_l" not in t, "旧内联 all_reduce 仍在 ✗"
    assert "avg_c12i = avg_c12c = avg_c12s = 0.0" not in t, "旧内联分组仍在 ✗"
    # divisor 仍留在调用方 ✓（它是窗口状态 ✓）
    assert "divisor = max(log_steps, 1)" in t, "divisor 应仍由调用方计算 ✓"


def test_module_is_still_pure_computation():
    """★ 边界 ✓: 本模块不打印、不改状态 ✓（只返回数据 ✓）"""
    import ast
    src = io.open("src/train/telemetry.py", encoding="utf-8").read()
    tree = ast.parse(src)
    # 只看 average_window 的函数体 ✓
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "average_window")
    body = ast.unparse(ast.Module(body=fn.body, type_ignores=[]))
    for bad in ("logger", "print", "self.", "backward", "step()"):
        assert bad not in body, f"average_window 不得涉及 `{bad}` ✗（纯计算 ✓）"
