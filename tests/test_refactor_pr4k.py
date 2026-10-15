# -*- coding: utf-8 -*-
"""test_refactor_pr4k.py —— 2026-10-14 [重构 PR4] **单步日志格式化** 的验收测试 ✓

审查 §4.2: `Trainer.run()` 只管 "…if should_log(step): **log_metrics(metrics)**" ✓
  ⇒ 日志**文本构造**是纯格式化 ✓ 应与训练状态分离 ✓

本组测试锁定:
  ① ★ 与**旧内联 f-string**逐字一致 ✓（含各段条件 ✓）
  ② ★★ 搬移的经典陷阱 ✗: 旧代码读 `globals().get('_DEFORM_EMA')` ✗
     ⇒ 函数化后必须**显式传参** ✓ 否则恒为 0 ✗✗（用"形变值真的出现"来证明 ✓）
  ③ 段落的**开关条件**逐项 ✓（`w>0` / `None` ✓）
  ④ 纯格式化 ✓（可注入 `mem` ⇒ 不依赖 CUDA ✓）
"""
from __future__ import annotations

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train.telemetry import format_step_log as F                        # noqa: E402

BASE = dict(step=5000, avg_d=0.2750, avg_r=0.0037, avg_std_mid=0.01,
            avg_skel=0.02, steps_per_sec=3.50, w_repa=0.03, mem=(22.56, 22.60))


def _old_format(*, step, avg_d, avg_r, avg_std_mid, avg_skel, steps_per_sec, w_repa,
                w_std_mid=0.0, w_deform_skel=0.0, deform_loss=0.0, deform_off=0.0,
                w_latent_skel=0.0, vae_kl=None, w_vae_kl=1e-5, train_vae_enc=False,
                vae_lr=None, lr=0.0, ema_decay=None, mem=(22.56, 22.60)):
    """**逐字复刻**旧内联 f-string ✓（把 globals 读改成显式参数 ✓ 其余一模一样 ✓）。"""
    ema_log = (f"EMA: {ema_decay:.6f} | " if ema_decay is not None else "")
    return (
        f"(step={step:07d}) Diff: {avg_d:.4f} | "
        f"REPA(w={w_repa:.2f}): {avg_r:.4f} | "
        + (f"StdMid(w={w_std_mid:.3f}): {avg_std_mid:.4f} | " if w_std_mid > 0 else "")
        + (f"Deform(w={w_deform_skel:.2f}): {deform_loss:.4f} off={deform_off:.4f} | "
           if w_deform_skel > 0 else "")
        + (f"SkelStruct(w={w_latent_skel:.3f}): {avg_skel:.4f} | "
           if w_latent_skel > 0 else "")
        + (f"VAE-KL(w={w_vae_kl:.1e}): {float(vae_kl):.4f} | "
           if train_vae_enc and vae_kl is not None else "")
        + (f"VAE-LR: {vae_lr:.2e} | " if train_vae_enc and vae_lr is not None else "")
        + f"LR: {lr:.2e} | {ema_log}"
        f"Steps/Sec: {steps_per_sec:.2f} | "
        f"Mem: {mem[0]:.2f}G/{mem[1]:.2f}G"
    )


# ─────────────── ① 逐字一致 ───────────────
def test_matches_old_inline_format_exactly():
    cases = [
        dict(),
        dict(w_std_mid=0.05),
        dict(w_deform_skel=0.1, deform_loss=0.1234, deform_off=0.5678),
        dict(w_latent_skel=0.02),
        dict(train_vae_enc=True, vae_kl=0.5, vae_lr=2e-6),
        dict(train_vae_enc=True, vae_kl=None, vae_lr=2e-6),
        dict(lr=4.16e-05, ema_decay=0.999),
        dict(w_std_mid=0.05, w_deform_skel=0.1, deform_loss=0.1, deform_off=0.2,
             w_latent_skel=0.02, train_vae_enc=True, vae_kl=0.5, vae_lr=2e-6,
             lr=4.16e-05, ema_decay=0.999),
    ]
    for extra in cases:
        kw = dict(BASE)
        kw.update(extra)
        assert F(**kw) == _old_format(**kw), (extra, F(**kw), _old_format(**kw))


# ─────────────── ② ★★ 形变值必须真的被传进来（陷阱 ✓）───────────────
def test_deform_values_come_from_parameters_not_globals():
    """★★ 旧代码读 `globals().get('_DEFORM_EMA')` ✗ ⇒ 函数化后若仍读 globals
    会读到**本模块**的全局 ⇒ 恒为空 ✗✗ ⇒ 形变段会恒显示 0.0000 ✗
    ⇒ 本测试要求传入的值**真的出现**在文本里 ✓
    """
    s = F(w_deform_skel=0.1, deform_loss=0.1234, deform_off=0.5678, **BASE)
    assert "Deform(w=0.10): 0.1234 off=0.5678" in s, s


def test_module_does_not_read_globals_for_deform():
    """★ 必须**剥掉三引号块**再查 ✗ —— 模块文档里**引用**了该全局名
    （作为"为何不能这么写"的说明 ✓）⇒ 朴素子串检查会误报 ✗（实测如此 ✓）

    ⚠ 用**正则**剥 ✓ —— 不要用 `ast.walk` 边遍历边改 `body` ✗
      （改动会让 walk 漏掉节点 ✗ 实测如此 ✓）
    """
    import re as _re
    src = io.open("src/train/telemetry.py", encoding="utf-8").read()
    code = _re.sub(r'""".*?"""', '""', src, flags=_re.S)
    code = _re.sub(r"'''.*?'''", "''", code, flags=_re.S)
    assert "_DEFORM_EMA" not in code, \
        "telemetry 不得读 `_DEFORM_EMA` 全局 ✗（必须显式传参 ✓ 否则恒为 0 ✗✗）"


def test_train_py_still_owns_the_global_read():
    """★ 反过来: `_DEFORM_EMA` 的读取**必须留在** train.py ✓（全局在那里 ✓）"""
    t = io.open("src/train/train.py", encoding="utf-8").read()
    assert "globals().get('_DEFORM_EMA')" in t, \
        "train.py 必须继续自己读该全局 ✓（然后只传值进去 ✓）"
    assert "_tel.format_step_log(" in t, "未接线 ✗"


# ─────────────── ③ 段开关条件 ───────────────
def test_segment_switches():
    base = F(**BASE)
    assert "StdMid" not in base and "Deform" not in base and "SkelStruct" not in base
    assert "VAE-KL" not in base and "VAE-LR" not in base and "EMA:" not in base
    assert "StdMid" in F(w_std_mid=1e-9, **BASE), "w>0（哪怕极小）必须出现 ✓"
    assert "StdMid" not in F(w_std_mid=0.0, **BASE)
    assert "EMA:" in F(ema_decay=0.999, **BASE)
    assert "VAE-LR" in F(train_vae_enc=True, vae_lr=2e-6, **BASE)
    assert "VAE-LR" not in F(train_vae_enc=True, vae_lr=None, **BASE)
    assert "VAE-KL" not in F(train_vae_enc=False, vae_kl=0.5, **BASE), \
        "未训 VAE Encoder 时不显示 KL ✓"
    assert "VAE-KL" not in F(train_vae_enc=True, vae_kl=None, **BASE)


def test_step_and_lr_formats():
    s = F(lr=4.16e-05, **BASE)
    assert "(step=0005000)" in s, "step 必须 7 位补零 ✓"
    assert "LR: 4.16e-05" in s
    assert "Steps/Sec: 3.50" in s and "Mem: 22.56G/22.60G" in s


def test_mem_can_be_injected_without_cuda():
    """★ 纯格式化 ✓: 注入 `mem` ⇒ 不查 CUDA ✓（便于 CPU 单测 ✓）"""
    kw = dict(BASE)
    kw["mem"] = (1.25, 9.99)          # ⚠ BASE 里已有 mem ✗ ⇒ 必须合并而非重复传参 ✗
    s = F(**kw)
    assert "Mem: 1.25G/9.99G" in s, s
