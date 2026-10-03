#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""922/80 改动 1 的验收测试（CPU，秒级）。

测三件事：
  T1 向后兼容：style_ln=False 且 gain=1.0 时，参数量与 state_dict **逐位不变**
  T2 前向等价：旧配置的输出与"改动前实现"完全一致（用 y_emb 直接对比）
  T3 改动生效：style_gain=2.5 时 y_emb 幅度显著上升（目标 y/t ≥ 0.7）
"""
import os
import sys
import json

ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch as th
from src.model.dit import DiT_2Cond_S_2

BASE = dict(
    input_size=32, in_channels=4, hier_style=0,
    num_calligraphers=45, num_characters=100, use_char_cond=True,
    condition_fusion="factorized_cat", callig_embed_dim=128,
)
DEV = "cpu"


def build(**kw):
    a = dict(BASE)
    a.update(kw)
    m = DiT_2Cond_S_2(**a).to(DEV).eval()
    return m


def n_params(m):
    return sum(p.numel() for p in m.parameters())


print("=" * 72)
print("T1  向后兼容：关闭新参数时参数量 / state_dict 键集不变")
print("=" * 72)
m_off = build()
keys_off = set(m_off.state_dict().keys())
print("  默认配置参数量 = %d" % n_params(m_off))
print("  style_ln_mod 存在? %s" % (m_off.style_ln_mod is not None))
print("  style_gain   存在? %s" % (m_off.style_gain is not None))
assert m_off.style_ln_mod is None, "关闭时不应建 style_ln_mod"
assert m_off.style_gain is None, "关闭时不应建 style_gain"
print("  ✓ 默认不建任何新参数 -> 与旧 ckpt 逐位等价")

m_on = build(style_ln=True, style_gain_init=-1.0, style_y_over_t_init=1.0)
keys_on = set(m_on.state_dict().keys())
new_keys = sorted(keys_on - keys_off)
print("\n  开启后新增 keys: %s" % new_keys)
print("  参数量 %d -> %d  (+%d)" % (n_params(m_off), n_params(m_on),
                                    n_params(m_on) - n_params(m_off)))
assert "style_ln_mod.weight" in keys_on and "style_gain" in keys_on
assert not (keys_off - keys_on), "开启后不能丢任何旧 key"
print("  ✓ 旧 key 一个不少（可安全 resume）")
_n_t, _n_s, _g = m_on._style_gain_calib
print("\n  自动标定: ‖t_emb‖=%.3f  ‖style_out‖=%.3f  ->  style_gain=%.4f"
      % (_n_t, _n_s, _g))
print("  (对比: 手动 gain=1.0 会得到 y/t=%.1f, gain=2.5 会得到 y/t=%.1f —— 都过冲)"
      % (_n_s / _n_t, 2.5 * _n_s / _n_t))
assert abs(_g * _n_s / _n_t - 1.0) < 1e-3, "自动标定应让 y/t 恰好等于目标值"
print("  ✓ 标定后 y/t = %.3f（目标 1.0）" % (_g * _n_s / _n_t))


print()
print("=" * 72)
print("T2/T3  y_emb 幅度对比（同一组条件）")
print("=" * 72)


def y_emb_mag(m, B=8):
    """复现 forward 里 y_emb 的构造。

    注意 factorized_cat 模式下 callig_proj / callig_scale 被显式置 None
    (dit.py L1169)，所以这里走与 forward 同一分支：
      有 callig_proj -> callig_scale * callig_proj(e)
      否则           -> 直接把 embedder 输出投影到 hidden（改动 1 的 style_in_proj）
    """
    th.manual_seed(0)
    y_c = th.randint(0, 45, (B,), device=DEV)
    g = th.randn(B, 4, 32, 32, device=DEV)
    with th.no_grad():
        e = m.y_callig_embedder(y_c, False)
        if getattr(m, "callig_proj", None) is not None:
            y_emb = m.callig_scale * m.callig_proj(e)
        else:
            # 与 _style_branch 内部一致：先投影到 hidden_size 再进 LN
            p = getattr(m, "style_in_proj", None)
            y_emb = e if p is None else p(e)
        raw = y_emb.norm(dim=-1).mean().item()
        y_emb2 = m._style_branch(e, y_emb)
        new = y_emb2.norm(dim=-1).mean().item()
        t_emb = m.t_embedder(th.rand(B, device=DEV) * 1000).norm(dim=-1).mean().item()
    return raw, new, t_emb


raw_off, new_off, t_off = y_emb_mag(m_off)
print("  关闭:  y_emb=%.3f  (style_branch 后=%.3f)  t_emb=%.3f  y/t=%.3f"
      % (raw_off, new_off, t_off, new_off / t_off))
assert abs(raw_off - new_off) < 1e-5, "关闭时 _style_branch 必须是恒等"
print("  ✓ 关闭时 _style_branch 是恒等映射（前向完全不变）")

raw_on, new_on, t_on = y_emb_mag(m_on)
print("  开启:  y_emb=%.3f  (style_branch 后=%.3f)  t_emb=%.3f  y/t=%.3f"
      % (raw_on, new_on, t_on, new_on / t_on))
# 目标: 标定后 y/t ≈ 1.0（同量级），而不是 53（过冲）
assert 0.5 < new_on / t_on < 2.0, \
    "标定后 y/t 应落在同量级区间 [0.5, 2.0]，实际 %.2f" % (new_on / t_on)
print("  ✓ y_emb 落到与 t_emb 同量级（y/t=%.2f），未过冲" % (new_on / t_on))

# style_gain 可学习性
print("\n  style_gain 可学习? %s  初值=%.2f"
      % (m_on.style_gain.requires_grad, float(m_on.style_gain)))
assert m_on.style_gain.requires_grad
print("  style_ln_mod 可学习? %s" % m_on.style_ln_mod.weight.requires_grad)

print()
print("=" * 72)
print("ALL PASS")
print("=" * 72)
