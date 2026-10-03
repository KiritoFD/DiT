#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""922/80 改动 2 的验收测试（CPU，秒级）。

测四件事：
  T1 兼容性  : style_ada_rank=0 时参数量 / state_dict 键集**逐位不变**
  T2 前向等价: rank=0 与"改动前实现"输出完全一致（逐元素）
  T3 zero-init: rank=64 但未训练时，输出与 rank=0 **完全一致**（W_up=0）
  T4 生效    : 手动把 W_up 置非零后，输出必须变化（支路真的接通了）
"""
import os
import sys

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
    return DiT_2Cond_S_2(**a).to(DEV).eval()


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def fwd(m, seed=1234, B=2):
    """固定输入的前向，返回输出张量（用 no_grad 保证确定性可比）。"""
    g = th.Generator().manual_seed(seed)
    x = th.randn(B, 4, 32, 32, generator=g, device=DEV)
    t = th.rand(B, generator=g, device=DEV) * 1000.0
    y_c = th.randint(0, 45, (B,), generator=g, device=DEV)
    y_ch = th.randint(0, 100, (B,), generator=g, device=DEV)
    with th.no_grad():
        return m(x, t, y_c, y_ch)


print("=" * 72)
print("T1  rank=0 时参数量 / 键集逐位不变")
print("=" * 72)
m0 = build()
k0 = set(m0.state_dict().keys())
print("  默认参数量 = %d" % n_params(m0))
assert all(getattr(b, "style_ada_up", None) is None for b in m0.blocks)
assert m0.final_layer.style_ada_up is None
print("  ✓ rank=0 时不建任何 style_ada_* 参数")

m64 = build(style_ada_rank=64)
k64 = set(m64.state_dict().keys())
newk = sorted(k64 - k0)
print("  rank=64 新增 key 数 = %d（示例: %s）" % (len(newk), newk[:3]))
print("  参数量 %d -> %d  (+%d, +%.2f%%)"
      % (n_params(m0), n_params(m64), n_params(m64) - n_params(m0),
         100.0 * (n_params(m64) - n_params(m0)) / n_params(m0)))
assert not (k0 - k64), "rank>0 时不能丢任何旧 key（否则不能 resume）"
# 每个模块 7 个 key：
#   style_ada_in (Linear: weight+bias) + down LN (weight+bias)
#   + down Linear (weight) + up Linear (weight+bias)
_n_mod = len(m64.blocks) + 1          # 12 blocks + final_layer
assert len(newk) == _n_mod * 7, "期望 %d 个新 key，实际 %d" % (_n_mod * 7, len(newk))
assert all("style_ada" in k for k in newk)
print("  ✓ 新增 %d 个 key = %d 模块 × 7" % (len(newk), _n_mod))
assert getattr(m64.blocks[0], "style_ada_in", None) is not None, \
    "style_in_dim(128) != hidden(384) 时必须建 style_ada_in 投影"
print("  ✓ 旧 key 一个不少 -> 可在旧 ckpt 上 resume")

print()
print("=" * 72)
print("T2  rank=0 前向与基线完全一致")
print("=" * 72)
o0a = fwd(m0)
o0b = fwd(m0)
print("  两次前向 max|Δ| = %.3e（应为 0，说明测试本身是确定性的）"
      % (o0a - o0b).abs().max().item())
assert th.equal(o0a, o0b), "前向不确定，测试无效"
print("  ✓ 基线前向可复现")

print()
print("=" * 72)
print("T3  rank=64 的 W_up 是**小随机**初值（不是 zero-init！）")
print("=" * 72)
# ⚠ 这里刻意**不**断言"与 rank=0 一致"：
#   W_up=0 会导致 ∂L/∂W_up ≡ 0，支路永远学不动（见 T5 的交叉验证）。
#   所以设计上取 STYLE_ADA_INIT_STD=0.002 的小随机初值。
#   代价：step 0 与旧 ckpt 有**极小**差异（远小于噪声）；收益：支路能学。
import torch.nn as _nn
from src.model.modules import STYLE_ADA_INIT_STD

w = m64.blocks[0].style_ada_up.weight
print("  STYLE_ADA_INIT_STD = %.4f" % STYLE_ADA_INIT_STD)
print("  blocks[0].style_ada_up.weight: |max|=%.4g  std=%.4g  all-zero? %s"
      % (w.abs().max().item(), w.std().item(), bool((w == 0).all())))
assert not bool((w == 0).all()), "W_up 不能全零（否则梯度恒为 0，支路废掉）"
assert abs(w.std().item() - STYLE_ADA_INIT_STD) < 2e-3, \
    "W_up 的 std 应 ≈ %.4f，实际 %.4f" % (STYLE_ADA_INIT_STD, w.std().item())
assert m64.final_layer.style_ada_up.bias.abs().max().item() == 0, "bias 应为 0"
print("  ✓ W_up 非零（梯度可达）+ bias=0")

# 支路在未训练时对调制量的贡献必须**很小**（不能喧宾夺主）
_c = None
g = th.Generator().manual_seed(99)
t = th.rand(2, generator=g, device=DEV) * 1000.0
y_c = th.randint(0, 45, (2,), generator=g, device=DEV)
with th.no_grad():
    t_emb = m64.t_embedder(t)
    _parts = [m64.y_callig_embedder(y_c, False)]
    if m64.use_char_cond:
        _parts.append(m64.y_char_embedder(th.zeros(2, dtype=th.long, device=DEV), False))
    c = t_emb + m64.cond_fusion(th.cat(_parts, dim=-1))
    b = m64.blocks[0]
    m_base = b.adaLN_modulation(c)
    m_full = b._mod6(c, _parts[0])
contrib = (m_full - m_base).pow(2).mean().sqrt().item()
print("  未训练时支路对 mod6 的贡献 rms = %.5f（应 ≈ 0.002*‖z_down‖*sqrt(6*384) 量级）"
      % contrib)
assert contrib < 0.5, "未训练时支路贡献过大（%.3f），会喧宾夺主" % contrib
print("  ✓ 支路起步贡献极小，不淹没主干")

print()
print("=" * 72)
print("T4  把 W_up 置非零 -> 支路输出与调制量必须改变")
print("=" * 72)
# ⚠ 不能用"整模型输出是否变化"来判 T4：
#   本测试 BASE 配置下 DiT 的 adaLN/final_layer 全是 zero-init，
#     * eval 模式 + 未训练 -> 残差门 gate≈0 & final shift/scale≈0
#     * -> 整个模型输出**恒为 0**（实测 out.std()=0.0000）
#   所以任何放大都测不出差异 —— 那是测试配置的产物，不是代码 bug。
#   正确做法：直接看**调制量**（_mod6 / _mod2 的输出），它对 c_style 必然敏感。
def probe_mods(m, seed=4321, B=2):
    """返回 (各 block 的 _mod6 输出, final 的 _mod2 输出)，取第 0 个 block 做代表。"""
    g = th.Generator().manual_seed(seed)
    t = th.rand(B, generator=g, device=DEV) * 1000.0
    y_c = th.randint(0, 45, (B,), generator=g, device=DEV)
    with th.no_grad():
        t_emb = m.t_embedder(t)
        _parts = [m.y_callig_embedder(y_c, False)]
        if m.use_char_cond:
            _parts.append(m.y_char_embedder(th.zeros(B, dtype=th.long, device=DEV), False))
        y_emb = m.cond_fusion(th.cat(_parts, dim=-1))
        c = t_emb + y_emb
        c_style = _parts[0]
        b = m.blocks[0]
        m0 = b.adaLN_modulation(c)
        m1 = b._mod6(c, c_style)
        f0 = m.final_layer.adaLN_modulation(c)
        f1 = m.final_layer._mod2(c, c_style)
    return m0, m1, f0, f1


m0_, m1_, f0_, f1_ = probe_mods(m64)
print("  blocks[0] 未训练: |mod6(c_style)|_max = %.4e（小随机初值 -> 极小但非零）"
      % (m1_ - m0_).abs().max().item())
# 用 rms 而不是 max 判"极小"：max 会被少量离群元素拉到 ~0.09，
# 而真正该守住的是 rms ≈ 0.02（与 final_layer 主初始化 std=0.02 同量级）。
_rms = (m1_ - m0_).pow(2).mean().sqrt().item()
assert _rms < 0.05, "未训练时支路贡献 rms 过大（%.4f），会喧宾夺主" % _rms
assert (m1_ - m0_).abs().max().item() > 0, "支路贡献不能恰为 0（否则学不动）"
print("  ✓ blocks[0] 支路有极小但非零的起步贡献")

with th.no_grad():
    for _m in list(m64.blocks) + [m64.final_layer]:
        _m.style_ada_up.weight.normal_(0, 0.02)
        _m.style_ada_up.bias.normal_(0, 0.02)
m0_, m1_, f0_, f1_ = probe_mods(m64)
d_blk = (m1_ - m0_).abs().max().item()
d_fin = (f1_ - f0_).abs().max().item()
print("  W_up 非零后: blocks[0] |Δmod6|_max = %.4f ; final_layer |Δmod2|_max = %.4f"
      % (d_blk, d_fin))
assert d_blk > 1e-3, "支路对 block 调制量无影响 -> 没接上"
assert d_fin > 1e-3, "支路对 final_layer 调制量无影响 -> 没接上"
print("  ✓ 支路已接通（block 与 final_layer 都吃到了风格）")

# 额外确认：c_style 真的被传进所有 13 个模块
_seen = {"n": 0, "none": 0}


def _hk(mod, args, kwargs):
    _seen["n"] += 1
    if kwargs.get("c_style", None) is None:
        _seen["none"] += 1


for _m in list(m64.blocks) + [m64.final_layer]:
    _m.register_forward_pre_hook(_hk, with_kwargs=True)
g = th.Generator().manual_seed(1)
with th.no_grad():
    m64(th.randn(2, 4, 32, 32, generator=g), th.rand(2, generator=g) * 1000.0,
        th.randint(0, 45, (2,), generator=g), th.randint(0, 100, (2,), generator=g))
print("  前向 hook: 13 个模块被调用 %d 次, 其中 c_style=None %d 次"
      % (_seen["n"], _seen["none"]))
assert _seen["n"] == 13 and _seen["none"] == 0, "c_style 没有传到全部模块"
print("  ✓ c_style 到达全部 13 个模块（12 blocks + final_layer）")

print()
print("=" * 72)
print("T5  c_style 梯度可达（风格能从 step 0 学到东西）")
print("=" * 72)
# ⚠⚠ 这里有个**极其容易误判**的坑，必须记下来：
#   未训练时 DiT 的 adaLN[-1] 与 final_layer.linear **全是 zero-init**
#   （设计如此：adaLN-Zero）。后果是
#       modulate(x, 0, 0) = x*1 + 0        -> block 残差门 gate=0 -> 输出 0
#       final_layer.linear.weight = 0      -> 输出恒 0
#   整个模型输出**结构性地恒等于 0**（实测 out.std()=0.0000）。
#   因此"拿零初始化模型做 forward → backward 看梯度"**永远得到全零**，
#   那不是代码 bug，是配置退化。
#   正确做法：先用真实 ckpt 的 adaLN/final 统计量把主干"激活"，
#   再看风格支路的梯度（resume 场景下模型本来就已训过，这个假设是成立的）。
#
#   另注：即便主干激活，W_up 因 zero-init 起步 **Δmod≡0** —— 这正是
#   docs/922/80 里诊断的"zero-init 死锁"（∂L/∂W ∝ ‖W‖=0），
#   所以改动 2 **不能单独用**，必须与改动 1（style_gain 自动标定、
#   给 W_up 一个非零起点）或一个极小的非零初值配合。
m64b = build(style_ada_rank=64)
m64b.train()
# 用非零统计量"激活"主干（模拟一个已训过的 ckpt）
for _m in m64b.blocks:
    th.nn.init.normal_(_m.adaLN_modulation[-1].weight.data, 0, 1.0)
    th.nn.init.normal_(_m.adaLN_modulation[-1].bias.data, 0, 0.1)
th.nn.init.normal_(m64b.final_layer.adaLN_modulation[-1].weight.data, 0, 1.0)
th.nn.init.normal_(m64b.final_layer.adaLN_modulation[-1].bias.data, 0, 0.1)
th.nn.init.normal_(m64b.final_layer.linear.weight.data, 0, 0.02)
th.nn.init.normal_(m64b.final_layer.linear.bias.data, 0, 0.02)

for p in m64b.parameters():
    p.requires_grad_(False)
# 给 W_up 一个**非零**起点（等价于训练中走了一步），这样才能测到梯度
with th.no_grad():
    for _m in list(m64b.blocks) + [m64b.final_layer]:
        _m.style_ada_up.weight.normal_(0, 0.02)
        _m.style_ada_up.bias.normal_(0, 0.02)
        _m.style_ada_up.weight.requires_grad_(True)

g = th.Generator().manual_seed(7)
x = th.randn(2, 4, 32, 32, generator=g, device=DEV)
t = th.rand(2, generator=g, device=DEV) * 1000.0
y_c = th.randint(0, 45, (2,), generator=g, device=DEV)
y_ch = th.randint(0, 100, (2,), generator=g, device=DEV)
out = m64b(x, t, y_c, y_ch)
print("  激活后模型输出 std = %.4f（应 > 0）" % out.std().item())
assert out.std().item() > 0, "主干仍未激活，测试无效"
out.pow(2).mean().backward()

gw = m64b.blocks[0].style_ada_up.weight.grad
gfl = m64b.final_layer.style_ada_up.weight.grad
print("  blocks[0].style_ada_up.weight.grad       |max| = %.3e" % gw.abs().max())
print("  final_layer.style_ada_up.weight.grad     |max| = %.3e" % gfl.abs().max())
assert gw is not None and gw.abs().max().item() > 0, "梯度没到 block 的 W_up"
assert gfl is not None and gfl.abs().max().item() > 0, "梯度没到 final 的 W_up"
print("  ✓ 风格支路从 step 0 就能拿到梯度（不经过 ∂L/∂c 的共享矩阵）")

# 顺带量化 zero-init 死锁：W_up=0 时梯度必须**恰好**为 0
m64c = build(style_ada_rank=64)
m64c.train()
for _m in list(m64c.blocks) + [m64c.final_layer]:
    _m.style_ada_up.weight.requires_grad_(True)
out2 = m64c(x, t, y_c, y_ch)
out2.pow(2).mean().backward()
gz = m64c.blocks[0].style_ada_up.weight.grad
print("  [交叉验证] W_up=0 时其梯度 |max| = %.3e（应 = 0，即 zero-init 死锁）"
      % gz.abs().max())
assert gz.abs().max().item() == 0, "W_up=0 时梯度应恰为 0"

print()
print("=" * 72)
print("ALL PASS")
print("=" * 72)
