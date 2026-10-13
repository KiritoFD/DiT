# -*- coding: utf-8 -*-
"""v85A 风格支路的性质测试（方案 §二 的验收点 ✓）。

6 条**可判定**性质：
  1. gain=0 ⇒ 输出与"完全没有风格支路"**逐位相同** ✓（warm-start 等价 ✓）
  2. 只在指定层建参数（其余层与旧模型逐位一致 ✓）
  3. `style_ada_at=""` ⇒ 全部层 + final_layer（**旧行为不破** ✓）
  4. 输入维度不匹配 ⇒ **报错**（不再在 forward 里偷偷重建 ✓）
  5. 初始时 ``style_gain`` 能拿到**非零梯度**（否则支路永远学不动 ✓）
  6. ★ 风格支路**不受 cond_router 白名单限制**（原实现在非白名单层把它一并切断 ✗）
"""
import pytest
import torch

from src.model.dit import DiT_2Cond


def _tiny(**kw):
    """构造微型模型（照 tests/test_std_soft_glyph.py 的可用写法 ✓）。"""
    torch.manual_seed(0)
    kw.setdefault("use_glyph_cond", True)
    kw.setdefault("std_soft_cond", True)
    kw.setdefault("glyph_in_channels", 1)
    m = DiT_2Cond(
        input_size=32, patch_size=2, in_channels=4,
        hidden_size=48, depth=4, num_heads=4, mlp_ratio=2.0,
        num_calligraphers=3, num_characters=5, learn_sigma=False,
        condition_fusion="legacy", **kw)
    m.eval()
    return m


def _fwd(m, bs=2):
    x = torch.randn(bs, 4, 32, 32)
    t = torch.full((bs,), 500.0)
    yc = torch.randint(0, 3, (bs,))
    ych = torch.randint(0, 5, (bs,))
    g = torch.rand(bs, 1, 64, 64)
    return m(x, t, yc, ych, g=g)


def _style_params(m):
    return [(n, p) for n, p in m.named_parameters() if "style_ada" in n or "style_gain" in n]


def test_gain_zero_is_bit_identical_to_no_branch():
    """★ warm-start 等价: gain=0 时支路**严格不生效** ⇒ 与无支路逐位相同 ✓。"""
    m = _tiny(style_ada_rank=16, style_ada_at="", style_ada_gain=True)
    with torch.no_grad():
        o0 = _fwd(m)
        # 把 up 权重放大 50 倍 ⇒ 若 gate 真生效, 输出必变 ✓
        for n, p in _style_params(m):
            if n.endswith("style_ada_up.weight"):
                p.mul_(50.0)
        o1 = _fwd(m)
        # 关掉 gate（等价"没有支路"）⇒ 仍应逐位相同 ✓
        for mod in list(m.blocks) + [m.final_layer]:
            if getattr(mod, "style_ada_up", None) is not None:
                mod._style_ada_gain_on = False
        o2 = _fwd(m)
    assert torch.equal(o0, o1), "gain=0 时支路仍在影响输出 ✗"
    assert torch.equal(o0, o2), "gain=0 与无支路不等价 ✗"
    assert any(n.endswith("style_gain") for n, _ in _style_params(m)), "未建 style_gain ✗"


def test_only_selected_layers_get_params():
    m = _tiny(style_ada_rank=16, style_ada_at="1,3", style_ada_gain=True)
    have = [i for i, b in enumerate(m.blocks) if getattr(b, "style_ada_up", None) is not None]
    assert have == [1, 3], "分层选择不对: %s ✗" % have
    assert getattr(m.final_layer, "style_ada_up", None) is None, "final 不该建 ✗"


def test_empty_at_keeps_legacy_all_layers_and_final():
    """旧行为: 未给 style_ada_at ⇒ 全部层 + final_layer 都有支路 ✓（不破坏老 ckpt ✓）。"""
    m = _tiny(style_ada_rank=8, style_ada_at="")
    assert all(getattr(b, "style_ada_up", None) is not None for b in m.blocks)
    assert getattr(m.final_layer, "style_ada_up", None) is not None
    assert not any(n.endswith("style_gain") for n, _ in _style_params(m)), "默认不该建 gain ✗"


def test_gain_off_reproduces_legacy_formula():
    """gain 关闭 ⇒ 与旧公式 ``m + up(down(in(c_style)))`` 逐位一致 ✓。"""
    m = _tiny(style_ada_rank=8, style_ada_at="2", style_ada_gain=False)
    blk = m.blocks[2]
    c = torch.randn(2, 48)
    # 期望的风格维度要从**记录值**取 ✓（style_in_dim == hidden 时投影层是 Identity,
    # 它没有 in_features ✗ —— 我第一版就踩了这个 ✓）
    cs = torch.randn(2, int(getattr(blk, "_style_in_dim_expect", 48)))
    with torch.no_grad():
        got = blk._mod6(c, cs)
        ref = blk.adaLN_modulation(c) + blk.style_ada_up(
            blk.style_ada_down(blk._style_ada_apply_in(cs)))
    assert torch.equal(got, ref), "gain 关闭时公式与旧实现不一致 ✗"


def test_mismatched_style_dim_raises():
    """维度不匹配必须**报错**（不再在 forward 里就地重建 ✗ —— optimizer 管不到 ✗）。"""
    m = _tiny(style_ada_rank=8, style_ada_at="0", style_ada_gain=True)
    blk = m.blocks[0]
    with pytest.raises(RuntimeError, match="不匹配"):
        blk._mod6(torch.randn(2, 48), torch.randn(2, 999))


def test_style_gain_gradient_is_nonzero_at_init():
    """★ 初始时 gain 的**局部**梯度必须非零, 否则支路永远启动不了 ✓。

    ⚠ 为什么不能拿整模型 ``out.sum()`` 来测（我第一版就是 ✗）::
        新建模型的 ``adaLN_modulation[-1]`` 与 ``final_layer`` 都是**零初始化**（DiT 标准 ✓）
        ⇒ **输出恒为 0** ⇒ ``∂out/∂x ≡ 0`` ⇒ 一切都拿不到梯度 ✗（与风格支路无关 ✓）。
        v84A@200k warm-start 时输出头**已训好** ✓ ⇒ 真训练里不存在这个退化 ✓。
    ⇒ 正确做法: 直接测 ``_mod6`` + 喂**合成上游梯度** ✓ —— 检验的正是
        ``∂L/∂gain = (∂L/∂m)·Δ·(1-tanh²)`` 里"Δ ≠ 0"这一条本质条件 ✓。
    """
    m = _tiny(style_ada_rank=8, style_ada_at="1", style_ada_gain=True)
    blk = m.blocks[1]
    c = torch.randn(4, 48)
    cs = torch.randn(4, int(getattr(blk, "_style_in_dim_expect", 48)))
    m_out = blk._mod6(c, cs)
    m_out.backward(torch.ones_like(m_out))       # 上游梯度全 1 ⇒ ∂/∂gain = Σ Δ ✓
    g = blk.style_gain.grad
    assert g is not None and float(g.abs()) > 0, "gain 初始梯度为 0 ⇒ 支路会废掉 ✗"
    # 且 up.bias 必须非零: cond_drop 把 e_callig 置零时 Δ 仍非零 ✓
    assert float(blk.style_ada_up.bias.abs().max()) > 0, "up.bias 被清零 ⇒ Δ 可能恒为 0 ✗"


def test_style_branch_survives_cond_router_whitelist():
    """★★ 风格支路**不受** cond_router 白名单限制（原实现会把它一并切断 ✗）。"""
    m = _tiny(style_ada_rank=8, style_ada_at="3", style_ada_gain=False)

    class _Router:
        def allows(self, i):
            return i < 1                      # 只放行第 0 层的内容条件 ✓

        def cond(self, i, t_emb, y_emb):
            return t_emb + y_emb

    m.cond_router = _Router()
    seen = []
    orig = m.blocks[3].forward

    def spy(x, c, rope=None, c_style=None):
        seen.append(c_style is not None)
        return orig(x, c, rope=rope, c_style=c_style)

    m.blocks[3].forward = spy
    with torch.no_grad():
        _fwd(m)
    assert seen == [True], "风格支路在第 3 层被 cond_router 切断了 ✗（v85A 的核心修正 ✓）"
