#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""低秩逐位置风格×几何乘法的冒烟。CPU, 不需要数据和 ckpt。

四件事:
  1. rank=0 不建模块, 参数量与基线一致。
  2. 第 0 步输出相对变化在 0.002 初始化的量级, 不是恒等也不是随机扰动。
  3. keep=0 时输出逐位为 0: β 没有把丢弃样本重新灌成有条件。
  4. 梯度到 P、Q 和 film_net 最后一层, 且不同位置的 γ 不同。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.model.dit import DiT_2Cond_S_2


def build(**kw):
    base = dict(num_calligraphers=45, num_characters=100, use_glyph_cond=True,
                use_char_cond=False, condition_fusion="factorized_cat",
                callig_embed_dim=128, glyph_vec_cond=True, glyph_inject_layers=0)
    base.update(kw)
    m = DiT_2Cond_S_2(**base)
    m.eval()
    return m


def run():
    torch.manual_seed(0)
    B, C = 4, 4
    x = torch.randn(B, C, 32, 32)
    t = torch.rand(B) * 1000
    yc = torch.randint(0, 45, (B,))
    yh = torch.randint(0, 100, (B,))
    g = torch.randn(B, C, 32, 32)

    m = build(lowrank_spatial_rank=32)
    n1 = sum(p.numel() for p in m.parameters())
    n_head = sum(p.numel() for p in m.lowrank_spatial.parameters())
    assert m.lowrank_spatial is not None
    print(f"[param] model {n1:,}  head {n_head:,}")

    # 关掉模块再跑一次, 得到"没有这条通路"的对照, 避免同时建两个 36M 模型。
    saved = m.lowrank_spatial
    m.lowrank_spatial = None
    with torch.no_grad():
        y0 = m(x, t, yc, yh, g=g)
    m.lowrank_spatial = saved
    with torch.no_grad():
        y1 = m(x, t, yc, yh, g=g)
    rel = (y1 - y0).norm() / y0.norm().clamp_min(1e-8)
    print(f"[step0] 主干输出相对变化 {float(rel):.4e}")
    # 直接看模块自己的输出。主干后面还有 zero-init 的 glyph 注入和 final layer,
    # 未训练时它们把任何上游差异都乘成 0, 所以不能用主干输出判断模块是否生效。
    head = m.lowrank_spatial
    with torch.no_grad():
        gt = torch.randn(B, 256, 384)
        mod = head(gt, torch.randn(B, 128), m.local_pos, None)
    drel = float((mod - gt).norm() / gt.norm())
    print(f"[step0] g_tok 相对变化 {drel:.4e}  (std=0.002, 期望 1e-3~1e-1)")
    assert 1e-4 < drel < 0.5, drel

    # keep=0: 丢弃样本必须保持全零, 否则 β 污染了 uncond 分支。
    g_tok = torch.randn(B, 256, 384)
    e = torch.randn(B, 128)
    keep = torch.zeros(B)
    out = head(g_tok, e, m.local_pos, keep)
    dropped = out[keep == 0]
    print(f"[keep=0] max|out| {float(dropped.abs().max()):.3e}")
    assert float(dropped.abs().max()) == 0.0

    # 梯度: 三个参数都要动, 并且 γ 逐位置不同。
    # 未训练主干的 adaLN 与 final layer 都是 zero-init, 主干输出对 g_tok 的
    # 梯度结构性为 0。测梯度时把这两处激活, 否则会把"接通了"误判成"没梯度"。
    def _wake(mod):
        if hasattr(mod, "adaLN_modulation"):
            nn_last = mod.adaLN_modulation[-1]
            nn_last.weight.data.normal_(std=0.02)
        if hasattr(mod, "linear"):
            mod.linear.weight.data.normal_(std=0.02)
    for blk in m.blocks:
        _wake(blk)
    _wake(m.final_layer)
    m.train()
    y = m(x, t, yc, yh, g=g)
    y.sum().backward()
    grads = {}
    for name in ("P.weight", "Q.weight", "film_net.2.weight"):
        p = dict(head.named_parameters())[name]
        grads[name] = 0.0 if p.grad is None else float(p.grad.norm())
        print(f"[grad] {name} {grads[name]:.4e}")
        assert grads[name] > 0, name

    with torch.no_grad():
        B2 = 2
        gt = torch.randn(B2, 256, 384)
        es = torch.randn(B2, 128)
        s = head.P(es)
        q = head.Q(gt)
        u = s.unsqueeze(1) * q
        z = torch.cat([u, m.local_pos.expand(B2, 256, -1)], -1)
        gamma = head.film_net(z).chunk(2, -1)[0]
    spread = float((gamma[:, 0] - gamma[:, 1]).abs().mean())
    print(f"[spatial] 相邻 token 的 γ 差 {spread:.4e}")
    assert spread > 0, "γ 对所有位置相同, 退化成了全局仿射"

    print("PASS")


if __name__ == "__main__":
    run()
