"""注入方式离线沙盒: 不跑扩散、不训练, 在 step0 判决"局部空间寻址能力"。

为什么成立: 所有注入头都 zero-init(out_proj=0) -> step0 输出恒等 ->
  单看输出没有信息 ✗; 唯一能判生死的, 是**梯度/雅可比的逐位置结构** ✓
  (把 out_proj 临时改成非零小值, 等价于"如果这个通路被训练起来, 它会怎么作用")。

两个判据 (都在 step0, 秒级):
  判据1 梯度溯源: 只对**某一个空间位置**的输出 backward -> 看 K 个风格 token 各自
        收到的梯度。若换到另一个位置, 梯度分布**不变** -> 全局调制(空间失明);
        若明显变化 -> 具备局部寻址。
  判据3 逐位置异构性: 固定骨架/画布, 只换风格条件 -> 测**每个位置**的输出变化量分布。
        均匀 = 全局; 强烈异构(有的位置被改、有的不变) = 局部。

用法: python tools/injection_probes.py [--hidden 384] [--grid 16] [--gain 0.02]
"""
import argparse
import os
import sys
import traceback

import torch as th
import torch.nn as nn
import torch.nn.functional as F

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--hidden", type=int, default=384)
ap.add_argument("--grid", type=int, default=16)
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--gain", type=float, default=0.02, help="把 zero-init 的 out_proj 抬到多大")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
th.manual_seed(a.seed)
dev = th.device("cuda" if th.cuda.is_available() else "cpu")
D, G, K = a.hidden, a.grid, a.k
N = G * G

from src.model.dit import (CalligStyleCrossAttn, GlyphStyleCrossAttn,   # noqa: E402
                           SpatialStyleFiLM)
from src.model.injections import ZeroAdaLNInjection              # noqa: E402

try:
    from src.model.dit import ZeroCrossAttention                        # noqa: E402
except Exception:                                                       # noqa: BLE001
    ZeroCrossAttention = None


def dezero(mod, gain):
    """把模块里所有 zero-init 的 Linear/Conv 抬成小随机值 (只在副本上做)。
    目的: zero-init 下 d(out)/d(x)=0, 梯度溯源会全零, 测不出结构。"""
    n = 0
    for m in mod.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            w = m.weight
            if float(w.detach().abs().max()) < 1e-8:
                with th.no_grad():
                    if isinstance(m, (nn.Linear, nn.Conv1d)):
                        w.normal_(0, gain)
                    else:
                        w.normal_(0, gain)
                    if m.bias is not None:
                        m.bias.normal_(0, gain)
                n += 1
    return n


def grad_trace(mod, call, x, cond1, cond2, tag):
    """判据1: 对位置 p 的输出 backward, 收集 cond1 的梯度; 换位置再测一次。
    返回 (每 token 梯度范数 @p0, @p1, 两位置梯度的余弦)。"""
    out = []
    for pos in (0, N - 1):
        c = cond1.clone().detach().requires_grad_(True)
        y = call(mod, x, c, cond2)                    # (B,N,D)
        # ★ autograd.grad 是**返回**梯度, 不是写进 .grad (踩过: 读 c.grad 得到 None)
        g = th.autograd.grad(y[0, pos].sum(), c, retain_graph=False)[0]
        out.append(g[0].detach())                          # (K,D)
    g0, g1 = out
    n0 = g0.norm(dim=1)                                    # (K,)
    n1 = g1.norm(dim=1)
    cos = F.cosine_similarity(g0.flatten(), g1.flatten(), dim=0).item()
    return n0, n1, cos


def hetero(mod, call, x, cond1, cond2, tag):
    """判据3: 换条件 -> 每个位置的输出变化量 (B,N) 的分布。"""
    with th.no_grad():
        y1 = call(mod, x, cond1, cond2)
        y0 = call(mod, x, th.zeros_like(cond1), cond2)
    d = (y1 - y0).norm(dim=-1)[0]                          # (N,)
    return d


def run(tag, build, call, needs_cond2=True):
    try:
        mod = build().to(dev).eval()
        n = dezero(mod, a.gain)
        x = th.randn(1, N, D, device=dev) * 0.5
        c1 = th.randn(1, K, D, device=dev) * 0.5
        c2 = th.randn(1, N, D, device=dev) * 0.5 if needs_cond2 else None
        for p in mod.parameters():
            p.requires_grad_(False)
        n0, n1, cos = grad_trace(mod, call, x, c1, c2, tag)
        try:
            d = hetero(mod, call, x, c1, c2, tag)
        except Exception as _he:                            # noqa: BLE001
            print(f"  [判据3] 跳过: {type(_he).__name__}: {_he}")
            d = None
        # 判据1 的"token 间差异": 4 个 token 梯度范数的相对标准差 (0 = 完全一样)
        cv0 = float(n0.std() / (n0.mean() + 1e-12))
        cv1 = float(n1.std() / (n1.mean() + 1e-12))
        # 判据3 的异构度: 逐位置变化量的归一化标准差 (0 = 全图均匀)
        hd = float(d.std() / (d.mean() + 1e-12)) if d is not None else float("nan")
        print(f"\n[{tag}] 抬非零参数 {n} 个")
        print(f"  判据1 token梯度范数 @位置0 = {[f'{v:.2e}' for v in n0.tolist()]}")
        print(f"         token梯度范数 @位置{N - 1} = {[f'{v:.2e}' for v in n1.tolist()]}")
        print(f"         两位置梯度的 cos = {cos:+.3f}  "
              f"(≈+1 = 无所谓位置, 空间失明; 明显<1 = 局部寻址)")
        print(f"         token 间不均匀度 cv = {cv0:.3f} -> {cv1:.3f} (0 = K 个 token 被同等对待)")
        print(f"  判据3 逐位置变化量 均值={float(d.mean()):.3e} std={float(d.std()):.3e} "
              f"异构度={hd:.3f} (0 = 全图同等影响)")
        return dict(tag=tag, cos=cos, cv0=cv0, cv1=cv1, hetero=hd,
                    mean=float(d.mean()))
    except Exception as e:                                  # noqa: BLE001
        print(f"\n[{tag}] ✗ 失败: {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)
        return None


res = []

# ① adaLN 全局调制 (基线): forward(x_tokens, feat(B,D))
res.append(run(
    "v15a adaLN 全局调制",
    lambda: ZeroAdaLNInjection(D, mode="modulate"),
    lambda m, x, c, _c2: m(x, c.mean(dim=1) if c.dim() == 3 else c),
    needs_cond2=False))

# ②/③ 骨架 cross-attn: forward(g_tok, style_tokens)
res.append(run(
    "v15b CalligStyleCrossAttn(骨架寻址)",
    lambda: CalligStyleCrossAttn(D, num_heads=8, grid_size=G),
    lambda m, x, c, _c2: m(x, c)))

# ④ GlyphStyleCrossAttn (本仓库专门给"风格 token 每层可见"那版)
res.append(run(
    "v15c GlyphStyleCrossAttn(每层注入)",
    lambda: GlyphStyleCrossAttn(D, num_heads=8),
    lambda m, x, c, _c2: m(x, c)))

# ⑤ DiT token 侧的 xattn (ZeroCrossAttention: x=噪声token, g_tok=条件token)
if ZeroCrossAttention is not None:
    res.append(run(
        "v15d ZeroCrossAttention(xattn)",
        lambda: ZeroCrossAttention(D, num_heads=8, grid_size=G, q_pos=True),
        lambda m, x, c, _c2: m(x, c),
        needs_cond2=False))

# ⑥ SPADE 式 (SpatialStyleFiLM): forward(g_tok, e_cond(B,D), pos(B,N,D))
res.append(run(
    "SPADE SpatialStyleFiLM",
    lambda: SpatialStyleFiLM(d_model=D, cond_dim=D),
    lambda m, x, c, _c2: m(x, c.mean(1), (_c2 if _c2 is not None
                                         else th.randn_like(x) * 0.01)),
    needs_cond2=False))

print("\n" + "=" * 84)
print("裁决表 (step0 纯数学, 无需训练)")
print(f"{'注入方式':<34} {'两位置cos':>10} {'token不均匀cv':>14} {'逐位置异构度':>13}")
print("-" * 84)
for r in res:
    if r:
        print(f"{r['tag']:<34} {r['cos']:>+10.3f} {r['cv0']:>14.3f} {r['hetero']:>13.3f}")
print("\n判读: cos≈+1 且 异构度≈0 -> 空间失明(全局调制); cos 明显<1 且 异构度大 -> 局部寻址 ✓")
