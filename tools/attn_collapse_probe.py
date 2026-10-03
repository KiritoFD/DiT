"""探针三: Attention Map 熵与塌缩排查 (Forward Only, 秒级)。

目的: 排除 Cross-Attention 的"初始化塌缩" —— 很多 CA 方案不 work 的隐藏杀手。
  危险信号 A (均匀塌缩): 每个 query 的权重都是 [0.25,0.25,0.25,0.25] -> 等价于 Mean-Pooling ✗
  危险信号 B (单点塌缩): 所有 query 都盯着同一个 token ([1,0,0,0]) -> 寻址死机 ✗
  健康信号: 不同 query 位置挑不同 token (权重分布**异构**) ✓

做法: 猴子补丁 torch.nn.functional.scaled_dot_product_attention, 截获真实 softmax
      权重 (config 里 attn_impl=sdpa, 一定走这条路径), 不改任何模型代码。

用法: python tools/attn_collapse_probe.py [--hidden 384] [--grid 16] [--k 4]
"""
import argparse
import importlib
import math
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
ap.add_argument("--head-dim", type=int, default=64)
ap.add_argument("--gain", type=float, default=0.02)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
th.manual_seed(a.seed)
dev = th.device("cuda" if th.cuda.is_available() else "cpu")
D, G, K = a.hidden, a.grid, a.k
N = G * G

CAPT = []


def _capture_sdpa(q, k, v, *args, **kwargs):
    """等价于 SDPA 的数学路径, 顺手把 softmax 权重存下来。"""
    try:
        scale = 1.0 / math.sqrt(q.shape[-1])
        w = th.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
        CAPT.append(w.detach())
        return w @ v
    except Exception:                                          # noqa: BLE001
        return _ORIG_SDPA(q, k, v, *args, **kwargs)


_ORIG_SDPA = F.scaled_dot_product_attention
F.scaled_dot_product_attention = _capture_sdpa
# 模型里可能以别名导入过, 一并替换
_PATCHED = ["torch.nn.functional.scaled_dot_product_attention"]
for _mn in ("src.model.dit", "src.model.modules", "src.model.injections"):
    try:
        _m = importlib.import_module(_mn)
        for _alias in ("sdpa", "scaled_dot_product_attention", "_sdpa"):
            if hasattr(_m, _alias):
                setattr(_m, _alias, _capture_sdpa)
                _PATCHED.append(f"{_mn}.{_alias}")
    except Exception:                                          # noqa: BLE001
        pass
print(f"[patch] {_PATCHED}")

from src.model.dit import (CalligStyleCrossAttn, GlyphStyleCrossAttn,   # noqa: E402
                           SpatialStyleFiLM)
from src.model.injections import ZeroAdaLNInjection              # noqa: E402

try:
    from src.model.dit import ZeroCrossAttention                        # noqa: E402
except Exception:                                                       # noqa: BLE001
    ZeroCrossAttention = None


def dezero(mod, gain):
    n = 0
    for m in mod.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            if float(m.weight.detach().abs().max()) < 1e-8:
                with th.no_grad():
                    m.weight.normal_(0, gain)
                    if m.bias is not None:
                        m.bias.normal_(0, gain)
                n += 1
    return n


def report(tag, w, note=""):
    """w: (B, H, Nq, Nk) 的注意力权重。"""
    B, H, Nq, Nk = w.shape
    ent = -(w.clamp_min(1e-12).log() * w).sum(-1)               # (B,H,Nq)
    ent_n = float(ent.mean() / math.log(Nk))                    # 归一化熵 (1=均匀)
    # 每个 query 的最大权重
    mx, am = w.max(-1)
    mx_mean = float(mx.mean())
    # 均匀塌缩: 权重与均匀分布的偏差
    uni_dev = float((w - 1.0 / Nk).abs().mean())
    # 单点塌缩: "所有 query 都挑同一个 token" 的比例
    am_flat = am.reshape(-1)                                     # (B*H*Nq,)
    top = th.bincount(am_flat, minlength=Nk).float()
    dominant = float(top.max() / top.sum())
    # 异构性: 各位置 argmax 的熵 (Bits) —— 越大越说明不同位置挑不同 token
    p = top / top.sum()
    hetero = float(-(p * (p + 1e-12).log()).sum())
    # 按位置看权重是否有变化 (query 间标准差)
    across_q = float(w.reshape(-1, Nk).std(0).mean())             # 各 token 权重在位置间的波动
    print(f"\n[{tag}] attn {tuple(w.shape)}  {note}")
    print(f"  归一化熵(1=均匀) = {ent_n:.4f}   偏离均匀 = {uni_dev:.4f}   最大权重均值 = {mx_mean:.4f}")
    print(f"  argmax token 分布 = {[int(x) for x in top.tolist()]}  "
          f"最集中占比 = {dominant:.3f}  位置异构熵 = {hetero:.3f} (上限 lnK={math.log(Nk):.3f})")
    print(f"  各 token 权重在位置间的波动(std 均值) = {across_q:.5f}")
    verdict = []
    if ent_n > 0.995 or uni_dev < 1e-3:
        verdict.append("✗ 均匀塌缩(等价 Mean-Pooling)")
    if mx_mean > 0.9 or dominant > 0.9:
        verdict.append("✗ 单点塌缩(寻址死机)")
    if across_q < 1e-4:
        verdict.append("✗ 跨位置无变化(对空间不敏感)")
    if not verdict:
        verdict.append("✓ 注意力异构, 未塌缩")
    print(f"  裁决: {' | '.join(verdict)}")
    return dict(tag=tag, ent=ent_n, uni=uni_dev, mx=mx_mean, dominant=dominant,
                hetero=hetero, across=across_q, verdict=" | ".join(verdict),
                shape=str(tuple(w.shape)))


res = []
CASES = [
    ("v15b CalligStyleCrossAttn", lambda: CalligStyleCrossAttn(D, num_heads=8, grid_size=G),
     lambda m, x, c, _: m(x, c), "Q=骨架token, K/V=风格token"),
    ("v15c GlyphStyleCrossAttn", lambda: GlyphStyleCrossAttn(D, num_heads=8),
     lambda m, x, c, _: m(x, c), "Q=骨架token, K/V=风格token"),
    ("v15a adaLN(无attention)", lambda: ZeroAdaLNInjection(D, mode="modulate"),
     lambda m, x, c, _: m(x, c.mean(1) if c.dim() == 3 else c), "对照: 结构上就没有 attention"),
    ("SPADE SpatialStyleFiLM", lambda: SpatialStyleFiLM(d_model=D, cond_dim=D),
     lambda m, x, c, _: m(x, c.mean(1), th.randn_like(x) * 0.01), "对照: 也无 attention"),
]
if ZeroCrossAttention is not None:
    CASES.insert(2, ("v15d ZeroCrossAttention(xattn)", 
                     lambda: ZeroCrossAttention(D, num_heads=8, grid_size=G, q_pos=True),
                     lambda m, x, c, _: m(x, c), "Q=噪声token, K/V=条件token"))

for tag, build, call, note in CASES:
    try:
        CAPT.clear()
        mod = build().to(dev).eval()
        n = dezero(mod, a.gain)
        for p in mod.parameters():
            p.requires_grad_(False)
        x = th.randn(1, N, D, device=dev) * 0.5
        c = th.randn(1, K, D, device=dev) * 0.5
        with th.no_grad():
            call(mod, x, c, None)
        if not CAPT:
            print(f"\n[{tag}] 无 attention 触发 (结构里就没有 SDPA) -> 记 '无attention'")
            res.append(dict(tag=tag, ent=float("nan"), uni=float("nan"), mx=float("nan"),
                            dominant=float("nan"), hetero=float("nan"), across=float("nan"),
                            verdict="— 结构上无 attention(不可能做 token 级空间寻址)",
                            shape="—"))
            continue
        w = CAPT[-1]                     # 最后一次(最深层)的注意力
        res.append(report(tag, w, f"{note} | 抬非零参数 {n}"))
    except Exception as e:                                     # noqa: BLE001
        print(f"\n[{tag}] ✗ 失败: {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)

print("\n" + "=" * 96)
print("探针三裁决表 (step0 纯前向, 秒级)")
print(f"{'注入方式':<32} {'归一化熵':>9} {'最大权重':>9} {'最集中占比':>10} {'位置异构熵':>10} {'裁决'}")
print("-" * 96)
for r in res:
    print(f"{r['tag']:<32} {r['ent']:>9.4f} {r['mx']:>9.4f} {r['dominant']:>10.3f} "
          f"{r['hetero']:>10.3f}  {r['verdict']}")
print("\n判读: 熵≈1 且 最大权重≈1/K -> 均匀塌缩(等于均值化) ✗; 最大权重≈1 -> 单点塌缩 ✗;")
print("      熵明显<1 且 argmax 分散 且 跨位置有波动 -> 健康 ✓")
