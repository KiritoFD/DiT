"""风格 token 路由探针 v2: 真 token x 温度锐化 x 两种竞争设定。

## v1 的缺陷 (已修)
v1 只在"4 个风格 token 列"内部算归一化熵, 但它们的**绝对注意力质量**只有 ~1.8%
(与 256 个骨架 token 竞争 softmax), 尺度与探针三不可比 -> 会误判成"硬路由"。
本版同时给出两个互补量:

  (1) style_mass  : 风格 token 拿到的**绝对注意力质量占比**
                    (B 路 v15c 设定: 4 风格列 vs 256 骨架列一起 softmax)
                    -> 太低(如 <5%)说明风格信号即便被选中也只是噪声级 -> B 路第二死因
  (2) 熵(竞争 vs 独占):
      - 独占 (v15b 设定): 只在 4 个风格 token 间 softmax -> 直接量"能否区分"
      - 竞争 (v15c 设定): 从 260 列里看那 4 列 (归一化后) -> 反映真实路由
    两者都用 tau 扫: softmax(logits / sqrt(d) / tau)

## 判读
  独占熵 ≈ 1.0 (0.97) = 4 个 token 不可区分 -> 伪多模态 (历史 0.884 的后果)
  独占熵 < 0.6       = 真的能区分
  style_mass < 5%    = 风格信号被骨架淹没 -> B 路无救 (与"成本 +33%, 口径最差"一致)

用法: python tools/probe_style_routing.py [--tau 1.0,0.3,0.1] [--cls 0]
"""
import argparse
import importlib
import math
import os
import sys

import torch as th
import torch.nn as nn
import torch.nn.functional as F

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--kmeans-pt", default="assets/multistyle_k4_top10.pt")
ap.add_argument("--pca-pt", default="assets/pca_multistyle_k4_top10.pt")
ap.add_argument("--tau", default="1.0,0.5,0.3,0.1")
ap.add_argument("--cls", default="0,1,2", help="取哪些类的风格 token (逗号分隔)")
ap.add_argument("--hidden", type=int, default=384)
ap.add_argument("--grid", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
th.manual_seed(a.seed)
TOLS = [float(x) for x in a.tau.split(",")]
CLS = [int(x) for x in a.cls.split(",")]
D, G = a.hidden, a.grid
N = G * G
NK = 256                       # 骨架 token 数

CAPT = []


def _sdpa(q, k, v, *args, **kwargs):
    CAPT.append((q.detach().float(), k.detach().float()))
    return _ORIG(q, k, v, *args, **kwargs)      # 输出不重要, 只截获 q/k


_ORIG = F.scaled_dot_product_attention
F.scaled_dot_product_attention = _sdpa
for _mn in ("src.model.dit", "src.model.modules"):
    try:
        _m = importlib.import_module(_mn)
        for _al in ("sdpa", "scaled_dot_product_attention", "_sdpa"):
            if hasattr(_m, _al):
                setattr(_m, _al, _sdpa)
    except Exception:                                              # noqa: BLE001
        pass

from src.model.dit import ZeroCrossAttention, get_2d_sincos_pos_embed   # noqa: E402


def dezero(mod, gain=0.02):
    for m in mod.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            if float(m.weight.detach().abs().max()) < 1e-8:
                with th.no_grad():
                    m.weight.normal_(0, gain)
                    if m.bias is not None:
                        m.bias.normal_(0, gain)


def load_tokens(p, cls):
    if not os.path.exists(p):
        return None, None
    d = th.load(p, map_location="cpu", weights_only=False)
    if isinstance(d, dict) and "centroids" in d:
        c = d["centroids"].float()
    elif isinstance(d, dict) and "embedding" in d:
        e = d["embedding"].float()
        c = e.view(e.shape[0], int(d.get("k_clusters", 4)), -1)
    else:
        return None, None
    tok = c[cls]
    n = tok / tok.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    sim = n @ n.T
    off = ~th.eye(tok.shape[0], dtype=th.bool)
    return tok, (float(sim[off].mean()), float(sim[off].max()))


def stats(w):
    """w: (..., M) 已归一化的一维分布 -> (归一化熵, 最大权重)。"""
    K = w.shape[-1]
    ent = -(w.clamp_min(1e-12).log() * w).sum(-1)
    return float(ent.mean() / math.log(K)), float(w.max(-1).values.mean())


pos = th.from_numpy(get_2d_sincos_pos_embed(D, G)).float().unsqueeze(0)
x = th.randn(1, N, D) * 0.5
g = th.randn(1, N, D) * 0.5
role = th.randn(4, D) * 0.02

mod = ZeroCrossAttention(D, num_heads=8, grid_size=G, q_pos=True).eval()
dezero(mod)
for p in mod.parameters():
    p.requires_grad_(False)

print("=" * 96)
print("风格 token 路由探针 v2  (真 token 表 x tau x 两种竞争设定)")
print("  独占 = 只在 K 个风格 token 间 softmax (v15b 设定, 量'能否区分')")
print("  竞争 = 4 风格列 + 256 骨架列一起 softmax (v15c/B 路设定, 量'风格能否浮出')")
print("=" * 96)

agg = {}
for tag, path in (("K-Means K4", a.kmeans_pt), ("PCA 正交 K4", a.pca_pt)):
    if not os.path.exists(path):
        print(f"\n[{tag}] 跳过: {path} 不存在")
        continue
    print(f"\n########## {tag}  ({path}) ##########")
    for cls in CLS:
        tok, geo = load_tokens(path, cls)
        if tok is None:
            continue
        K = tok.shape[0]
        ctx = th.cat([g + pos, tok.unsqueeze(0) + role[:K].unsqueeze(0)], dim=1)
        CAPT.clear()
        with th.no_grad():
            mod(x, ctx)
        q, k = CAPT[-1]                                    # (1,H,256,d),(1,H,260,d)
        scale = 1.0 / math.sqrt(q.shape[-1])
        logits = (q @ k.transpose(-2, -1)) * scale         # (1,H,256,260)
        print(f"  --- 类{cls}: token 两两余弦 均值={geo[0]:.4f} 最大={geo[1]:.4f} ---")
        for tau in TOLS:
            lg = logits / tau
            # 竞争: 全 260 列 softmax
            w_all = th.softmax(lg, dim=-1)
            mass = float(w_all[..., NK:].sum(-1).mean())    # 风格列占总质量
            w_comp = w_all[..., NK:]
            w_comp = w_comp / w_comp.sum(-1, keepdim=True).clamp_min(1e-12)
            e_c, m_c = stats(w_comp)
            # 独占: 只对 4 个风格列 softmax
            w_exc = th.softmax(lg[..., NK:], dim=-1)
            e_e, m_e = stats(w_exc)
            agg.setdefault((tag, tau), []).append((geo[0], mass, e_e, m_e, e_c, m_c))
            print(f"    tau={tau:<4} 风格质量占比={mass * 100:6.2f}%  |  独占熵={e_e:.4f} "
                  f"最大权重={m_e:.4f}  |  竞争熵(归一化)={e_c:.4f}")

print("\n" + "=" * 104)
print(f"{'token 几何':<14} {'tau':>5} {'风格质量%':>10} {'独占熵':>9} {'独占最大权':>11} "
      f"{'竞争熵':>8}  裁决")
print("-" * 104)
for key in sorted(agg, key=lambda z: (z[0], -z[1])):
    v = agg[key]
    tag, tau = key
    cos = sum(t[0] for t in v) / len(v)
    mass = sum(t[1] for t in v) / len(v) * 100
    e_e = sum(t[2] for t in v) / len(v)
    m_e = sum(t[3] for t in v) / len(v)
    e_c = sum(t[4] for t in v) / len(v)
    verdict = []
    if e_e > 0.95:
        verdict.append("✗ 独占仍均匀(不可区分)")
    elif e_e < 0.6:
        verdict.append("✓ 可区分")
    if mass < 5:
        verdict.append("✗ 风格被骨架淹没(<5%)")
    print(f"{tag:<14} {tau:>5} {mass:>10.2f} {e_e:>9.4f} {m_e:>11.4f} "
          f"{e_c:>8.4f}  {' | '.join(verdict) if verdict else '~ 中性'}")

print("\n判读: '独占熵' 回答能否区分 (K-Means 0.9 共线 -> 应接近 1.0);")
print("      '风格质量占比' 回答 B 路是否还有救 (风格列在 260 列竞争里能否浮出)。")
