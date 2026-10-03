# -*- coding: utf-8 -*-
"""_chk_style_divergence.py — 风格 token 分化度诊断 (核心: 容量是否"有效")。

若 N 个 style token 两两 cos -> 1 (塌缩), 则 attention 权重对所有 token 几乎相同,
风格容量是虚设的(无论 N 多大)。本脚本量化:
  1) 同一书家内 32 个 token 的 pairwise cos 均值/max
  2) token 的有效秩 (SVD, 90% 能量维数) —— 真实可用自由度
  3) 不同书家的 style token 是否可分 (书家间 cos)
"""
import glob
import numpy as np
import torch
import torch.nn as nn

CKPT = sorted(glob.glob(
    "/root/Workspace/xy/DiT/assets/results/smoke_c41x_sty32/*/checkpoints/*.pt"))[-1]
print("ckpt:", CKPT)
d = torch.load(CKPT, map_location="cpu")
sd = d["delta"]


def pre(k):
    return sd["_orig_mod." + k]


# 还原 style_proj = Sequential(LayerNorm(128), Linear(128, n_style*hidden))
_ln_w = pre("style_proj.0.weight").float()
_ln_b = pre("style_proj.0.bias").float()
_lin_w = pre("style_proj.1.weight").float()          # (n*hidden, 128)
_lin_b = pre("style_proj.1.bias").float()
role = pre("style_role").float()                      # (n, hidden)
N_S, HID = role.shape
print(f"n_style={N_S} hidden={HID}  linear_out={_lin_w.shape[0]}")

# 书家 embedding: 用真实预训练表(若可得), 否则随机
try:
    E = torch.load("/root/Workspace/xy/DiT/assets/callig_emb_pretrained.pt",
                   map_location="cpu")
    if isinstance(E, dict):
        E = E.get("weight", E.get("emb", next(iter(E.values()))))
    E = torch.as_tensor(E).float()
    if E.ndim > 2:
        E = E.reshape(E.shape[0], -1)
    if E.shape[1] != 128:
        E = E[:, :128] if E.shape[1] > 128 else torch.randn(E.shape[0], 128)
    print(f"callig emb: {tuple(E.shape)} (真实表)")
except Exception as e:
    print("callig emb 加载失败, 用随机:", e)
    E = torch.randn(41, 128)
n_c = min(E.shape[0], 41)
E = E[:n_c]

# style token = Linear(LN(E)) + role
En = nn.functional.layer_norm(E, (128,), weight=_ln_w, bias=_ln_b)
S = En @ _lin_w.T + _lin_b                     # (n_c, n_style*hidden)
S = S.view(n_c, N_S, HID) + role.unsqueeze(0)  # 加 role

print("\n=== 1) 同一书家内 style token 分化度 (越低越好, 1.0=完全塌缩) ===")
intra_means = []
for i in range(min(5, n_c)):
    X = S[i]                                    # (N_S, HID)
    Xn = X / X.norm(dim=-1, keepdim=True)
    C = Xn @ Xn.T
    off = C[~torch.eye(N_S, dtype=bool)].reshape(N_S, -1)
    intra_means.append(float(off.mean()))
    print(f"  书家{i}: pairwise cos mean={off.mean():.4f}  max={off.max():.4f}")
print(f"  -> 平均 intra-cos = {np.mean(intra_means):.4f}")

print("\n=== 2) style token 有效秩 (真实自由度, 上限 n_style) ===")
ranks = []
for i in range(min(5, n_c)):
    X = S[i] - S[i].mean(0, keepdim=True)
    sv = torch.linalg.svdvals(X)
    e = (sv ** 2)
    e = e / e.sum()
    r = int(torch.searchsorted(torch.cumsum(e, 0), 0.90).item() + 1)
    ranks.append(r)
    print(f"  书家{i}: eff_rank={r}/{N_S}")
print(f"  -> 平均有效秩 = {np.mean(ranks):.1f}/{N_S}")

print("\n=== 3) 书家间可分性 (不同书家的 style token 集合差异) ===")
flat = S.reshape(n_c, -1)
flat = flat / flat.norm(dim=-1, keepdim=True)
CC = flat @ flat.T
off = CC[~torch.eye(n_c, dtype=bool)].reshape(n_c, -1)
print(f"  书家间 cos mean={off.mean():.4f}  max={off.max():.4f}")
print(f"  (应与 intra-cos 有明显差距 -> 书家间可分)")

print("\n判读:")
print("  intra-cos < 0.3 且 eff_rank 接近 n_style -> 风格容量**有效**")
print("  intra-cos > 0.8 或 eff_rank << n_style   -> 塌缩, 需增大 role_init / 正交正则 / 减 n_style")
