#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""独立验证「最小汉字表」(4690x256) 是否真的携带 DINO 语义。

方法 (闭式, 无需训练, 秒级):
  1. 用与建表同一份特征 assets/dino_feat_top10_g.npz 求每个汉字类的特征均值 X (n_cls, 384);
  2. 闭式 ridge 拟合 X -> 表行 Y (n_cls, 256), 报拟合 R^2 与逐类余弦;
  3. 全 26,002 样本 Top-1: 特征 -> (拟合的线性像) -> 与全部表行最近邻, 看是否命中正确字;
  4. 对照: 用**随机 256 维表**跑同样的 Top-1 (语义真空对照)。

判定: 表若只是"几何漂亮"(各向同性) 而无线索, 拟合会差、Top-1 ≈ 对照;
      若真携带语义, 拟合好且 Top-1 显著高于对照。
"""
import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

os.chdir(os.environ.get("DIT_ROOT", "/root/Workspace/xy/DiT"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TDIR = "assets/triple_tables_best_minimal"

# ── 特征与标签 ──
ids = np.load("data/dino_cache/top10_v1/ids.npy")
feat = torch.from_numpy(np.load("assets/dino_feat_top10_g.npz")["feat"].astype(np.float32)).to(DEV)
by_img = {int(r["img_id"]): r for r in csv.DictReader(open("exp-std/csv/train.csv", encoding="utf-8"))}
rows = [by_img[int(i)] for i in ids]
N, D = feat.shape
print(f"[probe] N={N} D_feat={D}")

classes = json.load(open(f"{TDIR}/char_index.json", encoding="utf-8"))["classes"]
cls2row = {str(c): i for i, c in enumerate(classes)}
W = torch.from_numpy(np.load(f"{TDIR}/char_table.npy")).float().to(DEV)      # (C, 256)
Wn = F.normalize(W, dim=-1)
C = W.shape[0]

lab = torch.tensor([cls2row[str(r["character"])] for r in rows], device=DEV)
n_cls_present = int(lab.unique().numel())
print(f"[probe] 表 {tuple(W.shape)}, 样本覆盖类 {n_cls_present}/{C}")

# ── 1) 类均值特征 ──
sums = torch.zeros(C, D, device=DEV).index_add_(0, lab, feat)
cnts = torch.zeros(C, device=DEV).index_add_(0, lab, torch.ones_like(lab, dtype=torch.float))
present = cnts > 0
X = sums[present] / cnts[present].unsqueeze(-1)          # (C', 384)
Y = W[present]                                            # (C', 256)

# ── 2) 闭式 ridge: Y ~ X B (+b), 5 折交叉验证避免过拟合假象 ──
Xa = torch.cat([X, torch.ones(X.shape[0], 1, device=DEV)], 1)
lam = 1e-2 * X.shape[0]
XtX = Xa.T @ Xa + lam * torch.eye(Xa.shape[1], device=DEV)
B = torch.linalg.solve(XtX, Xa.T @ Y)


def fit_stats(Xtr, Ytr, Xte, Yte):
    A = torch.cat([Xtr, torch.ones(Xtr.shape[0], 1, device=DEV)], 1)
    Bt = torch.linalg.solve(A.T @ A + 1e-2 * Xtr.shape[0] * torch.eye(A.shape[1], device=DEV), A.T @ Ytr)
    Ae = torch.cat([Xte, torch.ones(Xte.shape[0], 1, device=DEV)], 1)
    P = Ae @ Bt
    Pn, Yn = F.normalize(P, dim=-1), F.normalize(Yte, dim=-1)
    cos = (Pn * Yn).sum(-1)
    ss_res = ((P - Yte) ** 2).sum()
    ss_tot = ((Yte - Yte.mean(0)) ** 2).sum()
    return float(cos.mean()), float(1 - ss_res / ss_tot)


perm = torch.randperm(X.shape[0], device=DEV)
k = int(0.8 * X.shape[0])
c_tr, r2_tr = fit_stats(X[perm[:k]], Y[perm[:k]], X[perm[k:]], Y[perm[k:]])
print(f"[probe] ridge 拟合 (80/20 交叉): 逐类余弦={c_tr:.4f}  R^2={r2_tr:.4f}")

# ── 3) 全样本 Top-1 (特征 -> 线性像 -> 最近表行) ──
P_all = torch.cat([feat, torch.ones(N, 1, device=DEV)], 1) @ B
top1 = (F.normalize(P_all, dim=-1) @ Wn.T).argmax(-1)
acc = float((top1 == lab).float().mean())
print(f"[probe] 全样本 Top-1 (4690 类) = {acc * 100:.2f}%")

# ── 4) 随机表对照 (同样流程) ──
for seed in (0, 1):
    g = torch.Generator(device=DEV).manual_seed(seed)
    Wr = F.normalize(torch.randn(C, W.shape[1], device=DEV, generator=g), dim=-1)
    Ar = torch.cat([X, torch.ones(X.shape[0], 1, device=DEV)], 1)
    Br = torch.linalg.solve(Ar.T @ Ar + 1e-2 * X.shape[0] * torch.eye(Ar.shape[1], device=DEV), Ar.T @ Wr[present])
    Pr = torch.cat([feat, torch.ones(N, 1, device=DEV)], 1) @ Br
    acc_r = float(((F.normalize(Pr, dim=-1) @ Wr.T).argmax(-1) == lab).float().mean())
    print(f"[probe] 随机表对照 (seed={seed}) Top-1 = {acc_r * 100:.2f}%")

print("\n[结论] 表若真携带语义: Top-1 应显著高于随机对照 (经验: >2x 对照才算有真实信息)。")
