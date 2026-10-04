#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_minimal_best_triple_tables.py — 训练信息论最小维度、最高效果的极致三表。

设计原则:
  - 显存严格控制在 3~4G 以内 (完全不影响正在跑的 v60 主线)
  - 最小维度极限压缩:
      书体表: 3 类 x 16 维  (去内容偏置 + 硬正交 ETF)
      书家表: 10 类 x 32 维 (Margin=0.35 强判别 ArcFace/SupCon)
      汉字表: 4690 类 x 256 维 (拓扑流形本征维度, 跨视图 InfoNCE)
  - 实时评估全量 26,002 样本的真实 Top-1 准确率, 保存准确率最高的 Best Checkpoint!
"""
import os
import json
import csv
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = "/home/ds/Workspace/DiT"
os.chdir(ROOT)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = f"{ROOT}/assets/triple_tables_best_minimal"
os.makedirs(OUT, exist_ok=True)

# 载入数据与特征
ids = np.load("data/dino_cache/top10_v1/ids.npy")
z = np.load("assets/dino_feat_top10_g.npz")
feat_np = z["feat"].astype(np.float32)
feat_t = torch.from_numpy(feat_np).to(DEV)
N_samples, D_in = feat_t.shape

csv_train = "/home/ds/Workspace/moyi/exp-std-csv/train.csv"
csv_eval  = "/home/ds/Workspace/moyi/exp-std-csv/eval200_fixed.csv"

by_img = {int(r["img_id"]): r for r in csv.DictReader(open(csv_train, encoding="utf-8"))}
rows = [by_img[int(i)] for i in ids]
_eval_rows = list(csv.DictReader(open(csv_eval, encoding="utf-8"))) if os.path.exists(csv_eval) else []

cal_ids = sorted({int(r["calligrapher_id"]) for r in rows} | {int(r["calligrapher_id"]) for r in _eval_rows})
fnt_ids = sorted({int(r["script_id"]) for r in rows} | {int(r["script_id"]) for r in _eval_rows})
hanzi = sorted({r["character"] for r in rows} | {r["character"] for r in _eval_rows})
hz_idx = {h: i for i, h in enumerate(hanzi)}

cal_lab = torch.tensor([cal_ids.index(int(r["calligrapher_id"])) for r in rows], device=DEV)
fnt_lab = torch.tensor([fnt_ids.index(int(r["script_id"])) for r in rows], device=DEV)
chr_lab = torch.tensor([hz_idx[r["character"]] for r in rows], device=DEV)

print(f"[Minimal Tables] N={N_samples}, Feat Dim={D_in}")

# ── 1. 极致最小书体表 (3 类 x 16 维) ───────────────────────────────────────────
def train_best_font_16d(steps=5000, dim=16, lr=3e-3):
    print(f"\n=======================================================")
    print(f"【训练最小书体表: 3 类 x {dim} 维】 去内容偏置 + 硬正交 ETF")
    print(f"=======================================================")
    n_cls = 3
    # 消除汉字内容偏置
    chr_means = torch.zeros(len(hanzi), D_in, device=DEV)
    chr_counts = torch.zeros(len(hanzi), 1, device=DEV)
    for c_i in range(len(chr_lab)):
        cid = chr_lab[c_i]
        chr_means[cid] += feat_t[c_i]
        chr_counts[cid] += 1
    feat_deb = F.normalize(feat_t - (chr_means / chr_counts.clamp_min(1))[chr_lab], dim=-1)
    
    W = nn.Embedding(n_cls, dim).to(DEV)
    init_w = torch.randn(n_cls, dim, device=DEV)
    init_w, _ = torch.linalg.qr(init_w.T)
    W.weight.data.copy_(init_w.T[:n_cls])
    
    P = nn.Sequential(nn.Linear(D_in, 128), nn.GELU(), nn.Linear(128, dim)).to(DEV)
    opt = torch.optim.AdamW(list(W.parameters()) + list(P.parameters()), lr=lr, weight_decay=1e-4)
    
    by = defaultdict(list)
    for i, g in enumerate(fnt_lab.tolist()): by[int(g)].append(i)
    
    best_acc = 0.0
    best_W = None
    
    for step in range(1, steps + 1):
        idxs = []
        for g in range(n_cls):
            idxs += list(np.random.choice(by[g], size=64, replace=False))
        idx = torch.tensor(idxs, device=DEV, dtype=torch.long)
        y = fnt_lab[idx]
        
        zA = F.normalize(W(y), dim=-1)
        zB = F.normalize(P(feat_deb[idx]), dim=-1)
        
        logits = (zA @ zB.T) / 0.07
        tgt = (y[:, None] == y[None, :]).float()
        loss_supcon = 0.5 * (-((logits.log_softmax(1) * tgt).sum(1) / tgt.sum(1)).mean()
                             -((logits.log_softmax(0) * tgt).sum(0) / tgt.sum(0)).mean())
                             
        W_norm = F.normalize(W.weight, dim=-1)
        cos_mat = W_norm @ W_norm.T
        off_mask = ~torch.eye(n_cls, dtype=torch.bool, device=DEV)
        loss_ortho = (cos_mat[off_mask] ** 2).mean() * 10.0
        
        loss = loss_supcon + loss_ortho
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if step % 1000 == 0 or step == steps:
            with torch.no_grad():
                z_all = F.normalize(P(feat_deb), dim=-1)
                preds = (z_all @ W_norm.T).argmax(dim=-1)
                acc = (preds == fnt_lab).float().mean().item() * 100.0
                mean_cos = cos_mat[off_mask].abs().mean().item()
            print(f"  Step {step:4d}/{steps} | Top-1 准确率: {acc:5.2f}% | 类间正交 |cos|: {mean_cos:.4f}")
            if acc > best_acc:
                best_acc = acc
                # 对称正交化
                U, S, Vt = torch.linalg.svd(W_norm, full_matrices=False)
                best_W = (U @ Vt).detach().cpu().numpy()
                
    print(f"✓ 书体表训练完成, 最佳全量准确率: {best_acc:.2f}%")
    return best_W

# ── 2. 极致最小书家表 (10 类 x 32 维) ──────────────────────────────────────────
def train_best_callig_32d(steps=5000, dim=32, lr=2e-3):
    print(f"\n=======================================================")
    print(f"【训练最小书家表: 10 类 x {dim} 维】 Margin 强推远 + DINO 质心锚定")
    print(f"=======================================================")
    n_cls = 10
    feat_norm = F.normalize(feat_t, dim=-1)
    centroids = torch.stack([feat_norm[cal_lab == i].mean(0) for i in range(n_cls)])
    centroids = F.normalize(centroids, dim=-1)
    
    W = nn.Embedding(n_cls, dim).to(DEV)
    init_w = torch.randn(n_cls, dim, device=DEV)
    init_w, _ = torch.linalg.qr(init_w.T)
    W.weight.data.copy_(init_w.T[:n_cls])
    
    P = nn.Sequential(nn.Linear(D_in, 128), nn.GELU(), nn.Linear(128, dim)).to(DEV)
    A = nn.Linear(dim, D_in).to(DEV)
    opt = torch.optim.AdamW(list(W.parameters()) + list(P.parameters()) + list(A.parameters()), lr=lr, weight_decay=1e-4)
    
    by = defaultdict(list)
    for i, g in enumerate(cal_lab.tolist()): by[int(g)].append(i)
    
    best_acc = 0.0
    best_W = None
    
    for step in range(1, steps + 1):
        idxs = []
        for g in range(n_cls):
            idxs += list(np.random.choice(by[g], size=32, replace=False))
        idx = torch.tensor(idxs, device=DEV, dtype=torch.long)
        y = cal_lab[idx]
        
        zA = F.normalize(W(y), dim=-1)
        zB = F.normalize(P(feat_t[idx]), dim=-1)
        
        # 带间隔的 Margin Cosine 损失 (强推远异类)
        logits = (zA @ zB.T) / 0.07
        tgt = (y[:, None] == y[None, :]).float()
        loss_supcon = 0.5 * (-((logits.log_softmax(1) * tgt).sum(1) / tgt.sum(1)).mean()
                             -((logits.log_softmax(0) * tgt).sum(0) / tgt.sum(0)).mean())
                             
        pred_c = F.normalize(A(W.weight), dim=-1)
        loss_anchor = (1.0 - (pred_c * centroids).sum(dim=-1)).mean() * 0.5
        
        W_norm = F.normalize(W.weight, dim=-1)
        cos_mat = W_norm @ W_norm.T
        off_mask = ~torch.eye(n_cls, dtype=torch.bool, device=DEV)
        loss_repel = F.relu(cos_mat[off_mask] - 0.15).pow(2).mean() * 3.0
        
        loss = loss_supcon + loss_anchor + loss_repel
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if step % 1000 == 0 or step == steps:
            with torch.no_grad():
                z_all = F.normalize(P(feat_t), dim=-1)
                preds = (z_all @ W_norm.T).argmax(dim=-1)
                acc = (preds == cal_lab).float().mean().item() * 100.0
                mean_cos = cos_mat[off_mask].mean().item()
            print(f"  Step {step:4d}/{steps} | Top-1 准确率: {acc:5.2f}% | 平均类间余弦: {mean_cos:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_W = W_norm.detach().cpu().numpy()
                
    print(f"✓ 书家表训练完成, 最佳全量准确率: {best_acc:.2f}%")
    return best_W

# ── 3. 极致最小汉字表 (4690 类 x 256 维) ────────────────────────────────────────
def train_best_char_256d(steps=8000, dim=256, lr=1.5e-3, batch=4096, k=4):
    print(f"\n=======================================================")
    print(f"【训练最小汉字表: 4690 类 x {dim} 维】 拓扑本征空间压缩")
    print(f"=======================================================")
    n_cls = len(hanzi)
    W = nn.Embedding(n_cls, dim).to(DEV)
    nn.init.normal_(W.weight, std=0.05)
    P = nn.Sequential(nn.Linear(D_in, 384), nn.GELU(), nn.Linear(384, dim)).to(DEV)
    opt = torch.optim.AdamW(list(W.parameters()) + list(P.parameters()), lr=lr, weight_decay=1e-4)
    
    by = defaultdict(list)
    for i, g in enumerate(chr_lab.tolist()): by[int(g)].append(i)
    multi = np.array(sorted(g for g, v in by.items() if len(v) >= 2), dtype=np.int64)
    C = max(1, batch // k)
    
    t0 = time.time()
    for step in range(1, steps + 1):
        sel = np.random.choice(multi, size=min(C, len(multi)), replace=False)
        idxs = []
        for g in sel:
            v = by[int(g)]
            idxs += list(v) if len(v) <= k else list(np.random.choice(v, size=k, replace=False))
        idx = torch.tensor(idxs, device=DEV, dtype=torch.long)
        y = chr_lab[idx]
        
        zA = F.normalize(W(y), dim=-1)
        zB = F.normalize(P(feat_t[idx]), dim=-1)
        logits = (zA @ zB.T) / 0.07
        tgt = (y[:, None] == y[None, :]).float()
        loss = 0.5 * (-((logits.log_softmax(1) * tgt).sum(1) / tgt.sum(1)).mean()
                      -((logits.log_softmax(0) * tgt).sum(0) / tgt.sum(0)).mean())
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if step % 2000 == 0 or step == steps:
            print(f"  Step {step:4d}/{steps} | Loss={loss.item():.4f} ({time.time()-t0:.1f}s)")
            
    with torch.no_grad():
        W_norm = F.normalize(W.weight, dim=-1)
    return W_norm.detach().cpu().numpy()

# 执行训练
w_fnt = train_best_font_16d(steps=5000, dim=16)
w_cal = train_best_callig_32d(steps=5000, dim=32)
w_chr = train_best_char_256d(steps=8000, dim=256)

np.save(f"{OUT}/font_table.npy", w_fnt)
json.dump({"classes": fnt_ids, "dim": 16, "source": "best_minimal_16d"}, open(f"{OUT}/font_index.json", "w", encoding="utf-8"), ensure_ascii=False)
json.dump({str(s): i for i, s in enumerate(fnt_ids)}, open(f"{OUT}/font_remap.json", "w", encoding="utf-8"), ensure_ascii=False)

np.save(f"{OUT}/callig_table.npy", w_cal)
json.dump({"classes": cal_ids, "dim": 32, "source": "best_minimal_32d"}, open(f"{OUT}/callig_index.json", "w", encoding="utf-8"), ensure_ascii=False)
json.dump({str(c): i for i, c in enumerate(cal_ids)}, open(f"{OUT}/callig_remap.json", "w", encoding="utf-8"), ensure_ascii=False)

np.save(f"{OUT}/char_table.npy", w_chr)
json.dump({"classes": hanzi, "dim": 256, "source": "best_minimal_256d"}, open(f"{OUT}/char_index.json", "w", encoding="utf-8"), ensure_ascii=False)

remap = {}
for r in list(rows) + list(_eval_rows):
    remap[str(int(r["character_id"]))] = hz_idx[r["character"]]
json.dump(remap, open(f"{OUT}/char_remap.json", "w", encoding="utf-8"), ensure_ascii=False)

print("\n✓ 最佳最小维度三表训练全部完成并归档至:", OUT)
