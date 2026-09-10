#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""debug_bottleneck.py — 定位 strict 0.568 的平台瓶颈在哪一层.

三问:
  A. 预训练书家表是否真学开? (41行 pairwise cos, 塌缩=0.323 旧值对比)
  B. callig_scale / glyph_scale 各学到多少? (两条通路活性)
  C. 换书家, 同一骨架+同一噪声, 输出 latent 变多少? (风格是否可控)
"""
import sys, os, glob, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np
import torch

# ── A. 预训练书家表塌缩度 ──
print("=" * 60)
print("[A] 预训练书家表塌缩度 (5script/callig_emb_pretrained.pt)")
pt = torch.load("5script/callig_emb_pretrained.pt", map_location="cpu", weights_only=False)
print("  文件 keys:", list(pt.keys()) if isinstance(pt, dict) else type(pt))
emb = None
if isinstance(pt, dict):
    for k in ("embedding", "weight", "table"):
        if k in pt:
            emb = pt[k].float(); break
    if emb is None:
        # 可能整个就是张量
        emb = pt
else:
    emb = pt.float()
if emb is not None:
    emb = emb.detach().float()
    if emb.dim() == 3:
        emb = emb[0]
    nc = emb.shape[0]
    e = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
    sim = e @ e.T
    off = sim[~torch.eye(nc, dtype=torch.bool)]
    print(f"  书家表形状: {tuple(emb.shape)}")
    print(f"  pairwise cos = mean {off.mean():.4f} / std {off.std():.4f}  (0=正交学开, 0.323=旧塌缩, 1=全同)")
    print(f"  ||emb|| = mean {emb.norm(dim=1).mean():.3f} ± {emb.norm(dim=1).std():.3f}")

# ── B. 模型里 callig/glyph scale ──
print("=" * 60)
print("[B] 最新 c41x_cos_e ckpt 的通路活性")
ck_paths = sorted(glob.glob("5script/results/v10b_stdskel_fame3_c41x_cos_e/*/checkpoints/*.pt"),
                  key=lambda p: int(os.path.basename(p).split(".")[0]))
if not ck_paths:
    ck_paths = sorted(glob.glob("5script/results/v10b_stdskel_fame3_c41x_cos/*/checkpoints/*.pt"),
                      key=lambda p: int(os.path.basename(p).split(".")[0]))
ck = torch.load(ck_paths[-1], map_location="cpu", weights_only=False)
a = ck.get("args", {}) or {}
sd = ck.get("ema") or ck.get("model") or ck
sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
cs = sd.get("callig_scale")
gs = sd.get("glyph_scale")
print(f"  ckpt: {os.path.basename(ck_paths[-1])}")
print(f"  callig_scale = {cs.item():.4f}  (init {a.get('callig_scale_init', 1.0)})" if cs is not None else "  callig_scale 不存在")
print(f"  glyph_scale  = {gs.item():.4f}  (init {a.get('glyph_scale_init', 0.6)})" if gs is not None else "  glyph_scale 不存在")

# 书家表在模型里的实际值 (freeze 后应 = 预训练表前41行)
print("=" * 60)
print("[C] 模型内书家表前41行塌缩度 (freeze 后应=预训练表)")
w = sd.get("y_callig_embedder.embedding_table.weight")
if w is not None:
    w41 = w[:41].float()
    e = w41 / (w41.norm(dim=1, keepdim=True) + 1e-8)
    s = e @ e.T
    off = s[~torch.eye(41, dtype=torch.bool)]
    print(f"  模型内 41 书家 pairwise cos = mean {off.mean():.4f} / std {off.std():.4f}")
    print(f"  前41行与预训练表是否一致: (freeze 后应逐元素相等)")
else:
    print("  未找到 y_callig_embedder.embedding_table.weight")
print("=" * 60)