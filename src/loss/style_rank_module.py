#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""StyleRankLoss: 冻结的 2.9M latent 编码器 + 87 pair DINO 质心 -> cosine 距离 loss。

- pred_xstart (B,4,32,32, 已含 vae scaling) -> adapter(冻结, 无参数学习) -> Enc.emb (256d)
- 目标 = 该样本 (书家×书体) pair 的质心 (256d, 由 rank_cent87.npy 查表, batch['y_pair'])
- loss = 1 - cos(f(pred_x0), centroid(pair))，仅对 t∈[t_min,t_max] 的样本生效
- 全部参数冻结 (added=0)；BN 无 -> eval 模式下确定性。
- ⚠ adapter 无参数: 直接把 VAE latent 喂给编码器（它就是在这个 latent 空间训的）。
"""
import csv
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Enc(nn.Module):
    def __init__(self, nc=45, d=256, ch=96):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(4, ch, 3, 1, 1), nn.GELU(), nn.GroupNorm(8, ch),
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.GELU(), nn.GroupNorm(8, ch * 2),
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1), nn.GELU(), nn.GroupNorm(16, ch * 4),
            nn.Conv2d(ch * 4, ch * 4, 3, 1, 1), nn.GELU(), nn.GroupNorm(16, ch * 4),
            nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(ch * 4, d)
        self.clf = nn.Linear(d, nc)

    def emb(self, x):
        return F.normalize(self.head(self.body(x).flatten(1)), dim=-1)


class StyleRankLoss(nn.Module):
    def __init__(self, ckpt='assets/style_enc_latent.pt', cent_npy='assets/rank_cent87.npy',
                 t_min=0.05, t_max=0.25):
        super().__init__()
        self.t_min, self.t_max = t_min, t_max
        enc = _Enc()
        sd = torch.load(ckpt, map_location='cpu', weights_only=True)
        enc.load_state_dict(sd)
        self.enc = enc.eval().requires_grad_(False)
        cent = torch.from_numpy(np.load(cent_npy)).float()
        self.register_buffer('cent', cent)          # (87, 256)

    @torch.no_grad()
    def _enc_force_eval(self):
        pass

    def forward(self, pred_xstart, y_pair, t):
        """pred_xstart (B,4,32,32); y_pair (B,) long; t (B,) float in [0,1]."""
        mask = (t >= self.t_min) & (t <= self.t_max)
        if not bool(mask.any()):
            return torch.tensor(0.0, device=pred_xstart.device), 0
        xs = pred_xstart[mask].float()
        yp = y_pair[mask]
        with torch.no_grad():
            # 编码器全程冻结; 对 pred_xstart 不回传编码器参数, 但要回传 pred_xstart
            pass
        f = self.enc.emb(xs)                        # (n,256) 有 grad(xs)
        tgt = self.cent[yp]                         # (n,256)
        cos = (f * tgt).sum(-1)
        loss = (1.0 - cos).mean()
        return loss, int(mask.sum())