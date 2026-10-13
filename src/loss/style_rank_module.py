#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""StyleRankLoss —— 冻结的 2.9M latent 编码器 + 书家×书体质心 -> 风格损失。

两种模式（**默认仍是旧行为** ✓，新路径必须显式开启 ✓ —— 不静默改变历史结果 ✓）

    mode="cos"（旧 ✓）
        ``L = 1 - cos(f(pred_x0), μ_target)``
        只要求"接近目标质心" ✗ —— 方案指出它不够: 它不要求**比其他书家更像目标** ✓。

    mode="rank"（v85B，方案 §三.3 ✓）
        ``L = log(1 + Σ_{j≠w} exp((s_j - s_w + margin)/τ))``
        其中 ``s_j = cos(f(pred_x0), μ_j)``，``j`` 取**同书体**的其他书家 ✓。
        ⇒ 不仅要求"像颜真卿"，还要求**比其他书家更像颜真卿** ✓。

⚠ 关键实现约束（方案 §三.3 明确列出的三点 ✓）
    1. **不要在编码器前向周围用 `torch.no_grad()`** ✗ —— 编码器参数冻结（`requires_grad_(False)` ✓），
       但梯度必须从风格损失穿过编码器传回 `pred_xstart` 再回到 DiT ✓（否则风格损失对生成毫无作用 ✓）。
    2. **不能直接复用旧的 `rank_cent87.npy`** ✗ —— 它的类别索引对应 **45 书家 × 87 pairs**，
       而 v84A 只有 **10 书家** ⇒ 索引语义不匹配 ✓。必须用**已验证的编码器**在真迹上重建 ✓。
    3. 原型只能由**训练真迹**构建 ✓（不同字符/不同作品 ✓）；**不能用目标 GT 建原型** ✗（那是泄漏 ✓）。

用法（训练侧）::
    crit = StyleRankLoss(ckpt=<validated.pt>, protos=<protos.npz>, mode="rank")
    loss, n = crit(pred_xstart, y_writer, y_script, t)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Enc(nn.Module):
    """(4,32,32) -> 256D L2 归一化嵌入（+ 分类头）。

    ⚠ `nc` 必须与 ckpt 一致；不一致时 `emb()` 仍然可用（只加载 body/head ✓），
      因为风格损失**只用 `emb`** ✓ ⇒ 不要把 clf 的尺寸当成硬约束 ✓。
    """

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


def load_encoder(ckpt, d=256, ch=96, nc=None, verbose=False):
    """加载编码器（**只严格要求 body/head** ✓；clf 尺寸不符时跳过它 ✓）。

    为什么: 训练脚本产出的 ckpt 里 `clf` 的类别数 = 数据里的书家数（如 10 ✓），
    而旧模块硬编码 45 ✗ ⇒ `load_state_dict` 会因 `clf.weight` 形状不符而失败 ✓。
    风格损失只用 `emb` ✓，所以正确做法是**跳过 clf** ✓，而不是把 nc 猜对 ✗。
    """
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck) if isinstance(ck, dict) else ck
    sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    # ⚠ 类别数必须**从 clf.weight 反推** ✓ —— 我第一版写成
    #   `nc or (ck.get("gates") and 45) or sd[...]` ✗，`gates` 为真时**短路成 45** ✗
    #   ⇒ 与 ckpt 的 (10,256) 尺寸不符 ⇒ load 报 size mismatch ✓（实测踩到 ✓）。
    if nc is not None:
        _nc = int(nc)
    elif "clf.weight" in sd:
        _nc = int(sd["clf.weight"].shape[0])
    else:
        _nc = 45
    enc = _Enc(nc=_nc, d=int(ck.get("d", d)) if isinstance(ck, dict) else d,
               ch=int(ck.get("ch", ch)) if isinstance(ck, dict) else ch)
    miss, unexp = enc.load_state_dict(sd, strict=False)
    if verbose:
        print("[style-rank] 加载 %s: missing=%s unexpected=%s" % (ckpt, list(miss), list(unexp)))
    return enc, (ck if isinstance(ck, dict) else {})


class StyleRankLoss(nn.Module):
    """见模块 docstring ✓。`mode="cos"` = 旧行为 ✓；`mode="rank"` = 方案 §三.3 ✓。"""

    def __init__(self, ckpt="assets/style_enc_latent.pt", cent_npy="assets/rank_cent87.npy",
                 t_min=0.05, t_max=0.25, mode="cos", margin=0.0, tau=0.07,
                 protos=None, nc=None, verbose=False):
        super().__init__()
        assert mode in ("cos", "rank"), "mode 只能是 cos/rank ✗"
        self.mode, self.margin, self.tau = mode, float(margin), float(tau)
        self.t_min, self.t_max = float(t_min), float(t_max)
        enc, ck = load_encoder(ckpt, nc=nc, verbose=verbose)
        self.enc = enc.eval().requires_grad_(False)
        if protos:
            z = np.load(protos, allow_pickle=True)
            cent = z["protos"].astype(np.float32)                  # (P,256) **按 pair id 索引** ✓
            self.register_buffer("cent", torch.from_numpy(cent))
            # ★ rival 掩码: (P, P)，rival[i, j] = "j 与 i 同书体但不同书家" ✓
            #   预计算 ⇒ **不需要在训练里传 y_script** ✓（batch 只用现成的 y_pair ✓）
            rv = z["rival"].astype(np.bool_)                       # (P,P) ✓
            self.register_buffer("rival_mask", torch.from_numpy(rv))
            self.register_buffer("pair_ok", torch.from_numpy(
                z["pair_ok"].astype(np.bool_)))                    # (P,) 该 pair 有原型 ✓
            self.n_pair = int(cent.shape[0])
        else:
            cent = torch.from_numpy(np.load(cent_npy)).float()      # (87,256) 旧表 ✓
            self.register_buffer("cent", cent)
            self.register_buffer("rival_mask", torch.ones(cent.shape[0], cent.shape[0],
                                                          dtype=torch.bool))
            self.register_buffer("pair_ok", torch.ones(cent.shape[0], dtype=torch.bool))
            self.n_pair = int(cent.shape[0])

    def forward(self, pred_xstart, y_pair, t, y_script=None):
        """pred_xstart (B,4,32,32) 已含 VAE scaling；t (B,) ∈ [0,1] ✓。

        y_pair 恒为 **pair id（书家×书体）** ✓；两种模式都用它索引 `cent` ✓
        （`mode="rank"` 用预计算的 `rival_mask` 取"同书体异书家" ✓ ⇒ 不需要 y_script ✓）。
        """
        mask = (t >= self.t_min) & (t <= self.t_max)
        if not bool(mask.any()):
            return torch.tensor(0.0, device=pred_xstart.device), 0
        xs = pred_xstart[mask].float()
        f = self.enc.emb(xs)                          # ★ 无 no_grad ⇒ 梯度回到 xs ✓
        yp = y_pair[mask].long()
        if self.mode == "cos":                        # 旧行为 ✓ 逐位不变 ✓
            tgt = self.cent[yp]
            loss = (1.0 - (f * tgt).sum(-1)).mean()
            return loss, int(mask.sum())

        # ---- rank: log(1 + Σ_{j≠w 且同书体} exp((s_j − s_w + m)/τ)) ✓ ----
        sw = (f * self.cent[yp]).sum(-1)              # (n,) 目标相似度 ✓
        sj = f @ self.cent.t()                        # (n,P) ✓
        rival = self.rival_mask[yp]                   # (n,P) ✓ 同书体异书家 ✓
        diff = (sj - sw.unsqueeze(1) + self.margin) / self.tau
        loss = torch.log1p((torch.exp(diff) * rival).sum(1)).mean()
        return loss, int(mask.sum())


__all__ = ["StyleRankLoss", "load_encoder"]
