#!/usr/bin/env python3
"""pretrain_callig_emb.py — 对比预训练书家 embedding (方案 i, 2026-09-08).

输入: DINO v2 CLS 特征 (51321×384, 含 raw calligrapher_id) + 干净词表映射 json
目标: 41×128 的 embedding 表, 端到端 DiT 训练下会塌缩 (pairwise cos 0.323),
      这里用显式监督训开: SupCon (同书家正对/异书家负对) + DINO 质心锚定.

  SupCon:  z = P(E[y]),  InfoNCE 拉近同书家/推远异书家 (temp 0.07)
  锚定:    1 - cos(A·E[y], centroid[y]), 让 embedding 继承 DINO 风格语义
           (P/A 是投影头, 训练后丢弃; E 即最终接入 DiT 的表)

输出: 5script/callig_emb_pretrained.pt  {"embedding": (41,128), "raw_ids": [...]}
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dino-npz", default="5script/dino_cls_train.npz")
    ap.add_argument("--id-map", default="5script/callig_id_map.json")
    ap.add_argument("--out", default="5script/callig_emb_pretrained.pt")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--w-anchor", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device("cuda")

    # ── 数据 ────────────────────────────────────────────────────────────
    from src.utils.callig_map import load_callig_id_map
    cmap, n_vocab = load_callig_id_map(args.id_map)
    d = np.load(args.dino_npz)
    feat_all = d["feat"].astype(np.float32)                 # (M, 384)
    raw_all = np.array([int(x) for x in d["calligs"].tolist()])
    keep = np.array([int(r) in cmap for r in raw_all])
    feat_all, raw_all = feat_all[keep], raw_all[keep]
    lab_all = np.array([cmap[int(r)] for r in raw_all])     # 0..40
    print(f"[data] DINO feats {feat_all.shape}, 词表 {n_vocab}, "
          f"覆盖样本 {len(lab_all)}")
    assert n_vocab == len(set(lab_all.tolist())), "词表与 DINO 标签不齐"

    feat = torch.from_numpy(feat_all).to(dev)
    lab = torch.from_numpy(lab_all).to(dev)
    feat_n = F.normalize(feat, dim=-1)

    # DINO 质心锚 (41×384, L2 归一化)
    C = torch.stack([feat_n[lab == i].mean(0) for i in range(n_vocab)])
    C = F.normalize(C, dim=-1)

    # ── 模型 ────────────────────────────────────────────────────────────
    E = nn.Embedding(n_vocab, args.dim).to(dev)
    # std 1.0: 若用 0.02, P 的 bias 量级 > 输入分量, 41 个方向全部塌到 bias
    # 方向 (logits 恒 = 1/temp, SupCon 退化), 实测 NaN
    nn.init.normal_(E.weight, std=1.0)
    P = nn.Sequential(nn.Linear(args.dim, args.dim, bias=False), nn.GELU(),
                      nn.Linear(args.dim, args.dim, bias=False)).to(dev)
    A = nn.Linear(args.dim, feat.shape[1]).to(dev)
    opt = torch.optim.Adam(list(E.parameters()) + list(P.parameters())
                           + list(A.parameters()), lr=1e-3)

    def pairwise_cos(w):
        z = F.normalize(w, dim=-1)
        return (z @ z.T)

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(lab), (args.batch,), device=dev)
        y = lab[idx]
        z = F.normalize(P(E(y)), dim=-1)
        logits = z @ z.T / args.temp
        eye = torch.eye(args.batch, dtype=torch.bool, device=dev)
        mask_pos = (y[:, None] == y[None, :]).float() * (~eye).float()
        n_pos = mask_pos.sum(1).clamp_min(1)
        # 标准稳定写法: 对角线置 -inf 后 logsumexp (内部自带 max 减除)
        log_prob = logits - torch.logsumexp(
            logits.masked_fill(eye[:, :], -float("inf")), dim=1, keepdim=True)
        loss_sup = -(log_prob * mask_pos).sum(1).div(n_pos).mean()
        # 质心锚定
        loss_anc = (1 - F.cosine_similarity(A(E(y)), C[y], dim=-1)).mean()
        loss = loss_sup + args.w_anchor * loss_anc
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == 1:
            with torch.no_grad():
                w = E.weight
                pc = pairwise_cos(w)
                off = pc[~torch.eye(n_vocab, dtype=torch.bool, device=dev)]
                ctx = F.cosine_similarity(A(E.weight), C, dim=-1)
            print(f"step {step}: sup={loss_sup:.4f} anchor={loss_anc:.4f} "
                  f"|pair cos| mean={off.abs().mean():.4f} "
                  f"cos-anchor={ctx.mean():.4f}", flush=True)

    w = E.weight.detach().cpu()
    pc = pairwise_cos(w)
    off = pc[~torch.eye(n_vocab, dtype=torch.bool)]
    print(f"\n[最终] 41 表 pairwise cos: mean={off.mean():.4f} |mean|={off.abs().mean():.4f} "
          f"max={off.max():.4f} min={off.min():.4f}  (塌缩基线 0.323)")
    torch.save({"embedding": w, "raw_ids": sorted(cmap.keys())},
               args.out)
    print(f"[ok] -> {args.out}")


if __name__ == "__main__":
    main()
