# -*- coding: utf-8 -*-
"""pretrain_callig_emb_base.py — base 版书家表 SupCon 预训练 (52x128).

与 pretrain_callig_emb.py 同配方 (SupCon InfoNCE + DINO 质心锚定), 差异:
  - id_map = callig_id_map_base.json (num_calligraphers=52; base 实际 45 家,
    未出现的 continuous id 行保持随机 — 模型索引永远不会采到它们)
  - 质心只对出现过的 label 建; assert 放宽为 标签集合 ⊆ 词表
输出: assets/callig_emb_pretrained_base.pt {"embedding": (52,128), "raw_ids": [...]}
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
    ap.add_argument("--dino-npz", default="assets/dino_cls_base.npz")
    ap.add_argument("--id-map", default="assets/callig_id_map_base.json")
    ap.add_argument("--out", default="assets/callig_emb_pretrained_base.pt")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--w-anchor", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device("cuda")

    from src.utils.callig_map import load_callig_id_map
    cmap, n_vocab = load_callig_id_map(args.id_map)
    d = np.load(args.dino_npz)
    feat_all = d["feat"].astype(np.float32)
    raw_all = np.array([int(x) for x in d["calligs"].tolist()])
    keep = np.array([int(r) in cmap for r in raw_all])
    feat_all, raw_all = feat_all[keep], raw_all[keep]
    lab_all = np.array([cmap[int(r)] for r in raw_all])
    seen_labels = sorted(set(lab_all.tolist()))
    n_vocab = int(cmap.get("num_calligraphers", max(seen_labels) + 1))
    assert set(lab_all.tolist()) <= set(range(n_vocab)), "标签越界"
    print(f"[data] DINO feats {feat_all.shape}, 词表 {n_vocab}, "
          f"出现书家 {len(seen_labels)}, 样本 {len(lab_all)}")

    feat = torch.from_numpy(feat_all).to(dev)
    lab = torch.from_numpy(lab_all).to(dev)
    feat_n = F.normalize(feat, dim=-1)

    # 质心 (仅出现过的 label; 未出现行置零, 永远不会被 y 采样到)
    C = torch.zeros(n_vocab, feat.shape[1], device=dev)
    for i in seen_labels:
        C[i] = feat_n[lab == i].mean(0)
    C = F.normalize(C, dim=-1)

    E = nn.Embedding(n_vocab, args.dim).to(dev)
    nn.init.normal_(E.weight, std=1.0)
    P = nn.Sequential(nn.Linear(args.dim, args.dim, bias=False), nn.GELU(),
                      nn.Linear(args.dim, args.dim, bias=False)).to(dev)
    A = nn.Linear(args.dim, feat.shape[1]).to(dev)
    opt = torch.optim.Adam(list(E.parameters()) + list(P.parameters())
                           + list(A.parameters()), lr=1e-3)

    def pairwise_cos(w):
        z = F.normalize(w, dim=-1)
        return z @ z.T

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(lab), (args.batch,), device=dev)
        y = lab[idx]
        z = F.normalize(P(E(y)), dim=-1)
        logits = z @ z.T / args.temp
        eye = torch.eye(args.batch, dtype=torch.bool, device=dev)
        mask_pos = (y[:, None] == y[None, :]).float() * (~eye).float()
        n_pos = mask_pos.sum(1).clamp_min(1)
        log_prob = logits - torch.logsumexp(
            logits.masked_fill(eye[:, :], -float("inf")), dim=1, keepdim=True)
        loss_sup = -(log_prob * mask_pos).sum(1).div(n_pos).mean()
        loss_anc = (1 - F.cosine_similarity(A(E(y)), C[y], dim=-1)).mean()
        loss = loss_sup + args.w_anchor * loss_anc
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == 1:
            with torch.no_grad():
                w = E.weight
                pc = pairwise_cos(w)
                seen_t = torch.tensor(seen_labels, device=dev)
                off = pc[seen_t][:, seen_t]
                off = off[~torch.eye(len(seen_labels), dtype=torch.bool, device=dev)]
                ctx = F.cosine_similarity(A(E.weight)[seen_t], C[seen_t], dim=-1)
            print(f"step {step}: sup={loss_sup:.4f} anchor={loss_anc:.4f} "
                  f"|pair cos| mean={off.abs().mean():.4f} "
                  f"cos-anchor={ctx.mean():.4f}", flush=True)

    w = E.weight.detach().cpu()
    torch.save({"embedding": w, "raw_ids": sorted(cmap.keys())}, args.out)
    print(f"[ok] -> {args.out}  shape={tuple(w.shape)}")


if __name__ == "__main__":
    main()
