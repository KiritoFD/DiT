#!/usr/bin/env python
"""用 SupCon 重新预训练 v15 的「书家x书体」多风格表 (87, 1536)。

## 为什么需要这个
实测（2026-09-21）:
  - DINO CLS 特征的 pair 质心间余弦 = **0.956** -> DINO **几乎不区分书家**，
    它捕捉的是"这是一张书法图"而不是"这是谁的风格"。
  - v15 原来的表（tools/build_multistyle_k4.py, DINO + K-Means）
    行间余弦 = **0.860** -> 87 个 pair 的向量几乎共线 -> 模型无法区分书家
    -> v15a strict 0.5699 ≈ v13 base 0.5703（多风格零增益）
    -> v16 few-shot diff 卡在 0.345 降不下去、grad-norm 只有 0.008

## 为什么 SupCon 能救
SupCon 的 InfoNCE **只用 pair 标签**（同 pair 近 / 异 pair 远），**不依赖 DINO**。
v13 的单向量表就是这么训出来的，行间余弦 **0.020**（近似正交）。
所以本脚本:
  - 标签 = pair_id（87 个），不是书家（45 个）
  - **--w-anchor 0**（默认）: 去掉 DINO 质心锚定 —— 既然 DINO 没区分度，
    锚定只会把表往"无区分"拉
  - 输出 (87, 1536)，并 reshape 出 centroids (87, 4, 384) 保持与原表同构

用法:
  python tools/pretrain_multistyle_supcon.py \
      --npz assets/dino_cls_50k.npz \
      --pair-map assets/callig_script_id_map.json \
      --out assets/multistyle_k4_supcon.pt \
      --dim 1536 --k 4 --steps 5000 --w-anchor 0.0
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.chdir("/root/Workspace/xy/DiT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="assets/dino_cls_50k.npz")
    ap.add_argument("--pair-map", default="assets/callig_script_id_map.json")
    ap.add_argument("--out", default="assets/multistyle_k4_supcon.pt")
    ap.add_argument("--dim", type=int, default=1536, help="= k * 384")
    ap.add_argument("--k", type=int, default=4, help="子风格数，用于 reshape centroids")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--w-anchor", type=float, default=0.0,
                    help="DINO 质心锚定权重。**默认 0 = 关闭**，"
                         "因为实测 DINO pair 质心余弦 0.956，锚定只会帮倒忙")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z = np.load(a.npz)
    feat = torch.from_numpy(z["feat"].astype(np.float32)).to(dev)
    calligs = z["calligs"] if "calligs" in z.files else None
    scripts = z["scripts"] if "scripts" in z.files else None
    print(f"[supcon] feat {tuple(feat.shape)}  calligs={calligs is not None} "
          f"scripts={scripts is not None}")

    pm = json.load(open(a.pair_map, encoding="utf-8"))["pair_map"]
    meta = json.load(open(a.pair_map, encoding="utf-8"))
    n_pairs = int(meta["num_pairs"])
    print(f"[supcon] pair_map {len(pm)} 条, num_pairs={n_pairs}")

    if calligs is None or scripts is None:
        raise SystemExit("npz 里没有 calligs/scripts，无法构造 pair 标签")
    cl = calligs.tolist() if hasattr(calligs, "tolist") else list(calligs)
    sc = scripts.tolist() if hasattr(scripts, "tolist") else list(scripts)
    keys = [f"{c}:{s}" for c, s in zip(cl, sc)]
    lab = torch.tensor([pm.get(k, -1) for k in keys], device=dev)
    ok = lab >= 0
    print(f"[supcon] 能映射为 pair 的样本: {int(ok.sum())}/{len(lab)}")
    feat = feat[ok]
    lab = lab[ok]

    E = nn.Embedding(n_pairs, a.dim).to(dev)
    nn.init.normal_(E.weight, std=1.0)
    P = nn.Sequential(nn.Linear(a.dim, a.dim, bias=False), nn.GELU(),
                      nn.Linear(a.dim, a.dim, bias=False)).to(dev)
    params = list(E.parameters()) + list(P.parameters())
    A = None
    C = None
    if a.w_anchor > 0:
        fn = F.normalize(feat, dim=-1)
        C = torch.stack([fn[lab == i].mean(0) for i in range(n_pairs)])
        C = F.normalize(C, dim=-1)
        A = nn.Linear(a.dim, feat.shape[1]).to(dev)
        params += list(A.parameters())
    opt = torch.optim.Adam(params, lr=a.lr)

    for step in range(1, a.steps + 1):
        idx = torch.randint(0, len(lab), (a.batch,), device=dev)
        y = lab[idx]
        zz = F.normalize(P(E(y)), dim=-1)
        logits = zz @ zz.T / a.temp
        eye = torch.eye(a.batch, dtype=torch.bool, device=dev)
        mask_pos = (y[:, None] == y[None, :]).float() * (~eye).float()
        n_pos = mask_pos.sum(1).clamp_min(1)
        log_prob = logits - torch.logsumexp(
            logits.masked_fill(eye, -float("inf")), dim=1, keepdim=True)
        loss_sup = -(log_prob * mask_pos).sum(1).div(n_pos).mean()
        loss = loss_sup
        if a.w_anchor > 0:
            loss = loss + a.w_anchor * (
                1 - F.cosine_similarity(A(E(y)), C[y], dim=-1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == 1:
            with torch.no_grad():
                w = F.normalize(E.weight, dim=-1)
                off = (w @ w.T)[~torch.eye(n_pairs, dtype=bool, device=dev)]
            print(f"  step {step}: sup={loss_sup:.4f} "
                  f"|pair cos| mean={off.abs().mean():.4f} "
                  f"max={off.max():.4f}", flush=True)

    w = E.weight.detach().cpu()
    wn = F.normalize(w, dim=-1)
    off = (wn @ wn.T)[~torch.eye(n_pairs, dtype=bool)]
    print(f"\n[最终] {tuple(w.shape)} pairwise cos: mean={off.mean():.4f} "
          f"|mean|={off.abs().mean():.4f} max={off.max():.4f} min={off.min():.4f}")
    print("  (v15 原表基线 mean=0.860 <- 几乎共线；v13 SupCon 表 0.020)")

    sub = a.dim // a.k
    torch.save({
        "embedding": w,
        "centroids": w.reshape(n_pairs, a.k, sub),
        "pair_mean": w.reshape(n_pairs, a.k, sub).mean(1),
        "pair_to_callig": meta.get("pair_to_callig"),
        "n_pairs": n_pairs,
        "k_clusters": a.k,
        "dim": sub,
        "map_path": a.pair_map,
        "npz_path": a.npz,
        "source": "supcon",
    }, a.out)
    print(f"[ok] -> {a.out}")


if __name__ == "__main__":
    main()
