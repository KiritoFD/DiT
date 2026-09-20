# -*- coding: utf-8 -*-
"""pretrain_callig_script_emb.py — (书家×书体) 联合风格表的层级 SupCon 预训练 (Stage 1).

## 目标
产出 `assets/callig_script_emb_pretrained.pt` = (87,128) 冻结风格表, 供 Stage 2/3 用。
沿用 pretrain_callig_emb.py 的成功配方(SupCon InfoNCE + DINO 质心锚定 -> 防塌缩),
两处升级针对"把一个人不同书体分开、但保留书家身份":

  1) **层级加权正对**(hierarchical SupCon):
       w(i,j) = 1.0           同 (书家,书体)         -> 强拉近(同一风格模态)
              = w_sibling     同书家、异书体          -> 弱拉近(共享"这个人"的身份)
              = 0             异书家                  -> 负对(推远)
     几何上: 每个书家成一簇, 簇内按书体分成子簇 -> 既解耦书体又不丢身份。

  2) **稀疏对质心回退**: 样本数 < min_samples 的 (书家,书体) 对(min=1 也存在),
     其 DINO 质心噪声极大 -> 锚定目标改用**书家级质心**(该书家全部图的均值)。
     这样稀疏对起步于"这个人的平均风格", 有数据时 SupCon 再把它推到书体特化位置。

## 输入
  --dino-npz   assets/dino_cls_50k.npz  (extract_dino_cls_50k.py 产出: feat/calligs/scripts)
  --map        assets/callig_script_id_map.json  (callig_script_map.py 产出)

## 用法(远端 GPU, 一次性)
  python tools/pretrain_callig_script_emb.py \
      --dino-npz assets/dino_cls_50k.npz \
      --map assets/callig_script_id_map.json \
      --out assets/callig_script_emb_pretrained.pt \
      --w-sibling 0.3 --steps 3000
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
    ap.add_argument("--dino-npz", default="assets/dino_cls_50k.npz")
    ap.add_argument("--map", dest="map_json", default="assets/callig_script_id_map.json")
    ap.add_argument("--out", default="assets/callig_script_emb_pretrained.pt")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--w-anchor", type=float, default=0.5)
    ap.add_argument("--w-sibling", type=float, default=0.3,
                    help="同书家异书体的弱正对权重(共享身份); 0=退化为纯 pair SupCon")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device("cuda")

    M = json.load(open(args.map_json, encoding="utf-8"))
    pair_map = M["pair_map"]                      # "c:s" -> pair_id
    callig_map = {int(k): v for k, v in M["callig_map"].items()}
    pair_to_callig = M["pair_to_callig"]          # pair_id -> callig_idx
    min_samples = int(M.get("min_samples", 50))
    pair_counts = M["pair_counts"]
    n_pairs = int(M["num_pairs"])
    n_callig = int(M["num_calligraphers"])

    d = np.load(args.dino_npz)
    feat_all = d["feat"].astype(np.float32)
    calligs = np.array([int(x) for x in d["calligs"].tolist()])
    scripts = np.array([int(x) for x in d["scripts"].tolist()])

    # 每行 -> (pair_id, callig_idx); 丢弃词表里没有的行
    pair_lab, callig_lab, keep = [], [], []
    for i in range(len(calligs)):
        k = f"{calligs[i]}:{scripts[i]}"
        if k in pair_map and calligs[i] in callig_map:
            pair_lab.append(pair_map[k])
            callig_lab.append(callig_map[calligs[i]])
            keep.append(i)
    keep = np.array(keep)
    feat_all = feat_all[keep]
    pair_lab = np.array(pair_lab, dtype=np.int64)
    callig_lab = np.array(callig_lab, dtype=np.int64)
    print(f"[data] DINO feats {feat_all.shape} | pairs={n_pairs} callig={n_callig} "
          f"| 覆盖样本 {len(pair_lab)}", flush=True)
    assert set(pair_lab.tolist()) <= set(range(n_pairs)), "pair 标签越界"

    feat = torch.from_numpy(feat_all).to(dev)
    yp = torch.from_numpy(pair_lab).to(dev)
    yc = torch.from_numpy(callig_lab).to(dev)
    feat_n = F.normalize(feat, dim=-1)

    # ── 质心: 每 pair 一个; 稀疏 pair 回退到书家级质心 ──────────────────────
    C_callig = torch.zeros(n_callig, feat.shape[1], device=dev)
    for c in range(n_callig):
        m = (yc == c)
        if m.any():
            C_callig[c] = feat_n[m].mean(0)
    C = torch.zeros(n_pairs, feat.shape[1], device=dev)
    n_backoff = 0
    for pid in range(n_pairs):
        key = next(k for k, v in pair_map.items() if v == pid)
        cnt = int(pair_counts.get(key, 0))
        m = (yp == pid)
        if cnt >= min_samples and m.any():
            C[pid] = feat_n[m].mean(0)
        else:
            # 稀疏回退: 用书家级质心(该书家全部书的平均风格)
            C[pid] = C_callig[pair_to_callig[pid]]
            n_backoff += 1
    C = F.normalize(C, dim=-1)
    print(f"[centroid] {n_pairs} 对质心, 其中 {n_backoff} 对稀疏(<{min_samples})回退到书家级",
          flush=True)

    # ── 模型: E(待训表) + P(投影头,SupCon用) + A(到DINO维,锚定用); P/A 训后丢弃 ──
    E = nn.Embedding(n_pairs, args.dim).to(dev)
    nn.init.normal_(E.weight, std=1.0)   # std=1.0: 0.02 会让 P 的 bias 主导 -> 塌缩/NaN
    P = nn.Sequential(nn.Linear(args.dim, args.dim, bias=False), nn.GELU(),
                      nn.Linear(args.dim, args.dim, bias=False)).to(dev)
    A = nn.Linear(args.dim, feat.shape[1]).to(dev)
    opt = torch.optim.Adam(list(E.parameters()) + list(P.parameters())
                           + list(A.parameters()), lr=1e-3)

    B = args.batch
    eye = torch.eye(B, dtype=torch.bool, device=dev)
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(yp), (B,), device=dev)
        y_pair, y_callig = yp[idx], yc[idx]
        z = F.normalize(P(E(y_pair)), dim=-1)
        logits = z @ z.T / args.temp

        # 层级加权正对矩阵 W (对角置 0)
        same_pair = (y_pair[:, None] == y_pair[None, :]).float()
        same_callig = (y_callig[:, None] == y_callig[None, :]).float()
        W = same_pair + args.w_sibling * (same_callig * (1.0 - same_pair))
        W = W * (~eye).float()

        log_prob = logits - torch.logsumexp(
            logits.masked_fill(eye, -float("inf")), dim=1, keepdim=True)
        denom = W.sum(1).clamp_min(1e-6)
        loss_sup = -(log_prob * W).sum(1).div(denom).mean()

        loss_anc = (1 - F.cosine_similarity(A(E(y_pair)), C[y_pair], dim=-1)).mean()
        loss = loss_sup + args.w_anchor * loss_anc
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 500 == 0 or step == 1:
            with torch.no_grad():
                w = F.normalize(E.weight, dim=-1)
                pc = w @ w.T
                off = pc[~torch.eye(n_pairs, dtype=torch.bool, device=dev)]
                # 同书家不同书体之间的 cos(应中等: 共享身份但已分开)
                sib = ((torch.tensor(pair_to_callig, device=dev)[:, None]
                        == torch.tensor(pair_to_callig, device=dev)[None, :])
                       & ~torch.eye(n_pairs, dtype=torch.bool, device=dev))
                ctx = F.cosine_similarity(A(E.weight), C, dim=-1)
            print(f"step {step}: sup={loss_sup:.4f} anc={loss_anc:.4f} "
                  f"|全对 cos|mean|={off.abs().mean():.4f} "
                  f"同书家异书体 cos={pc[sib].mean():.4f} "
                  f"cos-anchor={ctx.mean():.4f}", flush=True)

    w = E.weight.detach().cpu()
    torch.save({"embedding": w,
                "pair_to_callig": pair_to_callig,
                "n_pairs": n_pairs, "dim": args.dim},
               args.out)
    # 最终塌缩体检
    wn = F.normalize(w, dim=-1)
    pc = wn @ wn.T
    off = pc[~torch.eye(n_pairs, dtype=torch.bool)]
    print(f"\n[最终] {n_pairs} 表 pairwise cos: mean={off.mean():.4f} "
          f"|mean|={off.abs().mean():.4f} max={off.max():.4f} (塌缩基线 0.323)")
    print(f"[ok] -> {args.out}  shape={tuple(w.shape)}")


if __name__ == "__main__":
    main()
