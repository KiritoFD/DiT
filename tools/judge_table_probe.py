# -*- coding: utf-8 -*-
"""judge_table_probe.py — **表质量同协议对比**: 冻结表行, 各自拟合一个新投影头,
在 top10 DINO 特征的留出集上算近邻 top-1。谁高, 谁的表把特征空间组织得更好。

对比对象: 基线 (triple_tables_best_minimal, v68 现役) vs 候选 (如 big_dino_top10 切片)。
注意一个不可消除的不对称: 基线表是在**全量 26,002 个 top10 特征**上拟合的 (含留出集),
候选表 (raw 切片) 从未见过 top10 样本 -> 该协议**偏向基线**; 候选若仍更高则结论更强。

用法:
  PYTHONPATH=. python tools/judge_table_probe.py \
      --cand assets/triple_tables_big_dino_top10 --tag bigdino
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(os.environ.get("DIT_ROOT", "/root/Workspace/xy/DiT"))
DEV = "cuda"


class P(nn.Module):
    def __init__(self, d_in, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 256), nn.GELU(), nn.Linear(256, dim))

    def forward(self, x):
        return self.net(x)


def load_data(csv_path, ids_path, npz_path):
    ids = np.load(ids_path)
    z = np.load(npz_path)
    feat = z["feat"].astype(np.float32)
    by_img = {int(r["img_id"]): r for r in csv.DictReader(open(csv_path, encoding="utf-8"))}
    rows = [by_img[int(i)] for i in ids]
    return torch.from_numpy(feat), rows


def probe_one(tag, nm, tab_dir, feat, rows, labels, classes, holdout_idx, train_idx,
              steps=2000, batch=4096, lr=1e-3, temp=0.07):
    """冻结表行 T, 拟合投影头, 返回 holdout top-1 与行几何。"""
    dev = DEV
    T = torch.from_numpy(np.load(f"{tab_dir}/{nm}_table.npy")).to(dev).float()
    Tn = F.normalize(T, dim=-1)
    idx = {c: i for i, c in enumerate(classes)}
    y_all = torch.tensor([idx.get(c, -1) for c in labels], device=dev)
    keep = y_all >= 0
    y, f = y_all[keep], feat[keep.cpu().numpy()].to(dev)
    tr = torch.from_numpy(train_idx[keep.cpu().numpy()[train_idx]]).to(dev)
    ho = torch.from_numpy(holdout_idx[keep.cpu().numpy()[holdout_idx]]).to(dev)

    head = P(f.size(1), T.size(1)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.RandomState(0)
    n = tr.numel()
    t0 = time.time()
    for step in range(1, steps + 1):
        idx_b = tr[torch.from_numpy(rng.choice(n, size=min(batch, n), replace=False)).to(dev)]
        zA = Tn[y[idx_b]]
        zB = F.normalize(head(f[idx_b]), dim=-1)
        logits = (zA @ zB.t()) / temp
        yy = y[idx_b]
        tgt = (yy[:, None] == yy[None, :]).float()
        loss = 0.5 * (-(logits.log_softmax(1) * tgt).sum(1) / tgt.sum(1).clamp_min(1)).mean() \
            + 0.5 * (-(logits.log_softmax(0) * tgt).sum(0) / tgt.sum(0).clamp_min(1)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pr = (F.normalize(head(f[ho]), dim=-1) @ Tn.t()).argmax(1)
        acc = float((pr == y[ho]).float().mean())
        # 行几何放 CPU 算 (GPU 被下游训练占着, 4690x4690 会 OOM)
        Tc = Tn.detach().cpu()
        cos = Tc @ Tc.t()
        off = ~torch.eye(Tc.size(0), dtype=torch.bool)
        mc = float(cos[off].abs().mean())
    print(f"  [{tag}/{nm:7s}] holdout top-1 = {acc:.4f} (n={ho.numel():,}, "
          f"随机 1/{Tn.size(0)}) | 表行|cos| = {mc:.4f} | {time.time()-t0:.0f}s", flush=True)
    return dict(acc=acc, n=int(ho.numel()), n_cls=int(Tn.size(0)), mean_abs_cos=mc,
                steps=steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, help="候选表目录")
    ap.add_argument("--base", default="assets/triple_tables_best_minimal")
    ap.add_argument("--npz", default="assets/dino_feat_top10_g.npz")
    ap.add_argument("--ids", default="data/dino_cache/top10_v1/ids.npy")
    ap.add_argument("--csv", default="exp-std/csv/train.csv")
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--only", default="", help="只测指定头 (逗号分隔)")
    a = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    feat, rows = load_data(a.csv, a.ids, a.npz)
    print(f"[data] {feat.shape[0]:,} 样本 (top10 DINO 特征, 384d)", flush=True)
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(rows))
    n_ho = int(len(rows) * a.holdout)
    ho, tr = np.sort(perm[:n_ho]), np.sort(perm[n_ho:])
    print(f"[split] 训练 {len(tr):,} / holdout {len(ho):,} (seed 0)", flush=True)

    labels = {"font": [int(r["script_id"]) for r in rows],
              "callig": [int(r["calligrapher_id"]) for r in rows],
              "char": [r["character"] for r in rows]}
    out = {}
    heads = ("font", "callig", "char")
    if a.only:
        heads = tuple(s.strip() for s in a.only.split(",") if s.strip())
    for nm in heads:
        classes = json.load(open(f"{a.base}/{nm}_index.json", encoding="utf-8"))["classes"]
        print(f"\n[{nm}] 类数 {len(classes)}")
        out[nm] = {}
        out[nm]["base"] = probe_one("base", nm, a.base, feat, rows, labels[nm],
                                    classes, ho, tr, steps=a.steps)
        out[nm]["cand"] = probe_one("cand", nm, a.cand, feat, rows, labels[nm],
                                    classes, ho, tr, steps=a.steps)
        d = out[nm]["cand"]["acc"] - out[nm]["base"]["acc"]
        print(f"  -> Δ(cand-base) = {d:+.4f}  {'候选更好' if d > 0 else '基线更好'}",
              flush=True)

    print("\n" + "=" * 66)
    print(f"{'head':8s} | {'基线 top-1':>10} {'候选 top-1':>10} {'Δ':>8} | "
          f"基线|cos| 候选|cos|")
    print("-" * 66)
    for nm in heads:
        b, c = out[nm]["base"], out[nm]["cand"]
        print(f"{nm:8s} | {b['acc']:10.4f} {c['acc']:10.4f} {c['acc']-b['acc']:+8.4f} | "
              f"{b['mean_abs_cos']:8.4f} {c['mean_abs_cos']:8.4f}")
    print("=" * 66)
    print("注意: 基线表是在全量 26,002 特征上拟合的 (含留出集), 候选 (raw 切片) 从未见过"
          " top10 样本 -> 本协议偏向基线; 候选更高则结论更强。", flush=True)
    json.dump(out, open(f"_sync_work/judge_{a.tag}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
