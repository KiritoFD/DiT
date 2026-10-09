# -*- coding: utf-8 -*-
"""judge_callig_table.py — 用**与 judge_callig_rgb.py 完全相同的划分与口径**,
判定任意一个 (编码器 + 质心表) 的书家表在 top10 holdout 上的 per-sample top-1。

这样 0.5420(基线) / 0.2933(A) / 0.2027(B) / 新表 全部可直接对比。

用法:
  PYTHONPATH=. python tools/judge_callig_table.py --enc-dir assets/triple_tables_top10_rgb_v2 \
      --res 128 --dim 32 --tag v2_callig
"""
import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(os.environ.get("DIT_ROOT", "/root/Workspace/xy/DiT"))
DEV = "cuda"


class Enc(nn.Module):
    def __init__(self, dim, deep=0):
        super().__init__()

        def blk(i, o, s=2):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False),
                                 nn.BatchNorm2d(o), nn.SiLU())
        chans = [32, 64, 128, 256] + ([384] if deep else [])
        layers, i = [], 1
        for o in chans:
            layers.append(blk(i, o))
            i = o
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                   nn.Linear(i, dim, bias=False), nn.BatchNorm1d(dim)]
        self.enc = nn.Sequential(*layers)

    def forward(self, x):
        return self.enc(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="exp-std/csv/train.csv")
    ap.add_argument("--enc-dir", required=True)
    ap.add_argument("--enc-name", default="encoder_callig.pt")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--deep", type=int, default=1)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    ids = sorted({int(r["calligrapher_id"]) for r in rows})
    idx = {v: i for i, v in enumerate(ids)}
    lab = np.array([idx[int(r["calligrapher_id"])] for r in rows], np.int64)
    n_cls = len(ids)

    # ★ 与 judge_callig_rgb.py 逐字相同的划分
    rng = np.random.RandomState(a.seed)
    tr, ho = [], []
    for c in range(n_cls):
        m = np.where(lab == c)[0]
        rng.shuffle(m)
        n_ho = int(len(m) * a.holdout)
        ho += list(m[:n_ho])
        tr += list(m[n_ho:])
    tr, ho = np.sort(np.array(tr)), np.sort(np.array(ho))

    imgs = np.zeros((len(rows), a.res, a.res), np.uint8)

    def rd(i):
        cv2.setNumThreads(1)
        g = cv2.imread(rows[i]["image_path"], cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise RuntimeError(f"读图失败: {rows[i]['image_path']}")
        if g.shape[0] != a.res:
            g = cv2.resize(g, (a.res, a.res), interpolation=cv2.INTER_AREA)
        return i, g

    with ThreadPoolExecutor(a.workers) as pool:
        for i, g in pool.map(rd, range(len(rows))):
            imgs[i] = g
    print(f"[data] {len(rows):,} 张, 训练 {len(tr):,} / holdout {len(ho):,}", flush=True)

    p = os.path.join(a.enc_dir, a.enc_name)
    net = Enc(a.dim, a.deep).to(DEV).to(memory_format=torch.channels_last)
    net.load_state_dict(torch.load(p, map_location=DEV))
    net.eval()
    print(f"[enc] 已加载 {p}", flush=True)

    @torch.no_grad()
    def feats(s):
        out = []
        for b in range(0, len(s), 2048):
            x = torch.from_numpy(imgs[s[b:b + 2048]]).unsqueeze(1).to(DEV) \
                .float().div_(255.0)
            out.append(F.normalize(net(x).float(), dim=-1))
        return torch.cat(out)

    Ztr, Zho = feats(tr), feats(ho)
    gm = Ztr.mean(0)
    T = torch.zeros(n_cls, a.dim, device=DEV)
    for c in range(n_cls):
        m = torch.from_numpy(lab[tr] == c).to(DEV)
        T[c] = F.normalize(Ztr[m].mean(0) - gm, dim=-1)
    pr = ((Zho - gm) @ T.t()).argmax(1).cpu().numpy()
    y = lab[ho]
    acc = float((pr == y).mean())
    se = float(np.sqrt(acc * (1 - acc) / len(y)))
    rec = {int(c): float((pr[y == c] == c).mean()) for c in range(n_cls)}
    with torch.no_grad():
        cos = T @ T.t()
        off = ~torch.eye(n_cls, dtype=torch.bool, device=DEV)
        mc = float(cos[off].abs().mean())
    base = 0.5420
    print("\n" + "=" * 70)
    print(f"[{a.tag}] enc={a.enc_dir}/{a.enc_name}  res={a.res} dim={a.dim} deep={a.deep}")
    print(f"  holdout per-sample top-1 = {acc:.4f} ± {1.96*se:.4f} (n={len(y):,}, "
          f"随机 {1/n_cls:.2f})")
    print(f"  基线 (v1, top10 训练)     = {base:.4f} ± 0.0135   Δ = {acc-base:+.4f}")
    print(f"  每类召回: " + " ".join(f"c{c}={rec[c]:.3f}" for c in range(n_cls)))
    print(f"  表行|cos|均值 = {mc:.4f}")
    print("=" * 70, flush=True)
    json.dump(dict(tag=a.tag, enc_dir=a.enc_dir, acc=acc, ci95=1.96 * se,
                   n_holdout=int(len(y)), baseline=base, delta=acc - base,
                   per_class_recall=rec, mean_abs_cos=mc, res=a.res, dim=a.dim,
                   deep=a.deep),
              open(f"_sync_work/judge_{a.tag or 'table'}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
