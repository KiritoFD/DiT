# -*- coding: utf-8 -*-
"""bench_tables_big.py — 给大表训练的每个阶段标定 batch (目标: 峰值约 20G, 不 OOM)。

对 (维度, k, 每步类数) 的若干候选各跑 N 步, 报 it/s + 峰值显存 + loss 下降, 选出:
  - 峰值 <= 20.5G  (24G 卡留余量)
  - it/s 最大

用法:
  PYTHONPATH=. python tools/bench_tables_big.py --source dino
  PYTHONPATH=. python tools/bench_tables_big.py --source rgb --rgb-rows 40000
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")

from tools.train_tables_big import (CSV_RAW, NPZ_RAW, Enc,  # noqa: E402
                                    repel_penalty, supcon_rows_vs_feat)


def affine(x, dev, rot_deg=4.0, scale=0.05, shift=3.0):
    """几何仿射增强 (与 train_tables_big.feats_from_rgb 内联版同参)。"""
    N = x.size(0)
    ang = (torch.rand(N, device=dev) * 2 - 1) * (rot_deg * np.pi / 180.0)
    sc = 1.0 + (torch.rand(N, device=dev) * 2 - 1) * scale
    tx = (torch.rand(N, device=dev) * 2 - 1) * (shift / x.size(2)) * 2
    ty = (torch.rand(N, device=dev) * 2 - 1) * (shift / x.size(3)) * 2
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([torch.stack([cos * sc, -sin * sc, tx], -1),
                         torch.stack([sin * sc, cos * sc, ty], -1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="border")


def bench_head(feat, lab, n_cls, dim, k, c_per_step, *, name, N=40, lr=1.5e-3,
               mode="char", debias=None):
    dev = feat.device
    if debias is not None:
        cb = int(debias.max()) + 1
        mean = torch.zeros(cb, feat.size(1), device=dev)
        cnt = torch.zeros(cb, 1, device=dev)
        mean.index_add_(0, debias, feat)
        cnt.index_add_(0, debias, torch.ones(lab.size(0), 1, device=dev))
        feat = F.normalize(feat - (mean / cnt.clamp_min(1))[debias], dim=-1)

    elig = np.array(sorted([c for c in range(n_cls)
                            if int((lab == c).sum()) >= k]), dtype=np.int64)
    by = defaultdict(list)
    for i, g in enumerate(lab.tolist()):
        by[int(g)].append(i)
    n_pick = min(c_per_step, len(elig))

    W = nn.Embedding(n_cls, dim).to(dev)
    with torch.no_grad():
        g0 = torch.randn(max(dim, n_cls), dim, device=dev)
        q, _ = torch.linalg.qr(g0)
        W.weight.copy_(q[:n_cls])
    P = nn.Sequential(nn.Linear(feat.size(1), 256), nn.GELU(), nn.Linear(256, dim)).to(dev)
    opt = torch.optim.AdamW(list(W.parameters()) + list(P.parameters()), lr=lr,
                            weight_decay=1e-4)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    l0, l1 = None, None
    t0 = time.time()
    for step in range(1, N + 1):
        sel = np.random.choice(elig, size=n_pick, replace=False)
        idxs = []
        for g in sel:
            v = by[int(g)]
            idxs += list(v) if len(v) <= k else list(np.random.choice(v, size=k, replace=False))
        idx = torch.tensor(idxs, device=dev, dtype=torch.long)
        y = lab[idx]
        zA = F.normalize(W(y), dim=-1)
        zB = F.normalize(P(feat[idx]), dim=-1)
        loss = supcon_rows_vs_feat(zA, zB, y)
        Wn = F.normalize(W.weight, dim=-1)
        if mode == "script":
            loss = loss + 10.0 * ((Wn @ Wn.t())[~torch.eye(n_cls, dtype=torch.bool, device=dev)] ** 2).mean()
        elif mode == "callig":
            loss = loss + repel_penalty(Wn, 0.15, 3.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1:
            l0 = loss.item()
        l1 = loss.item()
    dt = time.time() - t0
    B = n_pick * k
    print(f"  {name:7s} k={k:3d} 类/步={n_pick:5d} -> 样本/步={B:6d} | dim={dim:4d} | "
          f"loss {l0:.3f}->{l1:.3f} (chance {np.log(B):.3f}, Δ={l0-l1:+.3f}) | "
          f"{N/dt:5.1f} it/s | 峰值 {torch.cuda.max_memory_allocated()/1e9:5.2f}G | "
          f"{B*N/dt/1000:5.1f}k 样本/s", flush=True)
    del W, P, opt, loss, zA, zB
    torch.cuda.empty_cache()


def bench_rgb(rows, lab, n_cls, k, n_pick, dim=256, res=96, N=20, lr=1e-3,
              limit_imgs=40000):
    import cv2
    dev = "cuda"
    sub = min(limit_imgs, len(rows))
    imgs = np.zeros((sub, res, res), np.uint8)
    for i in range(sub):
        p = rows[i]["image_path"]
        g = cv2.imread(p if os.path.isabs(p) else os.path.join(RAW_ROOT, p),
                       cv2.IMREAD_GRAYSCALE)
        if g is not None:
            imgs[i] = g if g.shape[0] == res else cv2.resize(g, (res, res),
                                                             interpolation=cv2.INTER_AREA)
    lab_sub = lab[:sub]
    elig = np.array(sorted([c for c in range(n_cls)
                            if int((lab_sub == c).sum()) >= k]), dtype=np.int64)
    by = defaultdict(list)
    for i, g in enumerate(lab_sub.tolist()):
        by[int(g)].append(i)
    n_pick = min(n_pick, len(elig))
    net = Enc(dim).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    l0 = l1 = None
    t0 = time.time()
    for step in range(1, N + 1):
        sel = np.random.choice(elig, size=n_pick, replace=False)
        idxs = []
        for g in sel:
            v = by[int(g)]
            idxs += list(v) if len(v) <= k else list(np.random.choice(v, size=k, replace=False))
        x = torch.from_numpy(imgs[idxs].astype(np.float32) / 255.0).unsqueeze(1).to(dev)
        y = torch.tensor([lab_sub[i] for i in idxs], dtype=torch.long, device=dev)
        with torch.cuda.amp.autocast(enabled=True):
            zn = F.normalize(net(affine(x, dev)), dim=-1)
            logits = (zn @ zn.t()) / 0.07
            tgt = (y[:, None] == y[None, :]).float()
            lp1, lp0 = logits.log_softmax(1), logits.log_softmax(0)
            loss = 0.5 * (-(lp1 * tgt).sum(1) / tgt.sum(1).clamp_min(1)).mean() \
                + 0.5 * (-(lp0 * tgt).sum(0) / tgt.sum(0).clamp_min(1)).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        if step == 1:
            l0 = loss.item()
        l1 = loss.item()
    dt = time.time() - t0
    B = n_pick * k
    print(f"  rgb     k={k:3d} 类/步={n_pick:5d} -> 图/步={B:6d} | "
          f"loss {l0:.3f}->{l1:.3f} (chance {np.log(B):.3f}, Δ={l0-l1:+.3f}) | "
          f"{N/dt:5.1f} it/s | 峰值 {torch.cuda.max_memory_allocated()/1e9:5.2f}G | "
          f"{B*N/dt:5.0f} 图/s", flush=True)
    del net, opt
    torch.cuda.empty_cache()


RAW_ROOT = "/root/Workspace/xy/UNIFIED_RAW"


def _guard(fn, *args, **kw):
    """单档 OOM 不打断整轮标定。"""
    try:
        fn(*args, **kw)
    except torch.cuda.OutOfMemoryError:
        print(f"  ---- OOM (跳过): k={kw.get('k')} 类/步={kw.get('c_per_step') or (args and '')}"
              " ----", flush=True)
        torch.cuda.empty_cache()
        return None
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  ---- OOM (跳过): {e} ----", flush=True)
            torch.cuda.empty_cache()
            return None
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["dino", "rgb"], required=True)
    ap.add_argument("--limit", type=int, default=150000)
    ap.add_argument("--steps", type=int, default=40)
    a = ap.parse_args()
    dev = "cuda"

    rows = list(csv.DictReader(open(CSV_RAW, encoding="utf-8")))[:a.limit]
    n_scr = max(int(r["script_id"]) for r in rows) + 1
    n_cal = max(int(r["calligrapher_id"]) for r in rows) + 1
    n_chr = max(int(r["character_id"]) for r in rows) + 1
    lab_s = torch.tensor([int(r["script_id"]) for r in rows], device=dev)
    lab_g = torch.tensor([int(r["calligrapher_id"]) for r in rows], device=dev)
    lab_c = torch.tensor([int(r["character_id"]) for r in rows], device=dev)
    print(f"[bench {a.source}] {len(rows):,} 行 script={n_scr} callig={n_cal} char={n_chr}",
          flush=True)

    if a.source == "dino":
        print("\n=== 书体头 dim=32 (12 类, 全类覆盖) ===")
        z = np.load(NPZ_RAW)
        ids, feat_np = z["ids"], z["feat"]
        pos = {int(i): k for k, i in enumerate(ids)}
        order = np.array([pos[int(r["img_id"])] for r in rows], dtype=np.int64)
        F_all = torch.from_numpy(feat_np[order]).to(dev)
        for c in (12, 64, 256, 1024):
            _guard(bench_head, F_all, lab_s, n_scr, 32, 32, c, name="script",
                   N=a.steps, lr=3e-3, mode="script", debias=lab_c)
        print("\n=== 书家头 dim=128 ===")
        for k, c in ((4, 256), (4, 1024), (8, 1024), (16, 1024), (16, 2048), (32, 2048)):
            _guard(bench_head, F_all, lab_g, n_cal, 128, k, c, name="callig",
                   N=a.steps, lr=2e-3, mode="callig")
        print("\n=== 汉字头 dim=384 ===")
        for k, c in ((4, 1024), (8, 1024), (16, 1024), (16, 2048), (32, 1024), (32, 2048)):
            _guard(bench_head, F_all, lab_c, n_chr, 384, k, c, name="char",
                   N=a.steps, lr=1.5e-3, mode="char")
    else:
        print("\n=== rgb 编码器 (独立, 每头一个) ===")
        for k, c in ((4, 256), (4, 512), (8, 512), (8, 1024), (16, 1024), (16, 2048)):
            _guard(bench_rgb, rows, lab_c, n_chr, k, c, dim=256, N=a.steps)
    print("\n[done]")


if __name__ == "__main__":
    main()
