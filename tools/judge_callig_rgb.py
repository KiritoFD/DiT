# -*- coding: utf-8 -*-
"""judge_callig_rgb.py — 判定 RGB 书家表到底能不能分辨书家。

上一轮的表用"每类留 1 张"做留一, n=10 -> 标准误 ±0.126, 分辨率不够 (0.20 与随机 0.10
无法区分)。这里换成**分层留出 20% + 按样本算 top-1** (~5,200 张, SE ±0.7%), 并加两个
诊断把问题定位到"编码器"还是"质心表":
  A) 表 = 训练集每类的归一化质心 (与正式表同口径) -> holdout top-1
  B) 线性探针 = 在**同一套编码器特征**上训一个线性分类器 -> 若 B 远高于 A,
     说明特征空间可分, 是"质心表"这一形态丢了信息
  C) 每类召回 (10 个) -> 看是否某一类全死
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
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


def affine(x, rot_deg=4.0, scale=0.05, shift=3.0):
    N = x.size(0)
    ang = (torch.rand(N, device=x.device) * 2 - 1) * (rot_deg * np.pi / 180.0)
    sc = 1.0 + (torch.rand(N, device=x.device) * 2 - 1) * scale
    tx = (torch.rand(N, device=x.device) * 2 - 1) * (shift / x.size(2)) * 2
    ty = (torch.rand(N, device=x.device) * 2 - 1) * (shift / x.size(3)) * 2
    cs, sn = torch.cos(ang), torch.sin(ang)
    th = torch.stack([torch.stack([cs * sc, -sn * sc, tx], -1),
                      torch.stack([sn * sc, cs * sc, ty], -1)], 1)
    return F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False),
                         align_corners=False, padding_mode="border")


class Enc(nn.Module):
    def __init__(self, dim, res):
        super().__init__()

        def blk(i, o, s=2):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False),
                                 nn.BatchNorm2d(o), nn.SiLU())
        self.enc = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), blk(128, 256),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                 nn.Linear(256, dim, bias=False), nn.BatchNorm1d(dim))

    def forward(self, x):
        return self.enc(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="exp-std/csv/train.csv")
    ap.add_argument("--res", type=int, default=96)
    ap.add_argument("--deep", type=int, default=0)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--k-pick", type=int, default=1024)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="base", help="本轮配置名 (存 json / 编码器用)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    ids = sorted({int(r["calligrapher_id"]) for r in rows})
    idx = {v: i for i, v in enumerate(ids)}
    lab = np.array([idx[int(r["calligrapher_id"])] for r in rows], np.int64)
    n_cls = len(ids)
    print(f"[data] {len(rows):,} 张, 书家 {n_cls} 类: {ids}", flush=True)

    # 分层留出
    rng = np.random.RandomState(a.seed)
    tr, ho = [], []
    for c in range(n_cls):
        m = np.where(lab == c)[0]
        rng.shuffle(m)
        n_ho = int(len(m) * a.holdout)
        ho += list(m[:n_ho])
        tr += list(m[n_ho:])
    tr, ho = np.sort(np.array(tr)), np.sort(np.array(ho))
    print(f"[split] 训练 {len(tr):,} / holdout {len(ho):,} ({a.holdout:.0%})", flush=True)

    imgs = np.zeros((len(rows), a.res, a.res), np.uint8)

    def rd(i):
        cv2.setNumThreads(1)
        g = cv2.imread(rows[i]["image_path"], cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise RuntimeError(f"读图失败: {rows[i]['image_path']}")
        if g.shape[0] != a.res:
            g = cv2.resize(g, (a.res, a.res), interpolation=cv2.INTER_AREA)
        return i, g

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as pool:
        for n, (i, g) in enumerate(pool.map(rd, range(len(rows)))):
            imgs[i] = g
            if n and n % 10000 == 0:
                print(f"  [preload] {n:,}/{len(rows):,}", flush=True)
    print(f"[preload] {time.time()-t0:.0f}s", flush=True)

    by = defaultdict(list)
    for i in tr:
        by[int(lab[i])].append(i)
    kp = min(a.k_pick, min(len(v) for v in by.values()))
    pad = np.zeros((n_cls, max(len(v) for v in by.values())), np.int64)
    cnt = np.zeros(n_cls, np.int64)
    for c, v in enumerate(by.values()):
        cnt[c] = len(v)
        pad[c, :len(v)] = v
    print(f"[train] 每类每步 {kp} 张, batch {n_cls*kp}, 下界 ln({kp})={np.log(kp):.3f}",
          flush=True)

    net = Enc(a.dim, a.res, deep=a.deep).to(DEV).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps,
                                                       eta_min=a.lr * 0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    rng2 = np.random.RandomState(1)
    for step in range(1, a.steps + 1):
        sel = rng2.choice(n_cls, size=n_cls, replace=False)
        off = (rng2.rand(n_cls, kp) * cnt[sel][:, None]).astype(np.int64)
        ii = pad[sel[:, None], off].ravel()
        x = torch.from_numpy(imgs[ii]).unsqueeze(1).to(DEV) \
            .float().div_(255.0).contiguous(memory_format=torch.channels_last)
        y = torch.from_numpy(lab[ii]).to(DEV)
        with torch.cuda.amp.autocast(enabled=True):
            zn = F.normalize(net(affine(x).float()), dim=-1)
            lg = (zn @ zn.t()) / 0.07
            tgt = (y[:, None] == y[None, :]).float()
            loss = 0.5 * (-(lg.log_softmax(1) * tgt).sum(1) / tgt.sum(1).clamp_min(1)).mean() \
                + 0.5 * (-(lg.log_softmax(0) * tgt).sum(0) / tgt.sum(0).clamp_min(1)).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % a.log_every == 0 or step == 1 or step == a.steps:
            print(f"  [step {step:5d}/{a.steps}] loss {loss.item():.4f} "
                  f"(下界 {np.log(kp):.3f}, 余量 {loss.item()-np.log(kp):+.3f}) | "
                  f"峰值 {torch.cuda.max_memory_allocated()/1e9:.1f}G", flush=True)

    # 特征
    net.eval()

    @torch.no_grad()
    def feats(sel_idx):
        out = []
        for b in range(0, len(sel_idx), 4096):
            s = sel_idx[b:b + 4096]
            x = torch.from_numpy(imgs[s]).unsqueeze(1).to(DEV).float().div_(255.0)
            out.append(net(x).float())
        return torch.cat(out)

    Ztr = feats(tr)
    Zho = feats(ho)
    Ztr_n = F.normalize(Ztr, dim=-1)
    Zho_n = F.normalize(Zho, dim=-1)
    gmean = Ztr_n.mean(0)
    T = torch.zeros(n_cls, a.dim, device=DEV)
    for c in range(n_cls):
        m = torch.from_numpy(lab[tr] == c).to(DEV)
        v = Ztr_n[m].mean(0) - gmean
        T[c] = F.normalize(v, dim=-1)

    # A) 质心表 top-1
    pred = (Zho_n - gmean) @ T.t()
    pr = pred.argmax(1).cpu().numpy()
    yl = lab[ho]
    acc = float((pr == yl).mean())
    se = float(np.sqrt(acc * (1 - acc) / max(1, len(yl))))
    print(f"\n[A] 质心表 holdout per-sample top-1 = {acc:.4f} ± {1.96*se:.4f} (n={len(yl):,}, "
          f"随机 1/{n_cls} = {1/n_cls:.2f})", flush=True)
    rec = []
    for c in range(n_cls):
        m = yl == c
        if m.sum():
            rec.append((c, float((pr[m] == c).mean()), int(m.sum())))
    print("     每类召回: " + "  ".join(f"c{c}={r:.3f}(n{n})" for c, r, n in rec), flush=True)

    # B) 线性探针 (同一套特征)
    lin = nn.Linear(a.dim, n_cls).to(DEV)
    opt2 = torch.optim.Adam(lin.parameters(), lr=1e-2)
    for it in range(600):
        lg2 = lin(Ztr_n)
        l2 = F.cross_entropy(lg2, torch.from_numpy(lab[tr]).to(DEV))
        opt2.zero_grad(set_to_none=True)
        l2.backward()
        opt2.step()
    with torch.no_grad():
        acc2 = float((lin(Zho_n).argmax(1).cpu().numpy() == yl).mean())
    print(f"[B] 同特征上的线性探针 top-1 = {acc2:.4f}  "
          f"(若远高于 A -> 特征可分, 是质心表这一形态丢了信息)", flush=True)

    # C) 表行分离度
    with torch.no_grad():
        cos = T @ T.t()
        off = ~torch.eye(n_cls, dtype=torch.bool, device=DEV)
        mc = float(cos[off].abs().mean())
        mx = float(cos[off].max())
    print(f"[C] 表行 |cos| 均值 = {mc:.4f} 最大 = {mx:.4f}", flush=True)
    torch.save(net.state_dict(), f"_sync_work/callig_enc_{a.tag}.pt")
    torch.save(T.detach().cpu(), f"_sync_work/callig_rows_{a.tag}.pt")
    json.dump(dict(tag=a.tag, acc_centroid=acc, acc_linear=acc2, n_holdout=int(len(yl)),
                   chance=1.0 / n_cls, mean_abs_cos=mc, max_cos=mx,
                   per_class_recall={int(c): r for c, r, _ in rec},
                   steps=a.steps, k_pick=int(kp), res=a.res, deep=a.deep),
              open(f"_sync_work/judge_{a.tag}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
