# -*- coding: utf-8 -*-
"""train_callig_raw_judge.py — 书家编码器改在 **raw 全量 393k / 2,133 书家** 上训,
再用**同一个 top10 留出集**判定: 是否比"只在 top10 26k/10 类上训"更好 (也更能泛化)。

动机 (用户 2026-10-08 提出):
  top10 只有 10 个书家、26k 张 -> 编码器只需区分 10 类, 是"10 分类器"而非风格空间。
  raw 有 2,133 个书家、393k 张 -> 学到的是通用风格表征; 给 v68 用时只需**切出那 10 行**。
  这样还顺带验证"大表 -> 切片"的路线 (10 位书家名在 raw 里全部存在, 已核)。

判定口径与 tools/judge_callig_rgb.py **完全一致** (同 seed / 同 20% 分层留出),
所以可以直接和 top10-训练的 0.5420 ± 0.0135 对比。

输出:
  <out>/callig_table.npy (10x32, 行序 = top10 排序 id) + index/remap  (可直接换进 v68)
  <out>/encoder_callig_raw.pt
  _sync_work/judge_callig_raw.json  (三组数字)
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
RAW = "/root/Workspace/xy/UNIFIED_RAW"
CSV_RAW = f"{RAW}/meta/train_clean.csv"
CSV_TOP10 = "exp-std/csv/train.csv"


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
    ap.add_argument("--res", type=int, default=96)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--deep", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--k-pick", type=int, default=4, help="raw 2133 类, 每类每步几张")
    ap.add_argument("--holdout", type=float, default=0.2, help="top10 判定用的留出比例")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--allow-missing", type=int, default=0,
                    help="冒烟用: 允许 top10 名字在 (被截断的) raw 里缺失")
    ap.add_argument("--out", default="assets/triple_tables_top10_rgb_v2")
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    os.makedirs(a.out, exist_ok=True)

    # ── raw 语料 ──
    raw = list(csv.DictReader(open(CSV_RAW, encoding="utf-8")))
    if a.limit:
        raw = raw[:a.limit]
    n_raw = len(raw)
    gid = np.array([int(r["calligrapher_id"]) for r in raw], np.int64)
    n_cls = int(gid.max()) + 1
    raw_names = {int(r["calligrapher_id"]): r["calligrapher"] for r in raw}
    paths = [os.path.join(RAW, r["image_path"]) for r in raw]
    cnt_all = np.bincount(gid, minlength=n_cls)
    print(f"[raw] {n_raw:,} 张, 书家 {n_cls} 类 "
          f"(每类中位 {int(np.median(cnt_all[cnt_all > 0]))} 张)", flush=True)

    # ── top10 (只需其中的书家名 + 划分) ──
    top = list(csv.DictReader(open(CSV_TOP10, encoding="utf-8")))
    t_ids = sorted({int(r["calligrapher_id"]) for r in top})
    t_idx = {v: i for i, v in enumerate(t_ids)}
    t_lab = np.array([t_idx[int(r["calligrapher_id"])] for r in top], np.int64)
    name2raw = {}
    for k, v in raw_names.items():
        name2raw.setdefault(v, k)
    t_names = {int(r["calligrapher_id"]): r["calligrapher"] for r in top}
    miss = [c for c in t_ids if t_names[c] not in name2raw]
    print(f"[top10] {len(top):,} 张, {len(t_ids)} 位: {[t_names[c] for c in t_ids]}", flush=True)
    print(f"[map] 名字 -> raw id: 命中 {len(t_ids)-len(miss)}/{len(t_ids)}"
          + (f", 未命中 {miss}" if miss else ""), flush=True)
    if miss and not a.allow_missing:
        raise SystemExit("[FATAL] top10 书家名在 raw 里缺失, 无法切片")
    if miss:                     # 冒烟: 只保留命中的那几位
        keep = [c for c in t_ids if c not in set(miss)]
        print(f"[map] allow-missing: 只用 {len(keep)} 位继续 (冒烟)", flush=True)
        t_ids = keep
        t_idx = {v: i for i, v in enumerate(t_ids)}
        top = [r for r in top if int(r["calligrapher_id"]) in set(t_ids)]
        t_lab = np.array([t_idx[int(r["calligrapher_id"])] for r in top], np.int64)

    # 与 judge_callig_rgb.py 完全相同的分层留出 (seed 0, 每类前 20%)
    rng = np.random.RandomState(a.seed)
    tr, ho = [], []
    for c in range(len(t_ids)):
        m = np.where(t_lab == c)[0]
        rng.shuffle(m)
        n_ho = int(len(m) * a.holdout)
        ho += list(m[:n_ho])
        tr += list(m[n_ho:])
    tr, ho = np.sort(np.array(tr)), np.sort(np.array(ho))
    print(f"[top10 split] 训练 {len(tr):,} / holdout {len(ho):,} "
          f"(与 top10 判定同 seed={a.seed})", flush=True)

    # 把 top10 训练样本映射到 raw 的路径 (用 raw 的图, 判定口径与 top10 版一致)
    t_paths_tr = [top[i]["image_path"] for i in tr]
    t_paths_ho = [top[i]["image_path"] for i in ho]

    # ── 预载 raw 图 ──
    t0 = time.time()
    imgs = np.zeros((n_raw, a.res, a.res), np.uint8)

    def rd(pp):
        i, p = pp
        cv2.setNumThreads(1)
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise RuntimeError(f"读图失败: {p}")
        if g.shape[0] != a.res:
            g = cv2.resize(g, (a.res, a.res), interpolation=cv2.INTER_AREA)
        return i, g

    with ThreadPoolExecutor(a.workers) as pool:
        for n, (i, g) in enumerate(pool.map(rd, list(enumerate(paths)))):
            imgs[i] = g
            if n and n % 50000 == 0:
                print(f"  [preload raw] {n:,}/{n_raw:,} {n/(time.time()-t0):.0f} img/s",
                      flush=True)
    print(f"[preload] raw {n_raw:,} 张 用时 {time.time()-t0:.0f}s", flush=True)

    # top10 图 (判定用, 单独小数组)
    t_imgs = np.zeros((len(top), a.res, a.res), np.uint8)

    def rd2(pp):
        i, p = pp
        cv2.setNumThreads(1)
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise RuntimeError(f"读图失败: {p}")
        if g.shape[0] != a.res:
            g = cv2.resize(g, (a.res, a.res), interpolation=cv2.INTER_AREA)
        return i, g

    with ThreadPoolExecutor(a.workers) as pool:
        for i, g in pool.map(rd2, list(enumerate([r["image_path"] for r in top]))):
            t_imgs[i] = g
    print(f"[preload] top10 {len(top):,} 张", flush=True)

    # ── 采样器 (raw 2133 类) ──
    by = [[] for _ in range(n_cls)]
    for i, c in enumerate(gid.tolist()):
        by[c].append(i)
    elig = np.array([c for c in range(n_cls) if len(by[c]) >= a.k_pick], np.int64)
    maxc = max(len(v) for v in by)
    pad = np.zeros((n_cls, maxc), np.int64)
    cnt = np.zeros(n_cls, np.int64)
    for c, v in enumerate(by):
        cnt[c] = len(v)
        if v:
            pad[c, :len(v)] = v
    n_pick = len(elig)
    print(f"[train] raw 可训练 {n_pick} 类 (每类>={a.k_pick}) x {a.k_pick} = "
          f"{n_pick*a.k_pick} 张/步, 下界 ln({a.k_pick})={np.log(a.k_pick):.3f}", flush=True)

    net = Enc(a.dim, a.deep).to(DEV).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=a.lr*0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    rng2 = np.random.RandomState(1)
    t1 = time.time()
    for step in range(1, a.steps + 1):
        sel = rng2.choice(elig, size=n_pick, replace=False)
        off = (rng2.rand(n_pick, a.k_pick) * cnt[sel][:, None]).astype(np.int64)
        ii = pad[sel[:, None], off].ravel()
        x = torch.from_numpy(imgs[ii]).unsqueeze(1).to(DEV).float().div_(255.0) \
            .contiguous(memory_format=torch.channels_last)
        y = torch.from_numpy(gid[ii]).to(DEV)
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
            el = time.time() - t1
            print(f"  [raw {step:5d}/{a.steps}] loss {loss.item():.4f} "
                  f"(下界 {np.log(a.k_pick):.3f}, 余量 {loss.item()-np.log(a.k_pick):+.3f})"
                  f" | {step*n_pick*a.k_pick/el:6.0f} 图/s | 峰值 "
                  f"{torch.cuda.max_memory_allocated()/1e9:.1f}G", flush=True)

    torch.save(net.state_dict(), f"{a.out}/encoder_callig_raw.pt")
    net.eval()

    @torch.no_grad()
    def feats_arr(arr):
        out = []
        for b in range(0, len(arr), 4096):
            x = torch.from_numpy(arr[b:b + 4096]).unsqueeze(1).to(DEV) \
                .float().div_(255.0)
            out.append(F.normalize(net(x).float(), dim=-1))
        return torch.cat(out)

    # ── 判定: 同一个 top10 留出集 ──
    Ztr = feats_arr(t_imgs[tr])
    Zho = feats_arr(t_imgs[ho])
    gmean_t = Ztr.mean(0)
    yho = t_lab[ho]

    def acc_with(T):
        pr = ((Zho - gmean_t) @ F.normalize(T, dim=-1).t()).argmax(1).cpu().numpy()
        return float((pr == yho).mean()), pr

    # A) 行 = top10 训练样本质心 (与 top10 版同口径)
    TA = torch.zeros(len(t_ids), a.dim, device=DEV)
    for c in range(len(t_ids)):
        m = torch.from_numpy(t_lab[tr] == c).to(DEV)
        TA[c] = F.normalize(Ztr[m].mean(0) - gmean_t, dim=-1)
    accA, _ = acc_with(TA)

    # B) 行 = **raw 里同名书家的全部样本**质心 (raw 规模的行)
    Zraw = feats_arr(imgs[:n_raw])                      # (N, dim) 已归一化
    zmean_raw = Zraw.mean(0)
    TB = torch.zeros(len(t_ids), a.dim, device=DEV)
    for c, cid in enumerate(t_ids):
        rid = name2raw[t_names[cid]]
        m = torch.from_numpy(gid == rid).to(DEV)
        v = Zraw[m].mean(0) - zmean_raw
        TB[c] = F.normalize(v, dim=-1)
    accB, _ = acc_with(TB)

    se = float(np.sqrt(0.542 * (1 - 0.542) / max(1, len(yho))))
    print("\n" + "=" * 74)
    print(f"top10 留出集 (n={len(yho):,}, 与 top10 版同划分) 的 per-sample top-1:")
    print(f"  基线 (top10 训练, 上一轮)       = 0.5420 ± 0.0135")
    print(f"  A) raw 编码器 + top10 训练行     = {accA:.4f}  (Δ {accA-0.5420:+.4f})")
    print(f"  B) raw 编码器 + raw 同名书家行   = {accB:.4f}  (Δ {accB-0.5420:+.4f})")
    print(f"     配对标准误参考 ±{1.96*se:.4f}")
    print("=" * 74, flush=True)

    # 存最好的那组为 callig_table (行序 = top10 排序 id, 与 v1 一致)
    Tbest = TB if accB >= accA else TA
    np.save(f"{a.out}/callig_table.npy", Tbest.cpu().numpy().astype(np.float32))
    json.dump({"classes": list(t_ids), "dim": int(a.dim),
               "source": f"rgb_raw2133_res{a.res}_deep{a.deep}"},
              open(f"{a.out}/callig_index.json", "w", encoding="utf-8"),
              ensure_ascii=False)
    json.dump({str(c): i for i, c in enumerate(t_ids)},
              open(f"{a.out}/callig_remap.json", "w", encoding="utf-8"),
              ensure_ascii=False)
    with torch.no_grad():
        e = Tbest
        cos = F.normalize(e, dim=-1) @ F.normalize(e, dim=-1).t()
        off = ~torch.eye(len(t_ids), dtype=torch.bool, device=DEV)
        mc = float(cos[off].abs().mean())
    print(f"[save] {a.out}/callig_table.npy {tuple(Tbest.shape)}  "
          f"行源={'raw同名书家' if accB >= accA else 'top10训练样本'}, 行|cos|均值={mc:.4f}",
          flush=True)
    json.dump(dict(acc_top10_baseline=0.5420, acc_A_raw_enc_top10rows=accA,
                   acc_B_raw_enc_rawrows=accB, n_holdout=int(len(yho)),
                   n_raw=n_raw, n_raw_cls=n_cls, n_raw_trained=int(n_pick),
                   k_pick=a.k_pick, steps=a.steps, res=a.res, deep=a.deep,
                   mean_abs_cos=mc, used_rows=("raw" if accB >= accA else "top10")),
              open("_sync_work/judge_callig_raw.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
