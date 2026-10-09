# -*- coding: utf-8 -*-
"""train_tables_big.py — 在 raw 大语料上训一张"大表" (书体12 / 书家2133 / 汉字9130)。

为什么这样设计 (2026-10-07 用户裁定后重写):
  1) **不再联合训练**。三个头互不相干, 各自独立训 (各自 optimizer / 各自特征源)。
     联合的唯一收益是"共享表征的正迁移", 代价是梯度互相干扰 —— 实测 callig 头卡在
     0.19 不涨, 就是代价被兑现。三个头合在一起也不省算力 (每图仍要过编码器正反向)。
  2) **正样本 = 同类不同样本** (跨样本), 不是"同一张图的增强视图"。
     用"同图两视图互为正"去训书家表, 等于要求主干抹掉粗细/形变 —— 而书家风格恰恰
     就是粗细。那是把目标函数搞反了。
  3) **K>=4 的类才参与训练** (用户裁定); 但**每一类都要有行** (稀有类用现有样本的
     质心兜底), 保证下游任何 id 都查得到。
  4) 输出按 raw 的 id 空间直接建行 (script 0-11 / callig 0-2132 / char 0-9129),
     另存 `*_names.json`。下游要哪几行由 tools/export_subset_tables.py 按键名切片。

特征源 (--source):
  dino : 冻结 DINOv2 CLS 特征 (assets/dino_feat_raw.npz, 已存在 393486x384)
  rgb  : 从零训三个**各自独立**的小 CNN 编码器 (每头一个), 再抽特征喂同一套头训练器

用法:
  PYTHONPATH=. python tools/train_tables_big.py --source dino
  PYTHONPATH=. python tools/train_tables_big.py --source rgb
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")

RAW = "/root/Workspace/xy/UNIFIED_RAW"
CSV_RAW = f"{RAW}/meta/train_clean.csv"
NPZ_RAW = "assets/dino_feat_raw.npz"


# ─────────────────────────── 损失 (主线配方) ───────────────────────────
def supcon_rows_vs_feat(zA, zB, y, temp=0.07):
    """表行 W(y) 与特征投影 P(x) 的双向对称 SupCon (与主线 train_minimal_best_triple_tables 一致)。"""
    logits = (zA @ zB.t()) / temp
    tgt = (y[:, None] == y[None, :]).float()
    lp1 = logits.log_softmax(1)
    lp0 = logits.log_softmax(0)
    l1 = -(lp1 * tgt).sum(1) / tgt.sum(1).clamp_min(1)
    l0 = -(lp0 * tgt).sum(0) / tgt.sum(0).clamp_min(1)
    return 0.5 * (l1.mean() + l0.mean())


def ortho_penalty(W_norm, w=10.0):
    n = W_norm.size(0)
    cos = W_norm @ W_norm.t()
    off = ~torch.eye(n, dtype=torch.bool, device=W_norm.device)
    return (cos[off] ** 2).mean() * w


def repel_penalty(W_norm, margin=0.15, w=3.0):
    """类间余弦超过 margin 就罚 —— 主线"把书家充分推开"的显式项。"""
    n = W_norm.size(0)
    cos = W_norm @ W_norm.t()
    off = ~torch.eye(n, dtype=torch.bool, device=W_norm.device)
    return F.relu(cos[off] - margin).pow(2).mean() * w


# ─────────────────── 向量化采样 (消掉每步上千次 numpy/标量同步) ───────────────────
def build_sampler(lab_cpu, n_cls):
    """把"每类样本索引"压成定长 padding 矩阵, 之后采样全向量化, 零 Python 循环。

    返回 padded (n_cls, max_cnt) int64 / cnt (n_cls,) int64
    """
    idx_by_cls = [[] for _ in range(n_cls)]
    for i, g in enumerate(lab_cpu):
        idx_by_cls[int(g)].append(i)
    maxc = max((len(v) for v in idx_by_cls), default=1)
    padded = np.zeros((n_cls, maxc), np.int64)
    cnt = np.zeros(n_cls, np.int64)
    for c, v in enumerate(idx_by_cls):
        cnt[c] = len(v)
        if v:
            padded[c, :len(v)] = v
    return padded, cnt


def sample_idx(padded, cnt, sel, k, rng):
    """sel: (S,) 选中的类; 每类取 k 个**互不重复**的真实样本 -> 恒定返回 S*k 个索引。

    做法: 只有 padded 的前 cnt 列是真样本; 把超出列的随机值置 2.0 (恒大于 rand),
    使它们必然排到 argsort 末尾 -> 前 k 个一定是有效样本。
    """
    take = np.minimum(cnt[sel], k)
    maxc = padded.shape[1]
    r = rng.rand(len(sel), maxc)
    cols = np.arange(maxc)[None, :]
    r = np.where(cols < take[:, None], r, np.float64(2.0))
    pick = np.argsort(r, axis=1)[:, :k]
    return padded[sel[:, None], pick].ravel()


def supcon_sym(zn, y, temp=0.07):
    """批内对称 SupCon (正样本 = 同类不同样本)。"""
    logits = (zn @ zn.t()) / temp
    tgt = (y[:, None] == y[None, :]).float()
    lp1 = logits.log_softmax(1)
    lp0 = logits.log_softmax(0)
    return 0.5 * (-(lp1 * tgt).sum(1) / tgt.sum(1).clamp_min(1)).mean() \
        + 0.5 * (-(lp0 * tgt).sum(0) / tgt.sum(0).clamp_min(1)).mean()


def pick_batch(probe, cands, target_gb, name):
    """标定 batch: 从小到大试, 取"峰值 <= target_gb"的最大档。probe(B) 返回峰值 GB。

    这是唯一会改变显存形状的地方 —— 标定完就锁定, 之后每步形状恒定, nvidia-smi 不再跳。
    """
    best = None
    for B in cands:
        try:
            gb = probe(B)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower():
                raise
            print(f"  [{name}] B={B:7d} OOM -> 停在 B={best}", flush=True)
            torch.cuda.empty_cache()
            break
        ok = gb <= target_gb
        print(f"  [{name}] B={B:7d} 峰值 {gb:5.2f}G {'✓ 采用' if ok else '✗ 超预算'}"
              f"{'' if ok else ' -> 回退上一档'}", flush=True)
        torch.cuda.empty_cache()
        if not ok:
            break
        best = B
    assert best, f"[{name}] 连最小档都超预算, 降低最小候选"
    return best


# ─────────────────────────── 头训练器 ───────────────────────────
def train_head(feat, lab, n_cls, dim, *, name, steps, lr, k, n_cls_per_step,
               mode, holdout_idx, D_in, debias_by=None, log_every=500, temp=0.07,
               target_gb=20.0, k_elig=None):
    """在**冻结特征**上训一个表头。三个头各自独立调用, 互不共享参数。

    k      : 每步每类采几张 (决定 batch)
    k_elig : 参与训练的最少样本数 (用户裁定的 K, 默认 = k)
    mode: 'script' -> 硬正交 ETF; 'callig' -> 类间推开; 'char' -> 纯跨样本 InfoNCE
    """
    dev = feat.device
    k_elig = k if k_elig is None else k_elig
    elig = np.array(sorted([c for c in range(n_cls)
                            if int((lab == c).sum()) >= k_elig]), dtype=np.int64)
    print(f"\n[{name}] 类数 {n_cls}, K>={k} 参与训练 {len(elig)} 类, dim={dim}", flush=True)

    # 去内容偏置 (书体头专用): 减去汉字均值, 去掉"字内容"带来的偏置
    if debias_by is not None:
        cb = int(debias_by.max()) + 1
        mean = torch.zeros(cb, D_in, device=dev)
        cnt = torch.zeros(cb, 1, device=dev)
        mean.index_add_(0, debias_by, feat)
        cnt.index_add_(0, debias_by, torch.ones(lab.size(0), 1, device=dev))
        feat_in = F.normalize(feat - (mean / cnt.clamp_min(1))[debias_by], dim=-1)
    else:
        feat_in = feat

    W = nn.Embedding(n_cls, dim).to(dev)
    with torch.no_grad():
        g = torch.randn(max(dim, n_cls), dim, device=dev)
        q, _ = torch.linalg.qr(g)
        W.weight.copy_(q[:n_cls])
    P = nn.Sequential(nn.Linear(D_in, 256), nn.GELU(), nn.Linear(256, dim)).to(dev)
    # ★ 表行用**类质心过初始投影**暖启动。
    #   随机初始化时同类的 zA·zB ≈ 0 -> softmax 均匀 -> 每对梯度 O(1/B), B=16k 时等于不学。
    #   质心暖启动让同类一开始就对齐 -> loss 立刻下降 -> 梯度恢复有意义; 之后再靠
    #   SupCon(类内聚) + 推开项(类间散) 塑形。
    with torch.no_grad():
        cen = torch.zeros(n_cls, D_in, device=dev)
        ccnt = torch.zeros(n_cls, 1, device=dev)
        cen.index_add_(0, lab, feat_in)
        ccnt.index_add_(0, lab, torch.ones(lab.size(0), 1, device=dev))
        has = (ccnt.squeeze(1) > 0)
        cen[has] = cen[has] / ccnt[has].clamp_min(1)
        warm = F.normalize(P(cen), dim=-1)
        W.weight.copy_(torch.where(has[:, None], warm, W.weight.detach()))
    print(f"  [{name}] 表行暖启动: {int(has.sum())} 类用质心, "
          f"{int((~has).sum())} 类无样本(留 QR 随机)", flush=True)
    opt = torch.optim.AdamW(list(W.parameters()) + list(P.parameters()),
                            lr=lr, weight_decay=1e-4)

    # 向量化采样: padded/cnt 一次性建好, 每步零 Python 循环
    padded, cnt = build_sampler(lab.cpu().numpy(), n_cls)
    rng = np.random.RandomState(0)

    def _core(n_pick):
        """一次真实训练步 (标定与正式训练共用)。"""
        idx = torch.from_numpy(
            sample_idx(padded, cnt, rng.choice(elig, size=n_pick, replace=False), k, rng)
        ).to(dev)
        y = lab[idx]
        zA = F.normalize(W(y), dim=-1)
        zB = F.normalize(P(feat_in[idx]), dim=-1)
        loss = supcon_rows_vs_feat(zA, zB, y, temp)
        Wn = F.normalize(W.weight, dim=-1)
        if mode == "script":
            loss = loss + ortho_penalty(Wn, 10.0)
        elif mode == "callig":
            loss = loss + repel_penalty(Wn, 0.15, 3.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss, zA.size(0)

    def probe(B):
        n_pick = max(1, B // k)
        if n_pick > len(elig):
            return float("inf")
        W.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            _core(n_pick)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1e9

    cands = sorted({c for c in (256, 512, 1024, 2048, 4096, 8192, len(elig))
                    if c <= len(elig)})
    cands = [c * k for c in cands] or [len(elig) * k]
    n_pick = max(1, min(pick_batch(probe, cands, target_gb, name) // k, len(elig)))
    print(f"  [{name}] 锁定 batch = {n_pick} 类 x {k} 张 = {n_pick*k} 样本/步", flush=True)

    best_acc, best_W = -1.0, None
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, steps + 1):
        loss, B = _core(n_pick)
        if step % log_every == 0 or step == steps:
            with torch.no_grad():
                Wn = F.normalize(W.weight, dim=-1)
                acc, cov = top1_eval(P(feat_in), Wn, lab, elig, holdout_idx)
            dt = time.time() - t0
            print(f"  [{name} {step:5d}/{steps}] loss {loss.item():.4f} (chance "
                  f"{np.log(B):.3f}) | holdout top-1 {acc:.4f} (cov {cov:.2f}) | "
                  f"{step/dt:.1f} it/s | 峰值 "
                  f"{torch.cuda.max_memory_allocated()/1e9:.2f}G", flush=True)
            if acc > best_acc:
                best_acc, best_W = acc, Wn.detach()
    return best_W.cpu().numpy().astype(np.float32), best_acc, elig, P, feat_in


@torch.no_grad()
def top1_eval(z, Wn, lab, elig, holdout_idx, chunk=8192):
    """近邻表行 top-1 (只在参与训练的类上评), 仅统计 holdout 样本。"""
    if holdout_idx is None or len(holdout_idx) == 0:
        return 0.0, 0.0
    elig_t = torch.from_numpy(elig).to(z.device)
    Ts = Wn[elig_t]                       # (E,d)
    hit = tot = 0
    for b in range(0, len(holdout_idx), chunk):
        sel = holdout_idx[b:b + chunk]
        zz = z[sel]
        pred = (zz @ Ts.t()).argmax(1)
        hit += int((elig_t[pred] == lab[sel]).sum())
        tot += len(sel)
    return hit / max(1, tot), 1.0


# ─────────────────────────── 特征源 ───────────────────────────
def feats_from_dino(rows, dev):
    z = np.load(NPZ_RAW)
    ids, feat = z["ids"], z["feat"]
    pos = {int(i): k for k, i in enumerate(ids)}
    order = np.array([pos[int(r["img_id"])] for r in rows], dtype=np.int64)
    f = torch.from_numpy(feat[order]).to(dev)
    print(f"[dino] 特征 {tuple(f.shape)}  (冻结, 直接可用)", flush=True)
    return f, f.shape[1]


class Enc(nn.Module):
    """小 CNN 编码器 (1x96x96 -> 256)。每个头各训一个, 不共享。"""

    def __init__(self, emb=256):
        super().__init__()

        def blk(i, o, s=2):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False),
                                 nn.BatchNorm2d(o), nn.SiLU())
        self.enc = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), blk(128, 256),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                 nn.Linear(256, emb, bias=False), nn.BatchNorm1d(emb))

    def forward(self, x):
        return self.enc(x)


MAXB_IMGS = 32768        # 每步最多图数 (双缓冲按此预分配, 标定只改切片)


def feats_from_rgb(rows, lab_script, lab_callig, lab_char, n_scr, n_cal, n_chr,
                   dims, dev, res=96, workers=16, steps=1500, k=4, c_per_step=256,
                   log_every=250, target_gb=20.0, k_script=256):
    """给每个标签空间**各自独立**训一个编码器, 再各自抽特征。

    正样本 = 同类不同样本 (跨样本), 增强只做几何仿射 (不做"同图两视图互为正")。
    """
    import cv2
    torch.backends.cudnn.benchmark = True
    t0 = time.time()
    imgs = np.zeros((len(rows), res, res), np.uint8)
    pool = ThreadPoolExecutor(workers)

    def rd(i):
        p = rows[i]["image_path"]
        full = p if os.path.isabs(p) else os.path.join(RAW, p)
        g = cv2.imread(full, cv2.IMREAD_GRAYSCALE)
        if g is None:
            return i, np.zeros((res, res), np.uint8)
        if g.shape[0] != res:
            g = cv2.resize(g, (res, res), interpolation=cv2.INTER_AREA)
        return i, g

    for n, (i, g) in enumerate(pool.map(rd, range(len(rows)))):
        imgs[i] = g
        if n % 100000 == 0 and n:
            print(f"  [preload] {n:,}/{len(rows):,} {n/(time.time()-t0):.0f} img/s", flush=True)
    print(f"[rgb] 预载 {len(rows):,} 张 {res}x{res} 用时 {time.time()-t0:.0f}s", flush=True)

    def affine(x, rot_deg=4.0, scale=0.05, shift=3.0):
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

    # 图片**留在 RAM**, 用 pinned 双缓冲 + 异步 H2D: 算第 t 步的同时传第 t+1 步
    imgs_cpu = torch.from_numpy(imgs)            # uint8 CPU view, 零拷贝
    Bmax = MAXB_IMGS                             # 固定上限, 标定只改切片长度 -> 形状不再跳变
    pin = [torch.empty((Bmax, res, res), dtype=torch.uint8, pin_memory=True)
           for _ in range(2)]
    gbuf = [torch.empty((Bmax, 1, res, res), dtype=torch.uint8, device=dev)
            for _ in range(2)]
    ev = [torch.cuda.Event(), torch.cuda.Event()]
    h2d = torch.cuda.Stream()
    print(f"[rgb] 图片留 RAM ({imgs.size/1e9:.2f}G); pinned 双缓冲 {Bmax}x{res}x{res} "
          f"= {2*Bmax*res*res/1e6:.0f}MB; H2D 与计算重叠", flush=True)

    def prepare(buf, idx_np):
        """把 idx 的图收集进 pinned 缓冲并异步上卡 (跑在 h2d 流里)。"""
        n = len(idx_np)
        pin[buf][:n].copy_(imgs_cpu[idx_np])
        with torch.cuda.stream(h2d):
            gbuf[buf][:n, 0].copy_(pin[buf][:n], non_blocking=True)
        return n

    out = {}
    for nm, lab, n_cls in (("script", lab_script, n_scr), ("callig", lab_callig, n_cal),
                           ("char", lab_char, n_chr)):
        kk = k_script if nm == "script" else k       # 每步每类采几张 (决定 batch)
        elig = np.array(sorted([c for c in range(n_cls)
                                if int((lab == c).sum()) >= k]), dtype=np.int64)
        padded, cnt = build_sampler(lab.cpu().numpy(), n_cls)
        rng = np.random.RandomState(0)
        net = Enc(256).to(dev).to(memory_format=torch.channels_last)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        n_max = max(1, min(len(elig), MAXB_IMGS // kk))
        print(f"\n[rgb/{nm}] 独立编码器: 类数 {n_cls}, K>={k} 参与 {len(elig)}, "
              f"每类采 {kk} 张", flush=True)

        def _next_idx(n_pick):
            return sample_idx(padded, cnt,
                              rng.choice(elig, size=n_pick, replace=False), kk, rng)

        def _core(buf, idx_np):
            """一次真实训练步 (等 h2d 流 -> 计算 -> 反传)。标定与正式训练共用。"""
            torch.cuda.current_stream().wait_event(ev[buf])
            x = gbuf[buf][:len(idx_np)].float().div_(255.0) \
                .contiguous(memory_format=torch.channels_last)
            y = lab[torch.from_numpy(idx_np).to(dev)]
            with torch.cuda.amp.autocast(enabled=True):
                loss = supcon_sym(F.normalize(net(affine(x).float()), dim=-1), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            return loss

        def probe(B):
            """实跑 3 步测峰值显存 (GB)。"""
            n_pick = max(1, B // kk)
            if n_pick > len(elig):
                return float("inf")
            net.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(3):
                idx = _next_idx(n_pick)
                prepare(0, idx)
                ev[0].record(h2d)
                _core(0, idx)
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() / 1e9

        cands = sorted({c for c in (128, 256, 512, 1024, 2048, 4096, len(elig))
                        if c <= len(elig)})
        cands = [c * kk for c in cands] or [len(elig) * kk]
        n_pick = max(1, min(pick_batch(probe, cands, target_gb, f"rgb/{nm}") // kk, n_max))
        print(f"  [rgb/{nm}] 锁定 batch = {n_pick} 类 x {kk} 张 = {n_pick*kk} 图/步",
              flush=True)

        t1 = time.time()
        torch.cuda.reset_peak_memory_stats()
        idx_cur = _next_idx(n_pick)
        prepare(0, idx_cur)
        ev[0].record(h2d)
        cur, nxt = 0, 1
        for step in range(1, steps + 1):
            if step < steps:                      # CPU 预取下一步, 与当前步 GPU 计算重叠
                idx_nxt = _next_idx(n_pick)
                prepare(nxt, idx_nxt)
                ev[nxt].record(h2d)
            loss = _core(cur, idx_cur)
            if step < steps:
                cur, nxt = nxt, cur
                idx_cur = idx_nxt
            if step % log_every == 0 or step == steps:
                print(f"  [rgb/{nm} {step:5d}/{steps}] loss {loss.item():.4f} | "
                      f"{step/(time.time()-t1):.2f} it/s | 峰值 "
                      f"{torch.cuda.max_memory_allocated()/1e9:.2f}G", flush=True)
        net.eval()
        # 抽全训练集特征: 同样走 pinned 异步 H2D
        EB = 4096
        pin_e = torch.empty((EB, res, res), dtype=torch.uint8, pin_memory=True)
        f = []
        with torch.no_grad():
            for b in range(0, len(rows), EB):
                n = min(EB, len(rows) - b)
                pin_e[:n].copy_(imgs_cpu[b:b + n])
                x = pin_e[:n].unsqueeze(1).to(dev, non_blocking=True).float().div_(255.0)
                f.append(net(x).float())
        out[nm] = torch.cat(f).to(dev)
        print(f"[rgb/{nm}] 特征抽出 {tuple(out[nm].shape)}", flush=True)
        torch.save(net.state_dict(), f"{ARGS.out}/encoder_{nm}.pt")
    return out, 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["dino", "rgb"], required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--k", type=int, default=4, help="每类最少样本数 (少于它的类不参与训练)")
    ap.add_argument("--k-script", type=int, default=256,
                    help="书体头每步每类采几张 (只有 12 类, 取大值把 batch 撑起来)")
    ap.add_argument("--d-script", type=int, default=32)
    ap.add_argument("--d-callig", type=int, default=128)
    ap.add_argument("--d-char", type=int, default=384)
    ap.add_argument("--steps-script", type=int, default=2000)
    ap.add_argument("--steps-callig", type=int, default=3000)
    ap.add_argument("--steps-char", type=int, default=4000)
    ap.add_argument("--c-per-step-callig", type=int, default=256)
    ap.add_argument("--c-per-step-char", type=int, default=1024)
    ap.add_argument("--enc-steps", type=int, default=1500, help="rgb: 每个编码器训练步数")
    ap.add_argument("--target-gb", type=float, default=20.0,
                    help="batch 标定的显存目标 (24G 卡留余量)")
    ap.add_argument("--holdout", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    global ARGS
    ARGS = a
    if not a.out:
        a.out = f"assets/triple_tables_big_{a.source}"
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(0); np.random.seed(0)

    rows = list(csv.DictReader(open(CSV_RAW, encoding="utf-8")))
    if a.limit:
        rows = rows[:a.limit]
    n_scr = max(int(r["script_id"]) for r in rows) + 1
    n_cal = max(int(r["calligrapher_id"]) for r in rows) + 1
    n_chr = max(int(r["character_id"]) for r in rows) + 1
    print(f"[data] {len(rows):,} 行  script={n_scr} callig={n_cal} char={n_chr}", flush=True)

    lab_s = torch.tensor([int(r["script_id"]) for r in rows], device=dev)
    lab_g = torch.tensor([int(r["calligrapher_id"]) for r in rows], device=dev)
    lab_c = torch.tensor([int(r["character_id"]) for r in rows], device=dev)

    # holdout: 按样本随机 5% (评表只看 holdout, 训练集只用来训)
    rng = np.random.RandomState(1234)
    perm = rng.permutation(len(rows))
    hold = np.sort(perm[:int(len(rows) * a.holdout)])
    print(f"[holdout] {len(hold):,} 张 ({a.holdout:.0%})", flush=True)

    if a.source == "dino":
        f_dino, D = feats_from_dino(rows, dev)
        F_s = F_g = F_c = f_dino
    else:
        fs, D = feats_from_rgb(rows, lab_s, lab_g, lab_c, n_scr, n_cal, n_chr,
                               (a.d_script, a.d_callig, a.d_char), dev,
                               steps=a.enc_steps, target_gb=a.target_gb,
                               k=a.k, k_script=a.k_script)
        F_s, F_g, F_c = fs["script"], fs["callig"], fs["char"]

    t0 = time.time()
    W_s, acc_s, elig_s, P_s, fi_s = train_head(
        F_s, lab_s, n_scr, a.d_script, name="script", steps=a.steps_script, lr=3e-3,
        k=a.k_script, n_cls_per_step=n_scr, mode="script", holdout_idx=hold, D_in=D,
        debias_by=lab_c, log_every=max(1, a.steps_script // 5), target_gb=a.target_gb,
        k_elig=a.k)
    W_g, acc_g, elig_g, P_g, fi_g = train_head(
        F_g, lab_g, n_cal, a.d_callig, name="callig", steps=a.steps_callig, lr=2e-3,
        k=a.k, n_cls_per_step=a.c_per_step_callig, mode="callig", holdout_idx=hold,
        D_in=D, log_every=max(1, a.steps_callig // 8), target_gb=a.target_gb)
    W_c, acc_c, elig_c, P_c, fi_c = train_head(
        F_c, lab_c, n_chr, a.d_char, name="char", steps=a.steps_char, lr=1.5e-3,
        k=a.k, n_cls_per_step=a.c_per_step_char, mode="char", holdout_idx=hold,
        D_in=D, log_every=max(1, a.steps_char // 8), target_gb=a.target_gb)

    # ── 稀有类兜底: K<k 的类用"该类特征质心经训好的投影 P"填行, 保证下游任何 id 都查得到 ──
    @torch.no_grad()
    def fill_rare(feat_in, lab, W, elig, P, nm):
        n_cls = W.shape[0]
        cen = torch.zeros(n_cls, feat_in.size(1), device=dev)
        cnt = torch.zeros(n_cls, 1, device=dev)
        cen.index_add_(0, lab, feat_in)
        cnt.index_add_(0, lab, torch.ones(lab.size(0), 1, device=dev))
        rare = [c for c in range(n_cls) if c not in set(elig.tolist()) and int(cnt[c]) > 0]
        empty = [c for c in range(n_cls) if int(cnt[c]) == 0]
        if rare:
            v = F.normalize(P(cen[rare] / cnt[rare].clamp_min(1)), dim=-1).cpu().numpy()
            W[np.array(rare)] = v
        print(f"[{nm}] 稀有类兜底(质心过投影) = {len(rare)}, 完全无样本的类 = {len(empty)}",
              flush=True)
        return W

    W_s = fill_rare(fi_s, lab_s, W_s, elig_s, P_s, "script")
    W_g = fill_rare(fi_g, lab_g, W_g, elig_g, P_g, "callig")
    W_c = fill_rare(fi_c, lab_c, W_c, elig_c, P_c, "char")

    names_s, names_g, names_c = {}, {}, {}
    for r in rows:
        names_s[int(r["script_id"])] = r["script"]
        names_g[int(r["calligrapher_id"])] = r["calligrapher"]
        names_c[int(r["character_id"])] = r["character"]

    for nm, W, names in (("font", W_s, names_s), ("callig", W_g, names_g),
                         ("char", W_c, names_c)):
        np.save(f"{a.out}/{nm}_table.npy", W)
        cls = [names.get(i, "") for i in range(W.shape[0])]
        json.dump({"classes": cls, "dim": int(W.shape[1]), "source": a.source,
                   "id_space": f"raw {nm} id"},
                  open(f"{a.out}/{nm}_index.json", "w", encoding="utf-8"),
                  ensure_ascii=False)
        json.dump({str(i): i for i in range(W.shape[0])},
                  open(f"{a.out}/{nm}_identity_remap.json", "w", encoding="utf-8"),
                  ensure_ascii=False)
        print(f"[save] {a.out}/{nm}_table.npy {W.shape}  classes={len(cls)}", flush=True)

    # 分离度: 表行的类间余弦 (书家表是最关心的)
    with torch.no_grad():
        for nm, W, elig in (("font", W_s, elig_s), ("callig", W_g, elig_g),
                            ("char", W_c, elig_c)):
            t = torch.from_numpy(W)[torch.from_numpy(elig)].to(dev)
            t = F.normalize(t, dim=-1)
            cos = t @ t.t()
            n = cos.size(0)
            off = ~torch.eye(n, dtype=torch.bool, device=dev)
            print(f"[sep] {nm:7s} 类间余弦 均值={cos[off].mean():+.4f} "
                  f"|cos|均值={cos[off].abs().mean():.4f} "
                  f">0.15 占比={(cos[off] > 0.15).float().mean():.4f}", flush=True)
            json.dump({"n_cls_trained": int(n), "mean_cos": float(cos[off].mean()),
                       "mean_abs_cos": float(cos[off].abs().mean()),
                       "share_gt_0.15": float((cos[off] > 0.15).float().mean())},
                      open(f"{a.out}/sep_{nm}.json", "w", encoding="utf-8"))

    json.dump({"source": a.source, "args": vars(a),
               "holdout_top1": {"script": acc_s, "callig": acc_g, "char": acc_c},
               "n_trained": {"script": len(elig_s), "callig": len(elig_g),
                             "char": len(elig_c)},
               "n_rows": {"script": int(W_s.shape[0]), "callig": int(W_g.shape[0]),
                          "char": int(W_c.shape[0])},
               "minutes": (time.time() - t0) / 60.0},
              open(f"{a.out}/train_log.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n[done] {a.out}  用时 {(time.time()-t0)/60:.1f}min  "
          f"holdout top-1: script {acc_s:.4f} / callig {acc_g:.4f} / char {acc_c:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
