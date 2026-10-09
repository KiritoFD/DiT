# -*- coding: utf-8 -*-
"""train_tables_top10_rgb.py — 在 **top10 数据集**上, 用**原始 RGB 图从零**训三张表。

为什么要它: 现有三表 (assets/triple_tables_best_minimal/) 都是在**冻结 DINO 特征**上训的
(tools/train_minimal_best_triple_tables.py 只读 assets/dino_feat_top10_g.npz, 不读图)。
本脚本改从**像素**从零训一个编码器, 用来做"换表版 v68"的单变量对照。

设计 (朴素正确优先):
  1) 三个标签空间**各自一个编码器**, 互不共享 (联合训练不省算力, 且会互相干扰)。
  2) **正样本 = 同类不同样本** (跨样本), 不是"同一张图的两个增强视图" —— 后者等于
     逼编码器抹掉粗细/形变, 而书家风格恰恰是粗细。
  3) 训练完 **表 = 该类训练样本在编码器空间的归一化质心** (与 DINO 三表的"类原型"同义)。
  4) 输出格式与 assets/triple_tables_best_minimal/ **逐字对齐**:
       {font,callig,char}_table.npy   形状 3x16 / 10x32 / 4690x256 (与 v68 config 的
                                      script/callig/char_embed_dim 完全一致)
       {font,callig,char}_index.json  {"classes": [...], "dim": N, "source": "..."}
       {font,callig,char}_remap.json  {"原始id": 行号}
     -> v68 只改 --triple-table-prefix 与三个 remap 路径 = 纯单变量"换表"。

用法:
  PYTHONPATH=. python tools/train_tables_top10_rgb.py            # 全量
  PYTHONPATH=. python tools/train_tables_top10_rgb.py --steps 60 # 冒烟
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


def build_sampler(lab, n_cls, k):
    """padded (n_cls,maxc) + cnt, 供向量化采样。"""
    by = [[] for _ in range(n_cls)]
    for i, g in enumerate(lab.tolist()):
        by[int(g)].append(i)
    maxc = max((len(v) for v in by), default=1)
    padded = np.zeros((n_cls, maxc), np.int64)
    cnt = np.zeros(n_cls, np.int64)
    for c, v in enumerate(by):
        cnt[c] = len(v)
        if v:
            padded[c, :len(v)] = v
    return padded, cnt


def sample_idx(padded, cnt, sel, k, rng):
    """每类取 k 个互不重复样本 (无效列随机值置 2.0 排到末尾)。"""
    take = np.minimum(cnt[sel], k)
    maxc = padded.shape[1]
    r = rng.rand(len(sel), maxc)
    r = np.where(np.arange(maxc)[None, :] < take[:, None], r, np.float64(2.0))
    return padded[sel[:, None], np.argsort(r, axis=1)[:, :k]].ravel()


def affine(x, rot_deg=4.0, scale=0.05, shift=3.0):
    N = x.size(0)
    ang = (torch.rand(N, device=x.device) * 2 - 1) * (rot_deg * np.pi / 180.0)
    sc = 1.0 + (torch.rand(N, device=x.device) * 2 - 1) * scale
    tx = (torch.rand(N, device=x.device) * 2 - 1) * (shift / x.size(2)) * 2
    ty = (torch.rand(N, device=x.device) * 2 - 1) * (shift / x.size(3)) * 2
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([torch.stack([cos * sc, -sin * sc, tx], -1),
                         torch.stack([sin * sc, cos * sc, ty], -1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="border")


class Enc(nn.Module):
    """1 x res x res -> dim。每个标签空间各训一个。deep=1 时多一层卷积 (增容量)。"""

    def __init__(self, dim, res=96, deep=0):
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


def train_head(name, imgs_cpu, lab_np, n_cls, dim, *, res, steps, k_pick, k_min, lr,
               temp, log_every, center, max_cls=None, init_from=None, save_enc=None,
               deep=0):
    dev = DEV
    lab = torch.from_numpy(lab_np).to(dev)
    padded, cnt = build_sampler(torch.from_numpy(lab_np), n_cls, k_pick)
    elig = np.where(cnt >= k_min)[0].astype(np.int64)          # K>=k_min 的类参与训练
    rng = np.random.RandomState(0)
    n_pick = min(len(elig), max_cls) if max_cls else len(elig)
    net = Enc(dim, res, deep=deep).to(dev).to(memory_format=torch.channels_last)
    if init_from:                      # ★ 真·续训: 从上一轮编码器暖启动
        p = f"{init_from}/encoder_{name}.pt"
        if os.path.isfile(p):
            net.load_state_dict(torch.load(p, map_location=dev))
            print(f"  [{name}] 续训: 已加载 {p}", flush=True)
        else:
            print(f"  [{name}] 续训: {p} 不存在, 从零开始", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps),
                                                       eta_min=lr * 0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    floor = float(np.log(k_pick))     # SupCon 理论下界: 正样本 k_pick 个均匀分质量
    print(f"\n[{name}] 类数 {n_cls}, 样本 {len(lab_np):,}, K>={k_min} 可训练 "
          f"{len(elig)} 类 (其余类用质心补行), dim={dim}, "
          f"每步 {n_pick} 类 x {k_pick} = {n_pick*k_pick} 张, loss 下界 {floor:.3f}",
          flush=True)
    t0 = time.time()
    for step in range(1, steps + 1):
        sel = rng.choice(elig, size=n_pick, replace=False)
        idxs = sample_idx(padded, cnt, sel, k_pick, rng)
        x = torch.from_numpy(imgs_cpu[idxs]).unsqueeze(1).to(dev) \
            .float().div_(255.0).contiguous(memory_format=torch.channels_last)
        y = lab[torch.from_numpy(idxs).to(dev)]
        with torch.cuda.amp.autocast(enabled=True):
            zn = F.normalize(net(affine(x).float()), dim=-1)
            logits = (zn @ zn.t()) / temp
            tgt = (y[:, None] == y[None, :]).float()
            loss = 0.5 * (-(logits.log_softmax(1) * tgt).sum(1)
                          / tgt.sum(1).clamp_min(1)).mean() \
                + 0.5 * (-(logits.log_softmax(0) * tgt).sum(0)
                         / tgt.sum(0).clamp_min(1)).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % log_every == 0 or step == 1 or step == steps:
            ips = step * n_pick * k_pick / max(time.time() - t0, 1e-9)
            excess = loss.item() - floor      # 距下界的余量: 趋于 0 才算收敛
            ep = step * n_pick * k_pick / max(1, len(lab_np))      # 已训 epoch 数
            print(f"  [{name} {step:5d}/{steps}] loss {loss.item():.4f} "
                  f"(下界 {floor:.3f}, 余量 {excess:+.3f}) | {ep:6.0f} epoch | "
                  f"{ips:5.0f} 图/s | 峰值 "
                  f"{torch.cuda.max_memory_allocated()/1e9:.1f}G", flush=True)

    # ── 表 = 每类训练样本的归一化质心 ──
    net.eval()
    Z = np.zeros((len(lab_np), dim), np.float32)
    with torch.no_grad():
        for b in range(0, len(lab_np), 4096):
            nb = min(4096, len(lab_np) - b)
            x = torch.from_numpy(imgs_cpu[b:b + nb]).unsqueeze(1).to(dev) \
                .float().div_(255.0)
            Z[b:b + nb] = F.normalize(net(x).float(), dim=-1).cpu().numpy()
    T = np.zeros((n_cls, dim), np.float32)
    gmean = Z.mean(0)                      # 去全局均值: 否则所有类质心都朝"平均图"方向,
    n_cen = n_rand = 0                     # 表行会共线 (实测 |cos| 0.9+)
    for c in range(n_cls):
        m = lab_np == c
        if m.sum():
            # ★ 每一行都要有值: 能训的(K>=k_min)用训练好的编码器质心, 样本少的类
            #   也用它自己的质心 (哪怕只有 1 张)。否则未训的 3666 个字会是零向量,
            #   v68 查这些字就拿到 0 —— DINO 那版是随机 init 所以没暴露这个问题。
            v = Z[m].mean(0) - (gmean if center else 0.0)
            T[c] = v / (np.linalg.norm(v) + 1e-9)
            n_cen += 1
        else:
            # 只出现在 eval 词表、训练集里一次都没出现的类: 给小随机 (与 DINO 版同策略)
            T[c] = np.random.RandomState(c).normal(0, 0.02, dim).astype(np.float32)
            n_rand += 1
    print(f"[{name}] 表行填充: 质心 {n_cen} 行 / 小随机 {n_rand} 行 "
          f"(0 行的类才是真缺陷)", flush=True)

    # ── 自检: 留一近邻 top-1 + 表行分离度 ──
    hit = tot = 0
    for c in range(n_cls):
        m = np.where(lab_np == c)[0]
        if len(m) < 2:
            continue
        q = Z[m[-1]] - (gmean if center else 0.0)
        p = int((T @ q).argmax())
        hit += int(p == c)
        tot += 1
    loo = hit / max(1, tot)
    with torch.no_grad():
        e = torch.from_numpy(elig).to(dev)
        w = F.normalize(torch.from_numpy(T).to(dev)[e], dim=-1)
        cos = w @ w.t()
        off = ~torch.eye(len(e), dtype=torch.bool, device=dev)
        mc = float(cos[off].abs().mean())
    print(f"[{name}] 自检: 留一近邻 top-1 = {loo:.4f} (n={tot} 类) | "
          f"表行|cos|均值 = {mc:.4f}", flush=True)
    if save_enc:                       # ★ 存编码器, 供下一轮真·续训 (--init-from)
        os.makedirs(save_enc, exist_ok=True)
        torch.save(net.state_dict(), f"{save_enc}/encoder_{name}.pt")
        print(f"[{name}] 编码器已存 {save_enc}/encoder_{name}.pt", flush=True)
    return T, dict(loo_top1=float(loo), mean_abs_cos=mc, n_trained=int(len(elig)),
                   n_rows=int(n_cls), dim=int(dim), steps=steps,
                   batch_imgs=int(n_pick * k_pick), k_pick=int(k_pick),
                   k_min=int(k_min), loss_floor=float(floor),
                   loss_final=float(loss.item()), resumed=bool(init_from))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="exp-std/csv/train.csv")
    ap.add_argument("--eval-csv", default="exp-std/csv/eval200_fixed.csv")
    ap.add_argument("--out", default="assets/triple_tables_top10_rgb")
    ap.add_argument("--res", type=int, default=96)
    ap.add_argument("--dim-script", type=int, default=16)
    ap.add_argument("--dim-callig", type=int, default=32)
    ap.add_argument("--dim-char", type=int, default=256)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--k-min", type=int, default=4,
                    help="参与训练的最少样本数 (用户裁定的 K=4)")
    ap.add_argument("--k-pick-script", type=int, default=1024, help="书体头每类每步采样数")
    ap.add_argument("--k-pick-callig", type=int, default=1024, help="书家头每类每步采样数")
    ap.add_argument("--k-pick-char", type=int, default=4, help="汉字头每类每步采样数")
    ap.add_argument("--center", type=int, default=1, help="表行去全局均值 (降低共线)")
    ap.add_argument("--init-from", default="",
                    help="上一轮的表目录: 存在 encoder_<头>.pt 则从其暖启动 (=真续训)")
    ap.add_argument("--save-enc", type=int, default=1, help="存编码器供下一轮续训")
    ap.add_argument("--only", default="", help="只训指定头 (逗号分隔: font,callig,char)")
    ap.add_argument("--deep", type=int, default=0, help="编码器多加一层卷积 (增容量)")
    ap.add_argument("--max-cls-per-step", type=int, default=4096,
                    help="每步最多取多少个类 (汉字头上千类时会太重)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    os.makedirs(a.out, exist_ok=True)

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    if a.limit:
        rows = rows[:a.limit]
    ev = list(csv.DictReader(open(a.eval_csv, encoding="utf-8"))) \
        if os.path.isfile(a.eval_csv) else []
    print(f"[data] train {len(rows):,} 行 + eval 词表 {len(ev):,} 行", flush=True)

    # 词表与现有三表一致: train ∪ eval 的 id/字形
    fnt_ids = sorted({int(r["script_id"]) for r in rows} | {int(r["script_id"]) for r in ev})
    cal_ids = sorted({int(r["calligrapher_id"]) for r in rows}
                     | {int(r["calligrapher_id"]) for r in ev})
    hanzi = sorted({r["character"] for r in rows} | {r["character"] for r in ev})
    f_idx = {v: i for i, v in enumerate(fnt_ids)}
    c_idx = {v: i for i, v in enumerate(cal_ids)}
    h_idx = {v: i for i, v in enumerate(hanzi)}
    print(f"[vocab] 书体 {len(fnt_ids)} ({fnt_ids}) / 书家 {len(cal_ids)} / "
          f"汉字 {len(hanzi)}", flush=True)

    lab_f = np.array([f_idx[int(r["script_id"])] for r in rows], np.int64)
    lab_c = np.array([c_idx[int(r["calligrapher_id"])] for r in rows], np.int64)
    lab_h = np.array([h_idx[r["character"]] for r in rows], np.int64)

    # ── 预载图 (26k 张 x res x res, RAM) ──
    t0 = time.time()
    imgs = np.zeros((len(rows), a.res, a.res), np.uint8)

    def rd(i):
        cv2.setNumThreads(1)
        p = rows[i]["image_path"]
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise RuntimeError(f"读图失败: {p}")
        if g.shape[0] != a.res:
            g = cv2.resize(g, (a.res, a.res), interpolation=cv2.INTER_AREA)
        return i, g

    with ThreadPoolExecutor(a.workers) as pool:
        for n, (i, g) in enumerate(pool.map(rd, range(len(rows)))):
            imgs[i] = g
            if n and n % 10000 == 0:
                print(f"  [preload] {n:,}/{len(rows):,}", flush=True)
    print(f"[preload] {len(rows):,} 张 {a.res}x{a.res} 用时 {time.time()-t0:.0f}s", flush=True)

    rng = np.random.RandomState(1234)
    hold = np.sort(rng.permutation(len(rows))[:int(len(rows) * 0.05)])

    jobs = [("font", lab_f, len(fnt_ids), a.dim_script, fnt_ids, a.k_pick_script),
            ("callig", lab_c, len(cal_ids), a.dim_callig, cal_ids, a.k_pick_callig),
            ("char", lab_h, len(hanzi), a.dim_char, hanzi, a.k_pick_char)]
    if a.only:
        want = {s.strip() for s in a.only.split(",") if s.strip()}
        jobs = [j for j in jobs if j[0] in want]
        print(f"[only] 只训 {sorted(want)}", flush=True)
    logs = {}
    for name, lab, n_cls, dim, ids, kp in jobs:
        mc = a.max_cls_per_step
        # OOM 自动减半重试 (避免一个头炸掉整轮; 每个头训完立即存)
        T = lg = None
        for attempt in range(5):
            try:
                T, lg = train_head(name, imgs, lab, n_cls, dim, res=a.res, steps=a.steps,
                                   k_pick=kp, k_min=a.k_min, lr=a.lr, temp=a.temp,
                                   log_every=a.log_every, center=bool(a.center),
                                   max_cls=mc, init_from=(a.init_from or None),
                                   save_enc=(a.out if a.save_enc else None),
                                   deep=a.deep)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                kp = max(1, kp // 2)
                mc = max(1, mc // 2)
                print(f"[{name}] OOM -> 减半重试 (k_pick={kp}, max_cls={mc})", flush=True)
        if T is None:
            raise SystemExit(f"[FATAL] {name} 连续 OOM")
        np.save(f"{a.out}/{name}_table.npy", T)
        json.dump({"classes": list(ids), "dim": int(dim),
                   "source": f"rgb_top10_centroid_res{a.res}"},
                  open(f"{a.out}/{name}_index.json", "w", encoding="utf-8"),
                  ensure_ascii=False)
        # remap: 原始 id(str) -> 行号。char 是 character_id -> 字形行 (与现有表一致)
        if name == "char":
            rm = {}
            for r in rows + ev:
                rm[str(int(r["character_id"]))] = h_idx[r["character"]]
        else:
            key = "script_id" if name == "font" else "calligrapher_id"
            idx = f_idx if name == "font" else c_idx
            rm = {}
            for r in rows + ev:
                rm[str(int(r[key]))] = idx[int(r[key])]
        json.dump(rm, open(f"{a.out}/{name}_remap.json", "w", encoding="utf-8"),
                  ensure_ascii=False)
        logs[name] = lg
        print(f"[save] {a.out}/{name}_table.npy {T.shape}  classes={len(ids)}  "
              f"remap={len(rm)}", flush=True)

    json.dump({"args": vars(a), "logs": logs},
              open(f"{a.out}/train_log.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n[done] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
