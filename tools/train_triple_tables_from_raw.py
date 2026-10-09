# -*- coding: utf-8 -*-
"""train_triple_tables_from_raw.py — 直接从 raw 图像对比学习三张表（无任何预训练）。

为什么不用 DINO 特征: 仓库自己的诊断 (docs/system/12_dino_diagnosis_20260829.md) 记载
  冻结 DINO 表的**有效秩只有 34.1/384**（PC1 占 26.3%），且"书体"信号远多于"字符身份"。
  DINOv2 追"对姿态/光照/纹理不变"，而汉字身份恰靠高频拓扑 -> 不必要的信息扭曲。

数据: /root/Workspace/xy/UNIFIED_RAW/imgs (393,486) + meta/train_clean.csv
      标签空间: 书体 12 / 书家 2,132 / 汉字 9,130

v3 关键设计 (2026-10-07 用户裁定: 显存打到 20G + 所有书家都要用到 + 把书家充分推开):
  1) 从零小 CNN 编码器 (1x96x96 -> 32/64/128/256 -> GAP -> 256) + 三个线性投影
  2) **全类覆盖采样** sample_balanced: 当 n >= 类数时保证每类至少 1 张
       callig 头 n = 全部 2,109 类 -> 每步每个书家都进 (不再因 K>=8 丢掉 2/3 的书家)
  3) **双视图互为正** (替代凑 K 的同类): 同一张图的两个视图构成正对
       char   头: view A + (A + 笔画粗细抖动)   -> 逼出"粗细不变、字形不变"
       callig 头: view A + view B (都是纯几何) -> **不扰粗细**(粗细是书家风格, 必须保留)
       script 头: 同 callig
  4) bf16 autocast + 大步批 (char 2048 / callig 2109 / script 768, 各 x2 视图) 吃满 ~20G
  5) 表 = 类质心: 训完在训练集上跑一遍编码器, 每类取归一化嵌入均值再归一化

用法:
  PYTHONPATH=. python tools/train_triple_tables_from_raw.py --limit 20000 --steps 60   # 冒烟
  PYTHONPATH=. python tools/train_triple_tables_from_raw.py --steps 8000               # 全量
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")


def supcon(z, y, temp=0.07):
    """z: (N,d) 已 L2 归一化; y: (N,)。标准 SupCon。无正样本的锚点自动跳过。"""
    N = z.size(0)
    if N < 2:
        return z.sum() * 0.0
    sim = (z @ z.t()) / temp
    notself = ~torch.eye(N, dtype=torch.bool, device=z.device)
    pos = (y[:, None] == y[None, :]) & notself
    pc = pos.sum(1)
    valid = pc > 0
    if not valid.any():
        return z.sum() * 0.0
    sim = sim - sim.max(1, keepdim=True).values.detach()
    exp = torch.exp(sim) * notself
    logp = sim - torch.log(exp.sum(1, keepdim=True) + 1e-12)
    return -((logp * pos).sum(1)[valid] / pc[valid].float()).mean()


class Net(nn.Module):
    def __init__(self, emb=256, d_s=16, d_g=32, d_h=256):
        super().__init__()

        def blk(i, o, s=2):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False),
                                 nn.BatchNorm2d(o), nn.SiLU())
        self.enc = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), blk(128, 256),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                 nn.Linear(256, emb, bias=False), nn.BatchNorm1d(emb))
        self.ps = nn.Linear(emb, d_s, bias=False)   # script
        self.pg = nn.Linear(emb, d_g, bias=False)   # callig
        self.ph = nn.Linear(emb, d_h, bias=False)   # char
        for m in (self.ps, self.pg, self.ph):
            nn.init.orthogonal_(m.weight)

    def forward(self, x):
        z = self.enc(x)
        return (F.normalize(self.ps(z), dim=-1),
                F.normalize(self.pg(z), dim=-1),
                F.normalize(self.ph(z), dim=-1))


def affine(x, rot_deg=4.0, scale=0.05, shift=3.0):
    N = x.size(0); dev = x.device
    ang = (torch.rand(N, device=dev) * 2 - 1) * (rot_deg * np.pi / 180.0)
    sc = 1.0 + (torch.rand(N, device=dev) * 2 - 1) * scale
    tx = (torch.rand(N, device=dev) * 2 - 1) * (shift / x.size(2)) * 2
    ty = (torch.rand(N, device=dev) * 2 - 1) * (shift / x.size(3)) * 2
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    th = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    return F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False),
                         align_corners=False, padding_mode="border")


def width_jitter(x):
    """随机阈值二值化 -> 改变笔画粗细 (1=白底, 0=墨)。只在 char 头当正样本用。"""
    tau = 0.35 + 0.30 * torch.rand(x.size(0), 1, 1, 1, device=x.device)
    return 1.0 - (x < tau).float()


class Centroids:
    def __init__(self, n, d):
        self.sum = np.zeros((n, d), np.float64)
        self.cnt = np.zeros(n, np.int64)

    def add(self, z, y):
        z = z.detach().float().cpu().numpy().reshape(len(z), -1)
        y = y.detach().cpu().numpy().reshape(-1).astype(np.int64)
        if z.shape[0] != y.shape[0]:
            raise RuntimeError(f"Centroids.add 形状不匹配: z{z.shape} y{y.shape}")
        for c in np.unique(y):
            m = (y == c)
            self.sum[c] += z[m].sum(0)
            self.cnt[c] += int(m.sum())

    def table(self, min_cnt=1):
        m = self.cnt >= min_cnt
        if not m.any():
            return None, None
        t = self.sum[m] / self.cnt[m, None]
        t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
        return t.astype(np.float32), np.where(m)[0]


def sample_balanced(keys, cls, n, rng):
    """全类覆盖采样: n >= 类数时保证**每类至少 1 张**（余量再按类补）。"""
    if n >= len(keys):
        per = max(1, n // len(keys))
        out = []
        for k in keys:
            v = cls[k]
            sel = rng.choice(len(v), size=min(per, len(v)), replace=False)
            out += [v[j] for j in sel]
        while len(out) < n:                       # 余量: 随机补
            k = keys[rng.randint(len(keys))]
            v = cls[k]
            out.append(v[rng.randint(len(v))])
        return np.array(out[:n])
    ch = rng.choice(len(keys), size=n, replace=False)
    return np.array([cls[keys[k]][rng.randint(len(cls[keys[k]]))] for k in ch])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/root/Workspace/xy/UNIFIED_RAW/meta/train_clean.csv")
    ap.add_argument("--img-root", default="/root/Workspace/xy/UNIFIED_RAW")
    ap.add_argument("--out", default="assets/triple_tables_raw")
    ap.add_argument("--res", type=int, default=96)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--n-char", type=int, default=2048, help="char 头每步图数 (>=类数则全类覆盖)")
    ap.add_argument("--n-callig", type=int, default=0, help="0 = 全部书家类各 1 张")
    ap.add_argument("--n-script", type=int, default=768)
    ap.add_argument("--d-char", type=int, default=256)
    ap.add_argument("--d-callig", type=int, default=64)
    ap.add_argument("--d-script", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--w-char", type=float, default=1.0)
    ap.add_argument("--w-callig", type=float, default=1.0)
    ap.add_argument("--w-script", type=float, default=1.0)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--holdout", type=float, default=0.02)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    print(f"[cfg] res={a.res} steps={a.steps} n_char={a.n_char} n_callig={a.n_callig} "
          f"n_script={a.n_script} dims char{a.d_char}/callig{a.d_callig}/script{a.d_script} "
          f"amp={a.amp} lr={a.lr}", flush=True)

    rows, cstr, gstr, sstr = [], {}, {}, {}
    with open(a.csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cid, gid, sid = (int(r["character_id"]), int(r["calligrapher_id"]), int(r["script_id"]))
            cstr[cid] = r["character"]; gstr[gid] = r["calligrapher"]; sstr[sid] = r["script"]
            rows.append((r["image_path"], sid, gid, cid))
    if a.limit:
        rows = rows[:a.limit]
    n_s, n_g, n_c = max(sstr) + 1, max(gstr) + 1, max(cstr) + 1
    print(f"[data] {len(rows):,} 行  script={n_s} callig={n_g} char={n_c}", flush=True)

    import cv2
    t0 = time.time()
    imgs = np.zeros((len(rows), a.res, a.res), np.uint8)
    pool = ThreadPoolExecutor(a.workers)

    def rd(i):
        p = rows[i][0]
        full = p if os.path.isabs(p) else os.path.join(a.img_root, p)
        g = cv2.imread(full, cv2.IMREAD_GRAYSCALE)
        if g is None:
            return i, np.zeros((a.res, a.res), np.uint8)
        if g.shape[0] != a.res:
            g = cv2.resize(g, (a.res, a.res), interpolation=cv2.INTER_AREA)
        return i, g

    for n, (i, g) in enumerate(pool.map(rd, range(len(rows)))):
        imgs[i] = g
        if n % 100000 == 0 and n:
            print(f"  [preload] {n:,}/{len(rows):,} {n/(time.time()-t0):.0f} img/s", flush=True)
    print(f"[preload] {len(rows):,} 张 {a.res}x{a.res} 用时 {time.time()-t0:.0f}s", flush=True)

    rng = np.random.RandomState(a.seed)
    perm = rng.permutation(len(rows)); n_ho = int(len(rows) * a.holdout)
    ho = set(perm[:n_ho].tolist())
    tr = [i for i in range(len(rows)) if i not in ho]

    def cls_index(idxs, k):
        d = {}
        for i in idxs:
            d.setdefault(rows[i][k], []).append(i)
        return d

    cls_char, cls_call, cls_scr = cls_index(tr, 3), cls_index(tr, 2), cls_index(tr, 1)
    keys_char, keys_call, keys_scr = list(cls_char), list(cls_call), list(cls_scr)
    n_call_used = a.n_callig if a.n_callig > 0 else len(keys_call)
    print(f"[采样] callig 全类 {len(keys_call)} 个 -> n_callig={n_call_used}"
          f" ({'全类覆盖' if n_call_used >= len(keys_call) else '部分覆盖'})", flush=True)
    print(f"       char  {len(keys_char)} 类 -> 每步 {a.n_char} 张 (轮转覆盖)", flush=True)

    def to_gpu(idx):
        return torch.from_numpy(imgs[idx].astype(np.float32) / 255.0).unsqueeze(1).to(dev)

    def labels(idx, k):
        return torch.tensor([rows[i][k] for i in idx], dtype=torch.long, device=dev)

    net = Net(256, a.d_script, a.d_callig, a.d_char).to(dev)
    print(f"[model] {sum(p.numel() for p in net.parameters())/1e6:.2f}M 参数", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=a.lr * 0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(a.amp))

    C_s = Centroids(n_s, a.d_script); C_g = Centroids(n_g, a.d_callig); C_c = Centroids(n_c, a.d_char)
    ho_idx = np.array(sorted(ho))
    ho_x = torch.from_numpy(imgs[ho_idx].astype(np.float32) / 255.0).unsqueeze(1).to(dev)

    @torch.no_grad()
    def evaluate():
        net.eval()
        out = {}
        tabs = {k: C.table()[0] for k, C in (("s", C_s), ("g", C_g), ("c", C_c))}
        keep = {k: C.table()[1] for k, C in (("s", C_s), ("g", C_g), ("c", C_c))}
        hit = {k: 0 for k in "sgc"}; tot = {k: 0 for k in "sgc"}; cov = {k: 0 for k in "sgc"}
        for b in range(0, len(ho_idx), 1024):
            x = ho_x[b:b + 1024]
            if x.size(0) == 0:
                continue
            zs, zg, zh = net(x)
            for k, zz, col in (("s", zs, 1), ("g", zg, 2), ("c", zh, 3)):
                if tabs[k] is None:
                    continue
                tgt = labels(ho_idx[b:b + 1024], col)
                mx = torch.from_numpy(keep[k]).to(dev)
                sel = torch.isin(tgt, mx)
                if sel.sum() == 0:
                    continue
                pred = (zz[sel] @ torch.from_numpy(tabs[k]).to(dev).t()).argmax(1)
                hit[k] += int((mx[pred] == tgt[sel]).sum()); tot[k] += int(sel.sum())
                cov[k] += int(sel.sum())
        net.train()
        for k in "sgc":
            out[k] = hit[k] / max(1, tot[k]); out[k + "_cov"] = cov[k] / max(1, len(ho_idx))
        out["_used"] = {k: len(C.table()[1]) if C.table()[1] is not None else 0
                        for k, C in (("s", C_s), ("g", C_g), ("c", C_c))}
        return out

    ev = None
    t0 = time.time()
    for step in range(1, a.steps + 1):
        # ---- 三头各自采样 + 双视图 (同类两视图互为正) ----
        i_c = sample_balanced(keys_char, cls_char, a.n_char, rng)
        i_g = sample_balanced(keys_call, cls_call, n_call_used, rng)
        i_s = sample_balanced(keys_scr, cls_scr, a.n_script, rng)

        xc, xg, xs = to_gpu(i_c), to_gpu(i_g), to_gpu(i_s)
        vc = torch.cat([affine(xc), width_jitter(affine(xc))], 0)   # char: 含粗细抖动
        vg = torch.cat([affine(xg), affine(xg)], 0)                 # callig: 纯几何
        vs = torch.cat([affine(xs), affine(xs)], 0)                 # script: 纯几何
        y_c = torch.cat([labels(i_c, 3)] * 2, 0)
        y_g = torch.cat([labels(i_g, 2)] * 2, 0)
        y_s = torch.cat([labels(i_s, 1)] * 2, 0)

        with torch.cuda.amp.autocast(enabled=bool(a.amp)):
            sC, _, hC = net(vc)
            _, gG, _ = net(vg)
            sS, _, _ = net(vs)
            L_char = supcon(hC, y_c, a.temp)
            L_call = supcon(gG, y_g, a.temp)
            L_scr = supcon(sS, y_s, a.temp)
            loss = a.w_char * L_char + a.w_callig * L_call + a.w_script * L_scr

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); sched.step()

        with torch.no_grad():
            C_s.add(sS, y_s); C_g.add(gG, y_g); C_c.add(hC, y_c)

        if step % 50 == 0 or step == 1:
            el = time.time() - t0
            it = step / max(el, 1e-9)
            gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"  [step {step}/{a.steps}] loss {loss.item():.4f} "
                  f"(char {L_char.item():.3f} call {L_call.item():.3f} scr {L_scr.item():.3f}) "
                  f"{it:.2f} it/s 峰值显存 {gb:.1f}G  ETA {(a.steps-step)/it/60:.0f}min", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate()
            print(f"  [eval {step}] script {ev['s']:.4f}(cov{ev['s_cov']:.2f}) | "
                  f"callig {ev['g']:.4f}(cov{ev['g_cov']:.2f}) | "
                  f"char {ev['c']:.4f}(cov{ev['c_cov']:.2f})  used={ev['_used']}", flush=True)

    print("[final] 训练集全量前向算类质心 ...", flush=True)
    Cf_s = Centroids(n_s, a.d_script); Cf_g = Centroids(n_g, a.d_callig); Cf_c = Centroids(n_c, a.d_char)
    tr_idx = np.array(tr)
    net.eval()
    with torch.no_grad():
        for b in range(0, len(tr_idx), 1024):
            chunk = tr_idx[b:b + 1024]
            x = torch.from_numpy(imgs[chunk].astype(np.float32) / 255.0).unsqueeze(1).to(dev)
            zs, zg, zh = net(x)
            Cf_s.add(zs, torch.tensor([rows[i][1] for i in chunk], dtype=torch.long))
            Cf_g.add(zg, torch.tensor([rows[i][2] for i in chunk], dtype=torch.long))
            Cf_c.add(zh, torch.tensor([rows[i][3] for i in chunk], dtype=torch.long))
    for nm, Cf, fn, dim in (("script", Cf_s, "script_table.npy", a.d_script),
                            ("callig", Cf_g, "callig_table.npy", a.d_callig),
                            ("char", Cf_c, "char_table.npy", a.d_char)):
        T, keep = Cf.table(min_cnt=1)
        full = np.zeros((Cf.cnt.shape[0], dim), np.float32)
        if T is not None:
            full[keep] = T
        np.save(f"{a.out}/{fn}", full)
        print(f"  {fn} {full.shape}  有效类 {0 if T is None else len(keep)}/{Cf.cnt.shape[0]}", flush=True)

    json.dump({"classes": [sstr[i] for i in sorted(sstr)], "dim": a.d_script},
              open(f"{a.out}/script_index.json", "w", encoding="utf-8"), ensure_ascii=False)
    json.dump({"classes": [gstr[i] for i in sorted(gstr)], "dim": a.d_callig},
              open(f"{a.out}/callig_index.json", "w", encoding="utf-8"), ensure_ascii=False)
    json.dump({"classes": [cstr[i] for i in sorted(cstr)], "dim": a.d_char},
              open(f"{a.out}/char_index.json", "w", encoding="utf-8"), ensure_ascii=False)
    torch.save(net.state_dict(), f"{a.out}/encoder.pt")
    json.dump({"args": vars(a), "final_eval": ev, "n_script": n_s, "n_callig": n_g, "n_char": n_c},
              open(f"{a.out}/train_log.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[done] -> {a.out}  总用时 {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
