# -*- coding: utf-8 -*-
"""extract_rgb_feats_raw.py — 从 raw 原图**从零**训三个独立编码器, 抽出特征 npz。

为什么要它: tools/build_triple_tables_raw.py 吃"预抽特征 npz"; DINO 那版用
assets/dino_feat_raw.npz, 本脚本产出 assets/rgb_feat_raw.npz (同键), 于是
"dino / raw rgb 各训一次三表"= 同一条代码路径换一个 --npz。

设计 (对齐仓库与用户裁定):
  1) **三个标签空间各自一个编码器**, 互不共享 (联合训练无用且互相干扰: 实测 callig 头
     在共享主干下卡住; 且每图仍要过一次编码器, 并不省算力)。
  2) **正样本 = 同类不同样本** (跨样本), 不是"同一张图的两个视图"。后者等于逼编码器
     抹掉粗细/形变, 而书家风格恰恰是粗细 -> 目标函数会搞反。
  3) **K>=4 的类参与训练**; 未参与训练的类在下游表里用投影质心兜底。
  4) 图**常驻 RAM**(uint8, 393k x 96 x 96 ~ 3.4G), pinned 预取 + 异步 H2D, 每步形状恒定。
  5) batch 启动时标定到 --target-gb (默认 20G), 之后锁定。

用法:
  PYTHONPATH=. python tools/extract_rgb_feats_raw.py                 # 全量
  PYTHONPATH=. python tools/extract_rgb_feats_raw.py --limit 20000 --steps 30   # 冒烟
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
os.chdir(os.environ.get("DIT_ROOT", "/root/Workspace/xy/DiT"))

RAW = "/root/Workspace/xy/UNIFIED_RAW"
CSV_RAW = f"{RAW}/meta/train_clean.csv"
DEV = "cuda"


class Enc(nn.Module):
    """小 CNN: 1x96x96 -> 256。每个标签空间各训一个, 不共享。"""

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


def sample_idx(padded, cnt, sel, k, rng):
    """每类取 k 个互不重复样本 -> 恒定 S*k 个索引 (无效列随机值置 2.0 排到末尾)。"""
    take = np.minimum(cnt[sel], k)
    maxc = padded.shape[1]
    r = rng.rand(len(sel), maxc)
    r = np.where(np.arange(maxc)[None, :] < take[:, None], r, np.float64(2.0))
    return padded[sel[:, None], np.argsort(r, axis=1)[:, :k]].ravel()


class Head:
    """一个标签空间的独立编码器 (自带 RAM 图库引用与采样器)。"""

    def __init__(self, name, imgs_cpu, lab, n_cls, k, res):
        self.name = name
        self.imgs = imgs_cpu
        self.lab = lab
        self.res = res
        self.k = k
        self.n_cls = n_cls
        by = [[] for _ in range(n_cls)]
        for i, g in enumerate(lab.tolist()):
            by[int(g)].append(i)
        maxc = max((len(v) for v in by), default=1)
        self.padded = np.zeros((n_cls, maxc), np.int64)
        self.cnt = np.zeros(n_cls, np.int64)
        for c, v in enumerate(by):
            self.cnt[c] = len(v)
            if v:
                self.padded[c, :len(v)] = v
        self.elig = np.where(self.cnt >= k)[0].astype(np.int64)
        self.rng = np.random.RandomState(0)
        self.n_max = max(1, len(self.elig))

    def indices(self, n_pick):
        sel = self.rng.choice(self.elig, size=min(n_pick, self.n_max), replace=False)
        return sample_idx(self.padded, self.cnt, sel, self.k, self.rng)


def calibrate(make_step, cands, target_gb, name):
    """从小到大实跑, 取峰值<=target 的最大档。make_step(B)->peak_gb。"""
    best = None
    for B in cands:
        try:
            g = make_step(B)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower():
                raise
            print(f"  [{name}] B={B} OOM -> 停在 {best}", flush=True)
            torch.cuda.empty_cache()
            break
        ok = g <= target_gb
        print(f"  [{name}] B={B:6d} 峰值 {g:5.2f}G {'✓ 采用' if ok else '✗ 回退'}", flush=True)
        torch.cuda.empty_cache()
        if not ok:
            break
        best = B
    assert best, f"[{name}] 最小档都超预算"
    return best


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--csv", default=CSV_RAW)
    a.add_argument("--img-root", default=RAW)
    a.add_argument("--out", default="assets/triple_tables_big_rgb")
    a.add_argument("--res", type=int, default=96)
    a.add_argument("--dim-script", type=int, default=32, help="= 下游 script_embed_dim")
    a.add_argument("--dim-callig", type=int, default=128, help="= 下游 callig_embed_dim")
    a.add_argument("--dim-char", type=int, default=384, help="= 下游 char_embed_dim")
    a.add_argument("--steps", type=int, default=1500)
    a.add_argument("--k", type=int, default=4, help="每类采样数, 也是参与训练的最少样本数")
    a.add_argument("--lr", type=float, default=1e-3)
    a.add_argument("--temp", type=float, default=0.07)
    a.add_argument("--workers", type=int, default=32)
    a.add_argument("--target-gb", type=float, default=20.0)
    a.add_argument("--batch-script", type=int, default=0, help="0=标定; 书体只 12 类")
    a.add_argument("--limit", type=int, default=0)
    a.add_argument("--log-every", type=int, default=250)
    a.add_argument("--seed", type=int, default=0)
    a = a.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    if a.limit:
        rows = rows[:a.limit]
    n_scr = max(int(r["script_id"]) for r in rows) + 1
    n_cal = max(int(r["calligrapher_id"]) for r in rows) + 1
    n_chr = max(int(r["character_id"]) for r in rows) + 1
    ids = np.array([int(r["img_id"]) for r in rows], dtype=np.int64)
    lab_s = np.array([int(r["script_id"]) for r in rows], dtype=np.int64)
    lab_g = np.array([int(r["calligrapher_id"]) for r in rows], dtype=np.int64)
    lab_c = np.array([int(r["character_id"]) for r in rows], dtype=np.int64)
    print(f"[data] {len(rows):,} 行  script={n_scr} callig={n_cal} char={n_chr}", flush=True)

    # ── 预载图到 RAM (多线程; 每个 worker 关掉 cv2 线程池) ──
    t0 = time.time()
    import cv2
    imgs = np.zeros((len(rows), a.res, a.res), np.uint8)

    def rd(i):
        cv2.setNumThreads(1)
        p = rows[i]["image_path"]
        full = p if os.path.isabs(p) else os.path.join(a.img_root, p)
        g = cv2.imread(full, cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise RuntimeError(f"读图失败: {full}")
        if g.shape[0] != a.res:
            g = cv2.resize(g, (a.res, a.res), interpolation=cv2.INTER_AREA)
        return i, g

    with ThreadPoolExecutor(a.workers) as pool:
        for n, (i, g) in enumerate(pool.map(rd, range(len(rows)))):
            imgs[i] = g
            if n and n % 100000 == 0:
                print(f"  [preload] {n:,}/{len(rows):,} {n/(time.time()-t0):.0f} img/s",
                      flush=True)
    imgs_cpu = torch.from_numpy(imgs)
    print(f"[preload] {len(rows):,} 张 {a.res}x{a.res} 用时 {time.time()-t0:.0f}s "
          f"(RAM {imgs.nbytes/1e9:.2f}G)", flush=True)

    heads = {"script": Head("script", imgs_cpu, torch.from_numpy(lab_s), n_scr, a.k, a.res),
             "callig": Head("callig", imgs_cpu, torch.from_numpy(lab_g), n_cal, a.k, a.res),
             "char": Head("char", imgs_cpu, torch.from_numpy(lab_c), n_chr, a.k, a.res)}
    names_map = {}
    for r in rows:
        names_map["script"] = names_map.get("script", {})
        names_map["callig"] = names_map.get("callig", {})
        names_map["char"] = names_map.get("char", {})
        names_map["script"][int(r["script_id"])] = r["script"]
        names_map["callig"][int(r["calligrapher_id"])] = r["calligrapher"]
        names_map["char"][int(r["character_id"])] = r["character"]

    dims = {"script": a.dim_script, "callig": a.dim_callig, "char": a.dim_char}
    lab_of = {"script": lab_s, "callig": lab_g, "char": lab_c}
    for nm in ("script", "callig", "char"):
        h = heads[nm]
        dim = dims[nm]
        print(f"\n[rgb/{nm}] 类数 {h.n_cls}, K>={a.k} 参与 {len(h.elig)}, "
              f"每步每类 {a.k} 张, 目标维度 {dim}", flush=True)
        Bmax = 32768
        pin = [torch.empty((Bmax, a.res, a.res), dtype=torch.uint8, pin_memory=True)
               for _ in range(2)]
        gbuf = [torch.empty((Bmax, 1, a.res, a.res), dtype=torch.uint8, device=DEV)
                for _ in range(2)]
        evs = [torch.cuda.Event(), torch.cuda.Event()]
        h2d = torch.cuda.Stream()

        def prepare(buf, n_pick):
            idx = h.indices(n_pick)
            n = len(idx)
            pin[buf][:n].copy_(imgs_cpu[idx])
            with torch.cuda.stream(h2d):
                gbuf[buf][:n, 0].copy_(pin[buf][:n], non_blocking=True)
            evs[buf].record(h2d)
            return idx, n

        net = Enc(dim).to(DEV).to(memory_format=torch.channels_last)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.steps),
                                                           eta_min=a.lr * 0.05)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        lab_gpu = h.lab.to(DEV)

        def core(buf, idx, n):
            torch.cuda.current_stream().wait_event(evs[buf])
            x = gbuf[buf][:n].float().div_(255.0).contiguous(memory_format=torch.channels_last)
            y = lab_gpu[torch.from_numpy(idx).to(DEV)]
            with torch.cuda.amp.autocast(enabled=True):
                zn = F.normalize(net(affine(x).float()), dim=-1)
                logits = (zn @ zn.t()) / a.temp
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
            return loss

        def make_probe(step_fn):
            def probe(B):
                n_pick = max(1, B // a.k)
                net.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                for _ in range(2):
                    idx, n = prepare(0, n_pick)
                    step_fn(0, idx, n)
                torch.cuda.synchronize()
                return torch.cuda.max_memory_allocated() / 1e9
            return probe

        cands = [c * a.k for c in (1024, 2048, 4096, 8192) if c * a.k <= Bmax]
        cands = [c for c in cands if c // a.k <= len(h.elig)]
        if not cands:
            cands = [min(len(h.elig), Bmax // a.k) * a.k]
        batch = calibrate(make_probe(core), cands, a.target_gb, f"rgb/{nm}")
        n_pick = max(1, min(batch // a.k, len(h.elig)))
        print(f"  [rgb/{nm}] 锁定 batch = {n_pick} 类 x {a.k} = {n_pick*a.k} 图/步",
              flush=True)

        t1 = time.time()
        torch.cuda.reset_peak_memory_stats()
        idx_cur, n_cur = prepare(0, n_pick)
        cur, nxt = 0, 1
        net.train()
        for step in range(1, a.steps + 1):
            if step < a.steps:
                idx_nxt, n_nxt = prepare(nxt, n_pick)
            loss = core(cur, idx_cur, n_cur)
            if step < a.steps:
                cur, nxt = nxt, cur
                idx_cur, n_cur = idx_nxt, n_nxt
            if step % a.log_every == 0 or step == 1 or step == a.steps:
                ips = step * n_pick * a.k / max(time.time() - t1, 1e-9)
                print(f"  [rgb/{nm} {step:5d}/{a.steps}] loss {loss.item():.4f} | "
                      f"{ips:6.0f} 图/s | 峰值 "
                      f"{torch.cuda.max_memory_allocated()/1e9:.2f}G", flush=True)

        # ── 表 = 该编码器空间里每类的**归一化质心** (与 DINO 臂的"类原型"同义) ──
        net.eval()
        outs, lab_l = [], lab_of[nm]
        ECH = min(2048, Bmax)                       # 抽取批固定小值: no_grad 下显存很小
        with torch.no_grad():
            for b in range(0, len(rows), ECH):
                nb = min(ECH, len(rows) - b)
                pin[0][:nb].copy_(imgs_cpu[b:b + nb])
                x = pin[0][:nb].unsqueeze(1).to(DEV, non_blocking=True).float().div_(255.0)
                outs.append(F.normalize(net(x).float(), dim=-1))
        Z = torch.cat(outs)                                    # (N, dim) 已归一化
        lab_t = torch.from_numpy(lab_l).to(DEV)
        csum = torch.zeros(h.n_cls, dim, device=DEV)
        ccnt = torch.zeros(h.n_cls, 1, device=DEV)
        csum.index_add_(0, lab_t, Z)
        ccnt.index_add_(0, lab_t, torch.ones(lab_t.size(0), 1, device=DEV))
        has = (ccnt.squeeze(1) > 0)
        T = torch.zeros(h.n_cls, dim, device=DEV)
        T[has] = F.normalize(csum[has] / ccnt[has].clamp_min(1), dim=-1)
        # 无样本类留零行 (下游不会用到; 若有则报数)
        print(f"  [rgb/{nm}] 表 {tuple(T.shape)}; 无样本类 {int((~has).sum())}", flush=True)

        # 自检: 留一法近邻 top-1 (不是训练指标, 是"这张表能不能查得准")
        with torch.no_grad():
            ok_c = np.where(has.cpu().numpy())[0]
            Tn = F.normalize(T[ok_c], dim=-1).cpu().numpy()
            Zn = Z.cpu().numpy()
            hit = tot = 0
            for c in ok_c:
                m = np.where(lab_l == c)[0]
                if len(m) < 2:
                    continue
                q = Zn[m[-1]]
                p = int((Tn @ q).argmax())
                hit += int(ok_c[p] == c)
                tot += 1
            loo = hit / max(1, tot)
        print(f"  [rgb/{nm}] 自检(留一近邻 top-1, 只算 K>=2 的类) = {loo:.4f} (n={tot} 类)",
              flush=True)
        # 分离度
        with torch.no_grad():
            e = torch.from_numpy(h.elig).to(DEV)
            w = F.normalize(T[e], dim=-1)
            cos = w @ w.t()
            off = ~torch.eye(len(e), dtype=torch.bool, device=DEV)
            sep = dict(mean_abs_cos=float(cos[off].abs().mean()),
                       max_cos=float(cos[off].max()),
                       share_gt_015=float((cos[off] > 0.15).float().mean()))
        print(f"  [rgb/{nm}] 表行|cos| 均值={sep['mean_abs_cos']:.4f} "
              f"max={sep['max_cos']:.4f} >0.15 占比={sep['share_gt_015']:.4f}", flush=True)

        Tn_np = T.cpu().numpy().astype(np.float32)
        np.save(f"{a.out}/{nm}_table.npy", Tn_np)
        json.dump({"classes": [names_map[nm].get(i, "") for i in range(h.n_cls)],
                   "dim": int(dim), "source": "rgb_encoder_centroid"},
                  open(f"{a.out}/{nm}_index.json", "w", encoding="utf-8"),
                  ensure_ascii=False)
        json.dump({str(i): i for i in range(h.n_cls)},
                  open(f"{a.out}/{nm}_identity_remap.json", "w", encoding="utf-8"),
                  ensure_ascii=False)
        json.dump({"dim": int(dim), "n_rows": int(h.n_cls), "n_trained": int(len(h.elig)),
                   "loo_top1": float(loo), "sep": sep, "steps": a.steps,
                   "batch_imgs": int(n_pick * a.k)},
                  open(f"{a.out}/log_{nm}.json", "w", encoding="utf-8"), ensure_ascii=False)
        torch.save(net.state_dict(), f"{a.out}/encoder_{nm}.pt")
        del net, opt, Z
        torch.cuda.empty_cache()

    print(f"\n[done] 三张表 -> {a.out}/  (script {a.dim_script}d / callig "
          f"{a.dim_callig}d / char {a.dim_char}d)", flush=True)


if __name__ == "__main__":
    main()
