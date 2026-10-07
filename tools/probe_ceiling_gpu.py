#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/probe_ceiling_gpu.py — 全数据集穷举「参考上界 / 学习上限」，**GPU 解码域**版。

自包含 (不依赖本仓库 src/), 供 48 机器 (dserver) 直接跑:
  /home/ds/miniconda3/envs/pytorch/bin/python tools/probe_ceiling_gpu.py

为什么在**解码域**测:
  我们评测时 GT 也是 VAE 解码出来的 (eval_auto.py: 预编码 latent -> 比对 decoded vs decoded),
  所以只有"解码后"的相似度才是与模型分数同一空间的参照。
  (像素域 vs 解码域只差约 0.001, 见 4090 上的 tools/probe_intraclass_pixelpairs.py)

四个口径:
  A 三键全同 (c,s,ch) 穷举两两       -> "人类自己写两遍同字"的相似度 = 经验参考线
  B 三键全同 留一质心 (条件均值)      -> "只会输出条件均值的完美模型" = (c,s,ch) 下的**学习上限**
  C 跨书家 (s,ch) 采样               -> 未见书家的泛化上界
  D 同字 (ch) 采样                   -> 未见组合的泛化上界
  E 三键全同 按键平均 (防大键主导)

SSIM 实现与评测 src.eval.metrics.ssim 等价 (win=11, sigma=1.5, 逐通道均值), 已核对 |Δ|<5e-5。
"""
import csv
import itertools
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

try:
    from scipy.ndimage import correlate1d
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

ROOT = "/home/ds/Workspace/moyi"
CSV = f"{ROOT}/exp-std-csv/train.csv"
SHARDS = f"{ROOT}/data/top10_style23/shards_img"
VAE = f"{ROOT}/models/sd-vae-ft-ema"
OUT = f"{ROOT}/results/ceiling_probe_summary.json"

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------- SSIM (与 src.eval.metrics.ssim 等价) ----------------
def _sep(a, k, axis):
    if HAVE_SCIPY:
        return correlate1d(a, k, axis=axis, mode="reflect")
    pad = len(k) // 2
    pads = [(pad, pad) if i == axis else (0, 0) for i in range(a.ndim)]
    ap = np.pad(a, pads, mode="reflect")
    out = np.zeros_like(a, dtype=np.float64)
    for i, w in enumerate(k):
        sl = [slice(None)] * a.ndim
        sl[axis] = slice(i, i + a.shape[axis])
        out += w * ap[tuple(sl)]
    return out


_K = None


def ssim(pred, gt, win=11, sigma=1.5, dr=1.0):
    global _K
    if _K is None:
        r = win // 2
        x = np.arange(-r, r + 1, dtype=np.float64)
        k = np.exp(-(x ** 2) / (2 * sigma ** 2))
        _K = (k / k.sum(), (0.01 * dr) ** 2, (0.03 * dr) ** 2)
    k, c1, c2 = _K

    def g(img):
        return _sep(_sep(img, k, 0), k, 1)

    out = []
    for ch in range(pred.shape[2]):
        xx = np.ascontiguousarray(pred[:, :, ch], dtype=np.float64)
        yy = np.ascontiguousarray(gt[:, :, ch], dtype=np.float64)
        ux, uy = g(xx), g(yy)
        ux2, uy2, uxy = ux ** 2, uy ** 2, ux * uy
        sx2, sy2, sxy = g(xx * xx) - ux2, g(yy * yy) - uy2, g(xx * yy) - uxy
        m = ((2 * uxy + c1) * (2 * sxy + c2)) / ((ux2 + uy2 + c1) * (sx2 + sy2 + c2))
        out.append(float(m.mean()))
    return float(np.mean(out))


def ink_iou(pred, gt, thresh=0.5):
    b1 = pred.mean(axis=-1) < thresh
    b2 = gt.mean(axis=-1) < thresh
    u = float((b1 | b2).sum())
    return 1.0 if u == 0 else float((b1 & b2).sum()) / u


def desc(tag, a, out):
    a = np.asarray(a, dtype=float)
    if not len(a):
        return
    q = np.percentile(a, [5, 25, 50, 75, 95])
    d = {"n": int(len(a)), "mean": float(a.mean()), "median": float(q[2]),
         "std": float(a.std()), "p5": float(q[0]), "p25": float(q[1]),
         "p75": float(q[3]), "p95": float(q[4]),
         "min": float(a.min()), "max": float(a.max())}
    out[tag] = d
    print(f"  {tag:<30} n={d['n']:>6}  均值 {d['mean']:.4f}  中位 {q[2]:.4f}  std {d['std']:.4f}")
    print(f"  {'':<30} p5 {q[0]:.4f}  p25 {q[1]:.4f}  p75 {q[3]:.4f}  p95 {q[4]:.4f}  "
          f"min {d['min']:.4f} max {d['max']:.4f}")


def main():
    t0 = time.time()
    print(f"scipy={HAVE_SCIPY}")
    import torch
    from diffusers.models import AutoencoderKL
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} device={dev} "
          f"{torch.cuda.get_device_name(0) if dev=='cuda' else ''}")

    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))

    def grp(cols):
        g = defaultdict(list)
        for r in rows:
            g[tuple(r[c] for c in cols)].append(int(r["img_id"]))
        return {k: v for k, v in g.items() if len(v) >= 2}

    g3 = grp(["calligrapher", "script", "character"])
    g2 = grp(["script", "character"])
    g1 = grp(["character"])
    n3 = sum(len(v) * (len(v) - 1) // 2 for v in g3.values())
    print(f"键: (c,s,ch)={len(g3):,}(穷举 {n3:,} 对)  (s,ch)={len(g2):,}  (ch)={len(g1):,}")

    need = set()
    for g in (g3, g2, g1):
        for v in g.values():
            need.update(v)
    print(f"需解码 {len(need):,} 张 latent ...")

    # ---- 载入 latents ----
    want = {str(i): None for i in need}
    import glob
    for sf in sorted(glob.glob(f"{SHARDS}/shard_*.npz")):
        with np.load(sf) as d:
            lats, ids = d["latents"], d["img_ids"]
            for j, iid in enumerate(ids):
                s = str(int(iid))
                if s in want and want[s] is None:
                    want[s] = lats[j]
        if all(v is not None for v in want.values()):
            break
    got = {k: v for k, v in want.items() if v is not None}
    print(f"命中 {len(got):,}/{len(need):,}  ({time.time()-t0:.0f}s)")

    # ---- GPU 解码 -> uint8 像素 ----
    vae = AutoencoderKL.from_pretrained(VAE).to(dev).eval()
    pix = {}
    keys = list(got.keys())
    B = 64
    t1 = time.time()
    with torch.no_grad():
        for i in range(0, len(keys), B):
            ks = keys[i:i + B]
            z = torch.from_numpy(np.stack([got[k] for k in ks]).astype(np.float32)).to(dev)
            dec = vae.decode(z / 0.18215).sample
            dec = torch.clamp((dec + 1.0) / 2.0, 0.0, 1.0)
            arr = (dec.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
            for k, a in zip(ks, arr):
                pix[k] = a
            if i % (B * 40) == 0:
                print(f"  解码 {i+len(ks):,}/{len(keys):,}  ({time.time()-t1:.0f}s)")
    del vae
    if dev == "cuda":
        torch.cuda.empty_cache()
    print(f"解码完成 {len(pix):,} 张 ({time.time()-t1:.0f}s)")

    def P(i):                       # img_id -> float 图
        return pix[str(i)].astype(np.float32) / 255.0

    S = {}                          # 结果容器

    # ---- A/B/E: 三键全同 穷举 ----
    print("\n[A/B/E] 三键全同穷举 ...")
    A, Ai, Bm, Bi, Kw = [], [], [], [], []
    t2 = time.time()
    for ki, (key, ids) in enumerate(sorted(g3.items())):
        ids = [i for i in ids if str(i) in pix]
        if len(ids) < 2:
            continue
        imgs = [P(i) for i in ids]
        loc = []
        for x, y in itertools.combinations(range(len(imgs)), 2):
            s = ssim(imgs[x], imgs[y])
            A.append(s)
            Ai.append(ink_iou(imgs[x], imgs[y]))
            loc.append(s)
        Kw.append(float(np.mean(loc)))
        st = np.stack(imgs, 0)
        tot = st.sum(0)
        n = len(imgs)
        for x in range(n):
            cent = (tot - st[x]) / (n - 1)
            Bm.append(ssim(imgs[x], cent))
            Bi.append(ink_iou(imgs[x], cent.astype(np.float32)))
        if ki % 1000 == 0 and ki:
            el = time.time() - t2
            print(f"   [{el:5.0f}s] 键 {ki}/{len(g3)} 对 {len(A):,} eta {el/ki*(len(g3)-ki):.0f}s")

    # ---- C/D: 采样 ----
    rnd = random.Random(42)

    def samp(g, n):
        items = list(g.items())
        rnd.shuffle(items)
        ss, ii = [], []
        for key, ids in items:
            if len(ss) >= n:
                break
            ids = [i for i in ids if str(i) in pix]
            if len(ids) < 2:
                continue
            pr = list(itertools.combinations(ids, 2))
            rnd.shuffle(pr)
            for a_, b_ in pr[:8]:
                ss.append(ssim(P(a_), P(b_)))
                ii.append(ink_iou(P(a_), P(b_)))
                if len(ss) >= n:
                    break
        return ss, ii

    SAMP = 3000
    print(f"\n[C] 跨书家 (s,ch) 采样 {SAMP} ...")
    C, Ci = samp(g2, SAMP)
    print(f"[D] 同字 (ch) 采样 {SAMP} ...")
    D, Di = samp(g1, SAMP)

    print("\n" + "=" * 92)
    print("  【SSIM · 解码域(与评测同空间)】")
    print("=" * 92)
    desc("A_三键全同_穷举两两", A, S)
    desc("B_三键全同_留一质心(学习上限)", Bm, S)
    desc("E_三键全同_按键平均", Kw, S)
    desc("C_跨书家(s,ch)_采样", C, S)
    desc("D_同字(ch)_采样", D, S)
    print("\n  【墨迹 IoU】")
    desc("A_三键全同_两两", Ai, S)
    desc("B_三键全同_留一质心", Bi, S)
    desc("C_跨书家(s,ch)", Ci, S)
    desc("D_同字(ch)", Di, S)

    S["_meta"] = {"csv": CSV, "shards": SHARDS, "domain": "vae-decoded(uint8)",
                  "n_keys_csch": len(g3), "n_pairs_exhaustive": len(A),
                  "n_loo": len(Bm), "sample_size": SAMP,
                  "secs": round(time.time() - t0, 1)}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(S, f, ensure_ascii=False, indent=2)
    print(f"\n已写 {OUT}   总用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
