#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DINO 字符条件诊断工具。

复现 docs/system/12_dino_diagnosis_20260829.md 里的所有数字:

    python tools/dino_diagnose.py --emb <emb.npy> --index <index.json> \
        [--csv 5script/train_mid_clean.csv ...] [--device cuda]

报告四组指标:
  1. 覆盖率    : 各 CSV 里的 glyph 有多少能在 DINO 表里找到
  2. 有效秩    : 用**完整 SVD** 算（不要用随机 SVD，容易把 σ² 又平方成 σ⁴）
  3. 检索能力  : 跨书体字符检索 top-1/5/10，以及"最近邻是同书体"的泄漏率
  4. 修正效果  : per-script centering / cross-script 平均 对上述指标的影响

有效秩定义（Shannon 熵形式）:
    p_i = σ_i² / Σσ²   ;   eff_rank = exp(-Σ p_i log p_i)
"""
import argparse
import csv
import io
import json
import os

import numpy as np
import torch


# --------------------------------------------------------------------------- I/O
def load(emb_path, index_path, device):
    emb = np.load(emb_path)
    with io.open(index_path, "r", encoding="utf-8") as f:
        idx = json.load(f)
    glyphs = [(int(g[0]), int(g[1])) for g in idx.get("glyphs", idx)]
    X = torch.from_numpy(np.ascontiguousarray(emb, dtype=np.float32)).to(device)
    sid = torch.tensor([g[0] for g in glyphs], device=device)
    cid = torch.tensor([g[1] for g in glyphs], device=device)
    return X, sid, cid, glyphs


def glyphs_of(path, num_ch=7026):
    out = set()
    with io.open(path, "r", encoding="utf-8") as f:
        for d in csv.DictReader(f):
            g = d.get("glyph_id")
            if g is None or g == "":
                g = int(d["script_id"]) * num_ch + int(d["character_id"])
            out.add(int(g))
    return out


# ------------------------------------------------------------------- metrics
def eff_rank(T):
    """精确有效秩。**不要**用随机 SVD 近似后直接 p=S**2 —— S 已经是 σ²。"""
    Tc = (T - T.mean(0, keepdim=True)).double()
    S = torch.linalg.svdvals(Tc)
    p = S ** 2
    p = p / p.sum()
    er = float(torch.exp(-(p * torch.log(p + 1e-30)).sum()))
    return er, float(p[0]), float(p[:10].sum()), float(p[:64].sum())


def l2(T):
    return torch.nn.functional.normalize(T, dim=-1)


def retrieval(M, cid, n_sel=2000, seed=0):
    """全表检索：top-k 是否命中同一 char_id（不排除同书体）。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    sel = torch.randperm(len(M), generator=g, device="cpu")[:n_sel].to(M.device)
    h = [0, 0, 0]
    ks = (1, 5, 10)
    n = 0
    for i in sel.tolist():
        sim = M @ M[i]
        sim[i] = -1e9
        top = torch.argsort(-sim)
        lab = cid[top]
        for j, k in enumerate(ks):
            h[j] += int((lab[:k] == cid[i]).any())
        n += 1
    return [100 * x / n for x in h]


def script_leak(M, sid, n_sel=800, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    sel = torch.randperm(len(M), generator=g, device="cpu")[:n_sel].to(M.device)
    same = 0
    for i in sel.tolist():
        sim = M @ M[i]
        sim[i] = -1e9
        j = int(torch.argmax(sim))
        same += int(sid[j] == sid[i])
    return 100 * same / len(sel)


# ------------------------------------------------------------------ variants
def per_script_center(X, sid):
    Xa = X.clone()
    for s in torch.unique(sid).tolist():
        m = sid == s
        if int(m.sum()) > 1:
            Xa[m] = X[m] - X[m].mean(0, keepdim=True)
    return l2(Xa)


def cross_script_avg(X, cid):
    uniq_c = torch.unique(cid)
    cv = torch.zeros(len(uniq_c), X.shape[1], device=X.device, dtype=X.dtype)
    cnt = torch.zeros(len(uniq_c), device=X.device, dtype=X.dtype)
    pos = {int(c): i for i, c in enumerate(uniq_c.tolist())}
    for i in range(len(X)):
        j = pos[int(cid[i])]
        cv[j] += X[i]
        cnt[j] += 1
    return cv / cnt[:, None], uniq_c


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="pretrained_models/dino_embeddings/"
                                     "glyph_dino_embeddings_384.npy")
    ap.add_argument("--index", default="pretrained_models/dino_embeddings/"
                                       "glyph_dino_index.json")
    ap.add_argument("--csv", nargs="*", default=[
        "5script/train_mid_clean.csv", "5script/train_mid_common.csv",
        "5script/eval_strict_top6.csv", "5script/eval_unseen_top6.csv",
        "5script/eval_strict_midclean.csv"])
    ap.add_argument("--num-ch", type=int, default=7026)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    X, sid, cid, glyphs = load(args.emb, args.index, dev)
    print(f"glyphs={len(glyphs)}  scripts={torch.unique(sid).tolist()}  "
          f"unique_chars={len(torch.unique(cid))}  dim={X.shape[1]}")

    # ---- 1. 覆盖率 ----
    print("\n[1] 覆盖率")
    dino_gid = set(s * args.num_ch + c for s, c in glyphs)
    dino_ch = set(c for _, c in glyphs)
    for p in args.csv:
        if not os.path.exists(p):
            print(f"    {os.path.basename(p):32s} MISSING")
            continue
        gs = glyphs_of(p, args.num_ch)
        hit = len(gs & dino_gid)
        pairs = set()
        print(f"    {os.path.basename(p):32s} uniq={len(gs):6d}  "
              f"hit={hit:6d} ({100*hit/len(gs):5.1f}%)  miss={len(gs)-hit}")

    # ---- 2/3. 有效秩 + 检索 ----
    print("\n[2] 有效秩 / 检索 / 书体泄漏")
    print(f"    {'variant':26s} {'eff_rank':>9s} {'PC1':>6s} {'PC10':>6s} "
          f"{'top1':>7s} {'top5':>7s} {'top10':>7s} {'leak':>7s}")
    rows = [
        ("raw (s19 现状)", l2(X), cid, sid),
        ("per-script centering", per_script_center(X, sid), cid, sid),
    ]
    for tag, M, lab, sl in rows:
        er, pc1, pc10, _ = eff_rank(M)
        r = retrieval(M, lab)
        lk = script_leak(M, sl)
        print(f"    {tag:26s} {er:9.1f} {pc1:6.3f} {pc10:6.3f} "
              f"{r[0]:6.1f}% {r[1]:6.1f}% {r[2]:6.1f}% {lk:6.1f}%")

    cv, uc = cross_script_avg(X, cid)
    er, pc1, pc10, _ = eff_rank(cv)
    print(f"    {'cross-script avg':26s} {er:9.1f} {pc1:6.3f} {pc10:6.3f} "
          f"{'—':>7s} {'—':>7s} {'—':>7s} {'—':>7s}   (每 char 仅一个向量，无法自检索)")
    nch = len(torch.unique(cid))
    print(f"\n    随机基线 top-1 = {100.0/nch:.3f}%  (1/{nch} 个 char)")
    print(f"    书体泄漏 chance  = {100.0/len(torch.unique(sid)):.1f}%")

    # ---- 4. unknown 行填充候选 ----
    print("\n[3] unknown 行填充向量候选")
    known = X[:1000]
    cands = {
        "mean of RAW, L2-norm": l2(X.mean(0, keepdim=True)).squeeze(),
        "mean of RAW (不归一)": X.mean(0),
        "mean of CENTERED": per_script_center(X, sid).mean(0),
        "N(0,0.02) (默认 init)": torch.randn(X.shape[1], device=dev) * 0.02,
    }
    for tag, v in cands.items():
        cos = torch.nn.functional.cosine_similarity(v[None], known, dim=-1)
        print(f"    {tag:24s} norm={v.norm():.4f}  cos-to-known mean={cos.mean():+.3f} "
              f"std={cos.std():.3f}")
    kk = torch.nn.functional.cosine_similarity(known[:200], known[200:400], dim=-1)
    print(f"    {'(参考) 已知行两两余弦':24s} mean={kk.mean():+.3f} std={kk.std():.3f}")
    print("\n    判据: 选 norm≈1.0 且 cos-to-known 接近已知行两两平均值的那个。")


if __name__ == "__main__":
    main()
