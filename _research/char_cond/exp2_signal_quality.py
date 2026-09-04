# -*- coding: utf-8 -*-
"""exp2: 修复 per-script centering + 分书体内评测（排除书体方差）。

exp1 发现: 所有方案 cos_同字(0.21) < cos_形近(0.30) → 书体方差 >> 字形方差。
  → 跨书体比较被书体主导，必须**在书体内**评测才有意义。

本实验:
  1) 正确实现 per-script centering（v8a 正在用的方案），验证其是否提升 cos_同字
  2) **within-script AUC**：每个书体内，形近字对 vs 随机字对的余弦 AUC
     → 直接衡量"字形结构信号"质量，排除书体干扰
  3) 跨书体 cos_同字（去书体效果）
"""
import os
import sys
import json
import csv
import numpy as np
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"G:\GitHub\DiT"
os.chdir(ROOT)

EMB = "pretrained_models/dino_embeddings/glyph_dino_embeddings_384.npy"
IDX = "pretrained_models/dino_embeddings/glyph_dino_index.json"
CSV = "5script/train_fame.csv"
SCRIPT_NAME = {0: "楷", 1: "行", 2: "草", 3: "篆", 4: "隶"}

SIM_PAIRS = [
    ("土","士"),("大","太"),("人","入"),("日","曰"),("天","夫"),("刀","力"),
    ("申","由"),("王","玉"),("牛","午"),("己","已"),("未","末"),("木","林"),
    ("休","体"),("侯","候"),("风","凤"),("几","凡"),("戊","戍"),("己","巳"),
    ("亳","毫"),("宋","宗"),("东","车"),("冈","同"),("三","王"),("十","干"),
    ("大","天"),("夫","天"),("犬","太"),("人","个"),("手","毛"),("毛","手"),
    ("子","孑"),("戊","戌"),("戍","戌"),("刀","刁"),("万","方"),("鸟","乌"),
    ("贝","见"),("龙","尤"),("失","矢"),("句","向"),("因","困"),("同","回"),
    ("问","间"),("门","们"),("口","曰"),("田","由"),("甲","申"),("白","百"),
    ("吉","古"),("夫","夭"),("天","夭"),("王","主"),("玉","主"),("人","入"),
]


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def per_script_center(e, sids):
    out = e.copy()
    for s in np.unique(sids):
        m = sids == s
        if m.sum() > 1:
            out[m] = e[m] - e[m].mean(0, keepdims=True)
    return out


def pca_whiten(e, eps=1e-6):
    x = e - e.mean(0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    return l2((u[:, :s.size] * s) @ vt / np.sqrt((s ** 2) / max(x.shape[0] - 1, 1) + eps))


def auc(pos, neg):
    s = np.concatenate([pos, neg])
    lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(s)
    ranks = np.where(lab[order] == 1)[0] + 1
    npos, nneg = len(pos), len(neg)
    if npos == 0 or nneg == 0:
        return 0.5
    return float((ranks.sum() - npos * (npos + 1) / 2) / (npos * nneg))


def eff_rank(m, thr=0.90):
    s = np.linalg.svd(m - m.mean(0, keepdims=True), compute_uv=False)
    e = s ** 2
    e = e / e.sum()
    return int(np.searchsorted(np.cumsum(e), thr) + 1)


def main():
    emb0 = np.load(EMB).astype(np.float32)
    glyphs = [tuple(x) for x in json.load(open(IDX, encoding="utf-8"))["glyphs"]]
    id2ch = {}
    for r in csv.DictReader(open(CSV, encoding="utf-8")):
        id2ch[int(r["character_id"])] = r["character"]
    sids = np.array([int(g[0]) for g in glyphs])
    cids = np.array([int(g[1]) for g in glyphs])
    print(f"glyph table {emb0.shape}, {len(np.unique(sids))} scripts")

    schemes = {
        "0 raw": lambda e: e,
        "1 L2": l2,
        "2 per-script center": lambda e: l2(per_script_center(e, sids)),
        "3 global demean": lambda e: l2(e - e.mean(0, keepdims=True)),
        "4 demean+script": lambda e: l2(per_script_center(e - e.mean(0, keepdims=True), sids)),
        "5 PCA-whiten": pca_whiten,
    }

    print("\n" + "=" * 92)
    print(f"{'方案':<22}{'跨书体cos同字':>14}{'书体内AUC':>12}{'书体内形近':>12}"
          f"{'书体内随机':>12}{'有效秩':>8}")
    print("=" * 92)

    results = []
    for name, fn in schemes.items():
        e = l2(np.asarray(fn(emb0), dtype=np.float32))
        # 跨书体 cos_同字
        by = defaultdict(list)
        for i, c in enumerate(cids):
            by[c].append(i)
        vals = []
        for c, ii in by.items():
            if len(ii) < 2:
                continue
            sub = e[ii]
            sm = sub @ sub.T
            k = min(len(ii), 6)
            vals += [sm[a, b] for a in range(k) for b in range(a + 1, k)]
        cos_same = float(np.mean(vals)) if vals else float("nan")

        # 书体内：形近 vs 随机 AUC
        aucs, csim, crand = [], [], []
        for s in np.unique(sids):
            m = sids == s
            ch_of = {}
            for i in np.where(m)[0]:
                ch = id2ch.get(cids[i])
                if ch and len(ch) == 1:
                    ch_of[ch] = i
            if len(ch_of) < 50:
                continue
            sp = [(a, b) for a, b in SIM_PAIRS if a in ch_of and b in ch_of]
            if not sp:
                continue
            pos = np.array([e[ch_of[a]] @ e[ch_of[b]] for a, b in sp])
            rng = np.random.default_rng(int(s))
            ks = list(ch_of.keys())
            neg = []
            sps = set(SIM_PAIRS) | {(b, a) for a, b in SIM_PAIRS}
            while len(neg) < 3000:
                a, b = rng.choice(ks, 2, replace=False)
                if (a, b) in sps:
                    continue
                neg.append(e[ch_of[a]] @ e[ch_of[b]])
            neg = np.array(neg)
            aucs.append(auc(pos, neg))
            csim.append(pos.mean())
            crand.append(neg.mean())
        results.append({
            "scheme": name, "cos_same_cross_script": round(cos_same, 4),
            "within_auc": round(float(np.mean(aucs)), 4),
            "within_cos_sim": round(float(np.mean(csim)), 4),
            "within_cos_rand": round(float(np.mean(crand)), 4),
            "eff_rank": eff_rank(e),
        })
        print(f"{name:<22}{cos_same:>14.4f}{np.mean(aucs):>12.4f}"
              f"{np.mean(csim):>12.4f}{np.mean(crand):>12.4f}"
              f"{eff_rank(e):>8}")
    print("=" * 92)

    out = "_research/char_cond/state/exp2_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
