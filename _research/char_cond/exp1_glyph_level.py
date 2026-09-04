# -*- coding: utf-8 -*-
"""exp1: glyph 级 DINO 表的多种"信号处理"方案对比 (本地 CPU).

核心判据 —— 三层次 cos 必须有序:
    cos(同字跨书体)  >  cos(形近字)  >  cos(随机字)
当前 DINO 的问题: 形近字 cos(0.986) > 同字 cos(0.817) → 层次错乱, 模型会混淆形近字。

对每个处理方案输出:
  cos_same / cos_sim / cos_rand / margin(same-sim) / hierarchy_ok / eff_rank / nn_hit

用法: python _research/char_cond/exp1_glyph_level.py
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

# 人工标注形近字对（沿用 _sync_work/_research_dino_char_embed.py）
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


def eff_rank(m, thr=0.90):
    """90% 能量所需维数（有效秩代理）。"""
    s = np.linalg.svd(m - m.mean(0, keepdims=True), compute_uv=False)
    e = (s ** 2)
    e = e / e.sum()
    c = np.cumsum(e)
    return int(np.searchsorted(c, thr) + 1)


def load():
    emb = np.load(EMB).astype(np.float32)
    glyphs = [tuple(x) for x in json.load(open(IDX, encoding="utf-8"))["glyphs"]]
    id2ch = {}
    for r in csv.DictReader(open(CSV, encoding="utf-8")):
        id2ch[int(r["character_id"])] = r["character"]
    return emb, glyphs, id2ch


def to_char_level(emb, glyphs, id2ch):
    """glyph 级 → char 级（同 char 跨书体平均 + L2）。"""
    acc = defaultdict(list)
    for (sid, cid), e in zip(glyphs, emb):
        ch = id2ch.get(cid)
        if not ch or len(ch) != 1:
            continue
        acc[ch].append(e)
    chars = sorted(acc.keys())
    mat = np.stack([np.mean(acc[c], axis=0) for c in chars])
    return l2(mat), chars


def evaluate(mat, chars, label):
    idx = {c: i for i, c in enumerate(chars)}
    sim = mat @ mat.T
    rng = np.random.default_rng(0)

    # 同字跨书体 cos（char 级已聚合，改用"同字的 glyph 变体"，见调用方）
    # 形近字对
    sp = [(a, b) for a, b in SIM_PAIRS if a in idx and b in idx]
    cos_sim = np.array([sim[idx[a], idx[b]] for a, b in sp]) if sp else np.array([0.0])
    # 随机字对
    neg = []
    sps = set(SIM_PAIRS) | {(b, a) for a, b in SIM_PAIRS}
    while len(neg) < 2000:
        a, b = rng.choice(len(chars), 2, replace=False)
        if (chars[a], chars[b]) in sps:
            continue
        neg.append(sim[a, b])
    cos_rand = np.array(neg)
    # 最近邻命中（形近字是否被误当最近邻 → 混淆度）
    hit_sim = 0
    for a, b in sp:
        order = np.argsort(-sim[idx[a]])
        top = [j for j in order if j != idx[a]][:1]
        if top and top[0] == idx[b]:
            hit_sim += 1
    return {
        "label": label,
        "n_char": len(chars),
        "cos_sim": float(cos_sim.mean()),
        "cos_rand": float(cos_rand.mean()),
        "eff_rank": eff_rank(mat),
        "nn_is_simpair": round(hit_sim / max(len(sp), 1), 4),  # 越低越好
    }


def report(rows, cos_same_map):
    print("\n" + "=" * 96)
    print(f"{'方案':<28}{'cos_同字':>10}{'cos_形近':>10}{'cos_随机':>10}"
          f"{'间隔':>10}{'有效秩':>8}{'NN=形近':>10}")
    print("=" * 96)
    for r in rows:
        cs = cos_same_map.get(r["label"], float("nan"))
        margin = cs - r["cos_sim"]
        ok = "✓" if (cs > r["cos_sim"] > r["cos_rand"]) else "✗"
        print(f"{r['label']:<28}{cs:>10.4f}{r['cos_sim']:>10.4f}"
              f"{r['cos_rand']:>10.4f}{margin:>+10.4f}{r['eff_rank']:>8}"
              f"{r['nn_is_simpair']:>9.1%} {ok}")
    print("=" * 96)


def main():
    emb0, glyphs, id2ch = load()
    print(f"glyph table: {emb0.shape}, glyphs={len(glyphs)}")
    sids = np.array([int(g[0]) for g in glyphs])
    rows = []
    cos_same = {}

    # ---- 方案定义：输入 glyph 级 emb，输出处理后 emb ----
    schemes = {}
    schemes["0 raw"] = lambda e: e
    schemes["1 L2"] = lambda e: l2(e)
    schemes["2 per-script center"] = lambda e: l2(np.stack([
        e[sids == s] - e[sids == s].mean(0, keepdims=True)
        for s in np.unique(sids)
    ])[np.argsort(np.concatenate([np.where(sids == s)[0]
                                 for s in np.unique(sids)]))])
    schemes["3 global demean"] = lambda e: l2(e - e.mean(0, keepdims=True))
    schemes["4 demean+script"] = lambda e: l2(
        (e - e.mean(0, keepdims=True)) - np.stack([
            (e - e.mean(0, keepdims=True))[sids == s].mean(0, keepdims=True)
            for s in np.repeat(np.unique(sids),
                               [int((sids == s).sum()) for s in np.unique(sids)])
        ]))
    schemes["5 PCA-whiten"] = lambda e: _pca_whiten(e)
    schemes["6 PCA-128"] = lambda e: _pca_keep(e, 128)
    schemes["7 rank-norm"] = lambda e: l2(_rank_norm(e))

    for name, fn in schemes.items():
        try:
            e2 = np.asarray(fn(emb0), dtype=np.float32)
        except Exception as ex:
            print(f"  [skip] {name}: {ex}")
            continue
        # char 级评测
        mat, chars = to_char_level(e2, glyphs, id2ch)
        r = evaluate(mat, chars, name)
        # 同字 cos：用 glyph 级（同 char 不同 script 的 glyph 两两 cos）
        cs = _cos_same_glyph(e2, glyphs)
        cos_same[name] = cs
        rows.append(r)
        print(f"  done {name}: n_char={len(chars)} cos_same={cs:.4f} "
              f"cos_sim={r['cos_sim']:.4f} cos_rand={r['cos_rand']:.4f} "
              f"rank={r['eff_rank']}")

    report(rows, cos_same)
    # 落盘
    out = os.path.join("_research", "char_cond", "state", "exp1_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump([{**r, "cos_same": cos_same.get(r["label"])} for r in rows],
                  f, ensure_ascii=False, indent=1)
    print(f"\n[saved] {out}")


def _cos_same_glyph(e, glyphs, max_pairs=4000):
    """同字不同书体 glyph 两两 cos 均值。"""
    by = defaultdict(list)
    for i, (sid, cid) in enumerate(glyphs):
        by[cid].append(i)
    m = l2(e)
    vals = []
    for cid, ii in by.items():
        if len(ii) < 2:
            continue
        for a in range(len(ii)):
            for b in range(a + 1, len(ii)):
                vals.append(float(m[ii[a]] @ m[ii[b]]))
        if len(vals) > max_pairs:
            break
    return float(np.mean(vals)) if vals else float("nan")


def _pca_whiten(e, eps=1e-6):
    x = e - e.mean(0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    w = (u[:, :s.size] * s) @ vt / np.sqrt((s ** 2) / max(x.shape[0] - 1, 1) + eps)
    return l2(w)


def _pca_keep(e, k):
    x = e - e.mean(0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    k = min(k, s.size)
    return l2((u[:, :k] * s[:k]) @ vt[:k])


def _rank_norm(e):
    """逐维秩归一化到 [-1,1]（去离群、均衡各维）。"""
    x = np.empty_like(e, dtype=np.float32)
    for d in range(e.shape[1]):
        r = np.argsort(np.argsort(e[:, d])).astype(np.float32)
        x[:, d] = 2.0 * r / max(e.shape[0] - 1, 1) - 1.0
    return x


if __name__ == "__main__":
    main()
