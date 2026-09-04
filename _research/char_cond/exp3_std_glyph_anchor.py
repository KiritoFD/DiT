# -*- coding: utf-8 -*-
"""exp3: 标准字形(印刷体 kai/li) vs 真迹 glyph 特征 —— 谁的条件信号更干净?

动机 (来自 Calliffusion / zi2zi / FFG 的共同做法):
  视觉特征 inherently 混合"内容"与"风格"。真迹 DINO 平均特征必然被书体/书家主导
  (exp2 实测: 跨书体 cos_同字 仅 0.24)。而同领域工作的 content 条件来自
  **标准字形或文本符号**，不是真迹视觉特征。

评测: 三层次 cos 是否有序
    cos(同字)  >  cos(形近字)  >  cos(随机字)
标准字形可用 kai 与 li 两种印刷体构造"同字"正对（同一字的两种印刷体）。
"""
import os
import sys
import json
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"G:\GitHub\DiT"
os.chdir(ROOT)
D = "_research/char_cond/data"

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


def load_std(name):
    e = np.load(os.path.join(D, f"std_{name}_cls.npy")).astype(np.float32)
    idx = json.load(open(os.path.join(D, f"std_{name}_index.json"), encoding="utf-8"))
    cps = idx["codepoints"] if isinstance(idx, dict) and "codepoints" in idx else idx
    return {chr(cp): e[i] for i, cp in enumerate(cps)}, e


def eff_rank(m, thr=0.90):
    s = np.linalg.svd(m - m.mean(0, keepdims=True), compute_uv=False)
    e = s ** 2
    e = e / e.sum()
    return int(np.searchsorted(np.cumsum(e), thr) + 1)


def auc(pos, neg):
    s = np.concatenate([pos, neg])
    lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(s)
    ranks = np.where(lab[order] == 1)[0] + 1
    npos, nneg = len(pos), len(neg)
    if npos == 0 or nneg == 0:
        return 0.5
    return float((ranks.sum() - npos * (npos + 1) / 2) / (npos * nneg))


def three_level(mat, chars, label):
    """在给定特征矩阵上算三层次: 需外部提供 same pairs。"""
    pass


def main():
    kai, kai_e = load_std("kai")
    li, li_e = load_std("li")
    print(f"std-kai: {kai_e.shape} chars={len(kai)}")
    print(f"std-li : {li_e.shape} chars={len(li)}")

    common = sorted(set(kai) & set(li))
    print(f" kai∩li 共 {len(common)} 字（用于构造同字正对）")

    # 同一字的 kai / li 特征（维度可能不同 → 用各自 L2 后需同维）
    d = min(kai_e.shape[1], li_e.shape[1])
    K = l2(np.stack([kai[c][:d] for c in common]))
    L = l2(np.stack([li[c][:d] for c in common]))

    # 同字 cos（kai vs li，同字不同印刷体）
    cos_same = np.array([K[i] @ L[i] for i in range(len(common))])

    idx = {c: i for i, c in enumerate(common)}
    sp = [(a, b) for a, b in SIM_PAIRS if a in idx and b in idx]
    # 形近 cos（kai 内部 + li 内部）
    cos_sim = np.array([K[idx[a]] @ K[idx[b]] for a, b in sp] +
                       [L[idx[a]] @ L[idx[b]] for a, b in sp])
    # 随机 cos
    rng = np.random.default_rng(0)
    sps = set(SIM_PAIRS) | {(b, a) for a, b in SIM_PAIRS}
    neg = []
    while len(neg) < 4000:
        a, b = rng.choice(len(common), 2, replace=False)
        if (common[a], common[b]) in sps:
            continue
        neg.append(K[a] @ K[b])
    cos_rand = np.array(neg)

    print("\n" + "=" * 70)
    print("标准字形(kai/li) 三层次评测")
    print("=" * 70)
    print(f"  同字(kai↔li) cos = {cos_same.mean():.4f} ± {cos_same.std():.4f}")
    print(f"  形近字 cos       = {cos_sim.mean():.4f} ± {cos_sim.std():.4f}  (n={len(sp)} 对)")
    print(f"  随机字 cos       = {cos_rand.mean():.4f} ± {cos_rand.std():.4f}")
    ok = cos_same.mean() > cos_sim.mean() > cos_rand.mean()
    print(f"  三层次有序?      {'✓ 是' if ok else '✗ 否'}")
    print(f"  间隔(同字-形近)  = {cos_same.mean()-cos_sim.mean():+.4f}")
    print(f"  形近/随机 AUC    = {auc(cos_sim, cos_rand):.4f}")
    print(f"  有效秩(kai)      = {eff_rank(K)} / {d}")
    print("=" * 70)

    # 对比真迹（exp2 结果）
    print("\n对比 —— 真迹 glyph 表 (exp2, per-script center):")
    print("  跨书体 cos_同字 = 0.2418 | 书体内 形近 0.4020 / 随机 0.0019")
    print("  → 真迹: 同字(0.24) < 形近(0.40) ✗ 层次错乱")
    print(f"  → 标准字形: 同字({cos_same.mean():.4f}) vs 形近({cos_sim.mean():.4f}) "
          f"{'✓ 层次正确' if ok else '✗ 仍错乱'}")

    out = "_research/char_cond/state/exp3_results.json"
    json.dump({
        "n_common": len(common),
        "cos_same_kai_li": round(float(cos_same.mean()), 4),
        "cos_same_std": round(float(cos_same.std()), 4),
        "cos_sim": round(float(cos_sim.mean()), 4),
        "cos_rand": round(float(cos_rand.mean()), 4),
        "hierarchy_ok": bool(ok),
        "margin": round(float(cos_same.mean() - cos_sim.mean()), 4),
        "auc_sim_rand": round(auc(cos_sim, cos_rand), 4),
        "eff_rank_kai": int(eff_rank(K)),
    }, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
