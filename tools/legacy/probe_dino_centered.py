# -*- coding: utf-8 -*-
"""
probe_dino_centered.py — 验证「去掉书体主成分后，字形信息是否可分离」。

背景
----
DINO CLS 的信息含量探测（probe_dino_info_content.py）显示：
  - 同字跨书体最近邻 top-1 = 2.0%  <- 字形信息被书体主导
  - PC1 占 26.3% 能量，有效维 34
  - 同字距离 0.845 < 不同字距离 0.991（统计上同字更接近，但被竞争淹没）

假设：字形信息是次主成分，被书体主导的 PC1 压住。
若去掉书体主导分量（per-script 去均值 / 去 PC1），字形信息应更可分离。

本脚本测量去均值后的最近邻命中率，验证该假设。
若命中率显著提升 -> 正确信号 = per-script 去均值后的 DINO，无需换特征。
"""
import os, sys, json, collections
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np


def main():
    emb_path = "pretrained_models/dino_embeddings/glyph_dino_embeddings_384.npy"
    idx_path = "pretrained_models/dino_embeddings/glyph_dino_index.json"
    emb = np.load(emb_path).astype(np.float64)
    glyphs = json.load(open(idx_path, encoding="utf-8"))
    glyphs = glyphs.get("glyphs", glyphs)
    N = min(len(glyphs), emb.shape[0])
    emb = emb[:N]
    glyphs = glyphs[:N]

    NUM_CH = 7026
    pairs = []
    for g in glyphs:
        sid, cid = (int(g[0]), int(g[1])) if isinstance(g, (list, tuple)) \
            else divmod(int(g), NUM_CH)
        pairs.append((sid, cid))
    sids = np.array([p[0] for p in pairs])
    cids = np.array([p[1] for p in pairs])

    char2rows = collections.defaultdict(list)
    for i, (s, c) in enumerate(pairs):
        char2rows[c].append(i)
    multi = {c: rows for c, rows in char2rows.items() if len(rows) >= 2}

    def nn_rate(E, label):
        """同字跨书体最近邻命中率。E 已归一化。"""
        sim = E @ E.T
        hit = tot = 0
        for c, rows in multi.items():
            same = set(rows)
            for i in rows:
                order = np.argsort(-sim[i])
                for j in order:
                    if j != i:
                        break
                tot += 1
                if j in same:
                    hit += 1
        rate = hit / tot if tot else 0
        print(f"  {label:<48} 最近邻命中率 = {rate*100:.2f}% ({hit}/{tot})")
        return rate

    print("=" * 78)
    print("DINO CLS 字形信息分离实验")
    print("=" * 78)

    # 0. 原始
    E0 = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    r0 = nn_rate(E0, "原始 (书体+字混合)")

    # 1. per-script 去均值（去掉书体主导分量）
    E1 = E0.copy()
    for s in np.unique(sids):
        m = sids == s
        if m.sum() > 1:
            E1[m] -= E1[m].mean(0, keepdims=True)
    E1 = E1 / np.maximum(np.linalg.norm(E1, axis=1, keepdims=True), 1e-12)
    r1 = nn_rate(E1, "per-script 去均值后")

    # 2. 去 PC1（全局最主导方向，即书体分量）
    Em = E0 - E0.mean(0, keepdims=True)
    _, s, Vt = np.linalg.svd(Em, full_matrices=False)
    PC1 = Vt[0]
    E2 = E0 - np.outer(E0 @ PC1, PC1)
    E2 = E2 / np.maximum(np.linalg.norm(E2, axis=1, keepdims=True), 1e-12)
    r2 = nn_rate(E2, "去全局 PC1 后")

    # 3. 去前 k 个主成分（去掉书体占主导的 PC 集群）
    for k in (4, 10, 30):
        PCs = Vt[:k]
        E3 = E0 - (E0 @ PCs.T) @ PCs
        E3 = E3 / np.maximum(np.linalg.norm(E3, axis=1, keepdims=True), 1e-12)
        r = nn_rate(E3, f"去前 {k} 个主成分后")

    # 4. 白化：PCA 后按能量归一化（放大低维字形分量）
    #    先降维到能捕捉字形的维度（如 200），再白化
    k_whiten = 200
    Em_c = E0 - E0.mean(0, keepdims=True)
    u, sw, vw = np.linalg.svd(Em_c, full_matrices=False)
    # u: (20468, 384), vw: (384, 384)
    # 投影到前 k 个 PC: E_k = (E0 - mean) @ V[:, :k]
    Vk = vw[:k_whiten].T            # (384, k)
    E4 = Em_c @ Vk                  # (N, k)
    E4 = E4 / np.maximum(np.linalg.norm(E4, axis=1, keepdims=True), 1e-12)
    r4 = nn_rate(E4, f"PCA 投影到 {k_whiten} 维 (等权)")

    print("\n" + "=" * 78)
    print("汇总")
    print("=" * 78)
    print(f"  原始命中率         = {r0*100:.2f}%")
    print(f"  per-script 去均值  = {r1*100:.2f}%   ({r1/r0:.1f}x)")
    print(f"  去 PC1             = {r2*100:.2f}%   ({r2/r0:.1f}x)")
    print(f"  PCA{200} 等权        = {r4*100:.2f}%   ({r4/r0:.1f}x)")
    print("\n  判读:")
    print(f"  若 per-script 去均值显著提升 -> 字形信息是次主成分，"
          f"正确信号 = 去均值 DINO（无需换特征）")
    print(f"  若去主成分都提升不大       -> 字形信息分布太散，"
          f"需更强表示（patch tokens）")

    os.makedirs("5script", exist_ok=True)
    with open("5script/dino_centered.json", "w", encoding="utf-8") as f:
        json.dump({"raw": r0, "per_script_center": r1, "no_pc1": r2,
                   "no_pc4": 0, "no_pc30": 0, "pca200": r4}, f)


if __name__ == "__main__":
    main()
