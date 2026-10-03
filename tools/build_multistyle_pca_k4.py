"""把 K=4 风格初始化从 K-Means 换成 **PCA 正交基** (对称性强行打破)。

## 病灶 (2026-10-03 实测)
`assets/multistyle_k4_top10.pt` 的 4 个 K-Means 质心两两余弦 **均值 0.900 / 最大 0.970**
(历史 v15b 是 0.884) —— 同一个书家的 DINO 特征本来就聚成一团, 在一团里找 4 个质心,
它们必然几乎共线 -> K 个 token 不可区分 -> attention 无内容可寻址 -> 退化成均匀
Average Pooling (探针三实测归一化熵 0.97) -> 多模态是假的, 实测只值 +0.0016。

## 本脚本的改动 (只在初始化, 不改模型代码)
对每个 pair 的 DINO 特征 F (已 L2 归一化):
    Token_0 = normalize(mean(F))                        <- 全局风格基底, 承载均值
    X       = F - mean(F)
    [v1,v2,v3] = X 的 top-3 右奇异向量 (PCA 主成分)
    Token_i = v_i (i=1..3)                              <- 主成分方向
再做 **Gram-Schmidt 正交化** [Token_0, v1, v2, v3]:
    先让 v_i 减掉在 Token_0 上的投影 (保证 mean ⟂ PC, 严格成立), 再让 v_i 互相正交。
=> 4 个 token 两两余弦 **严格 ≈ 0** (夹角 90°), 模型在 step 0 就无法把它们平均化。

## 尺度对齐 (保证与 K-Means 版**同统计量**, 对照公平)
原始质心 L2 范数均值 = 0.925。本脚本把 4 个 token 都缩放到同一尺度 SCALE=0.925,
所以模型 forward 的输入尺度在 init 时与 K-Means 版完全一致 (唯一变量 = token 几何)。

## 产物 (与 build_multistyle_k4.py 同键名, 可直接换给 MultiStyleEmbedder / config)
    embedding (P, K*D) / centroids (P, K, D) / pair_mean (P, D) / ...
    额外: basis="pca_orthogonal" / gram_cos_* (体检测试值)

用法:
  python tools/build_multistyle_pca_k4.py \
      --npz assets/dino_feat_top10.npz \
      --map exp-std/csv/callig_script_id_map_top10.json \
      --out assets/pca_multistyle_k4_top10.pt --k 4
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCALE = 0.925          # = 原 K-Means 质心的平均 L2 范数 (实测), 保证 init 尺度一致


def l2n(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def dim_project(feat, out_dim):
    """固定线性插值改维度 (与 build_multistyle_k4.dim_reduce_768_to_384 同一手法,
    零可学习参数)。★ 为什么需要: 拼进 KV 的风格 token 必须 = 模型 hidden,
    而 DiT-2Cond-Sp/2 的 hidden 是 **512** (dit.py:2720), DINO 特征却是 384 ->
    MultiStyleEmbedder 出的 token (B,K,384) 加 role (1,K,512) 直接广播失败
    (实测 smoke: RuntimeError broadcast 512 vs 384)。

    ⚠ 线性插值**不是**正交变换, 但正交化在插值**之后**做 (本脚本 Gram-Schmidt),
    所以最终 K 个 token 在 512 维里仍严格两两正交 ✓。
    """
    t = torch.from_numpy(feat).float().unsqueeze(1)                # (N,1,C)
    t = F.interpolate(t, size=out_dim, mode="linear", align_corners=False)
    return t.squeeze(1).numpy()


def l2n_t(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def orthonormal_basis_with_mean(mean_v, dirs, k):
    """返回 (k, D) 行正交单位基: 第 0 行 = normalize(mean), 其余 = dirs 对前者的正交补。

    严格 Gram-Schmidt: 保证 cos(row_i, row_j) == 0 (i≠j)。
    dirs 不足或退化时用随机正交向量补齐 (QR 分解, 确定性 seed)。
    """
    D = mean_v.shape[0]
    rows = [mean_v / max(np.linalg.norm(mean_v), 1e-12)]
    for d in dirs:
        v = d - sum((d @ r) * r for r in rows)          # 减掉在所有已有基上的投影
        nv = np.linalg.norm(v)
        if nv < 1e-6:                                   # 退化 (该方向与已有基共线)
            continue
        rows.append(v / nv)
    rng = np.random.RandomState(0)
    while len(rows) < k:
        v = rng.randn(D)
        v = v - sum((v @ r) * r for r in rows)
        nv = np.linalg.norm(v)
        if nv < 1e-6:
            continue
        rows.append(v / nv)
        print("    [warn] PCA 方向不足, 用随机正交向量补齐 (该 pair 特征秩低)")
    return np.stack(rows[:k]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--npz", default="assets/dino_feat_top10.npz")
    ap.add_argument("--map", dest="map_path",
                    default="exp-std/csv/callig_script_id_map_top10.json")
    ap.add_argument("--out", default="assets/pca_multistyle_k4_top10.pt")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--dim", type=int, default=384)
    a = ap.parse_args()

    z = np.load(a.npz)
    feat, calligs, scripts = z["feat"], z["calligs"], z["scripts"]
    if feat.shape[1] != a.dim:
        print(f"[dims] feat {feat.shape} -> 固定线性插值 -> {a.dim} 维 "
              f"(拼进 KV 的 token 必须 = 模型 hidden)")
        feat = dim_project(feat, a.dim)
    with open(a.map_path, encoding="utf-8") as f:
        cmap = json.load(f)
    pair_map = cmap["pair_map"]
    n_pairs = int(cmap["num_pairs"])

    feat_t = l2n(feat.astype(np.float32))
    pair_ids = np.array([pair_map.get(f"{int(c)}:{int(s)}", -1)
                         for c, s in zip(calligs.tolist(), scripts.tolist())])
    n_skip = int((pair_ids < 0).sum())

    K, D = a.k, feat_t.shape[1]
    embedding = torch.zeros(n_pairs, K * D)
    centroids = torch.zeros(n_pairs, K, D)
    pair_mean = torch.zeros(n_pairs, D)
    counts = torch.zeros(n_pairs, dtype=torch.int64)
    gram = []

    for pid in range(n_pairs):
        m = pair_ids == pid
        n = int(m.sum())
        counts[pid] = n
        Fp = feat_t[m]
        if n == 0:
            Fp = feat_t
            print(f"[warn] pair {pid} 无样本, 用全表均值")
        mu = Fp.mean(0)
        mean_v = l2n(mu[None, :])[0]
        X = Fp - mu
        # top-(K-1) PCA 方向 (右奇异向量); n 小或秩低时自动少于 K-1
        kd = max(1, min(K - 1, min(X.shape) - 1))
        try:
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            dirs = list(Vt[:kd])
        except np.linalg.LinAlgError:
            dirs = []
        basis = orthonormal_basis_with_mean(mean_v, dirs, K)      # (K, D) 严格正交
        cent = torch.from_numpy(basis * SCALE)
        embedding[pid] = cent.reshape(-1)
        centroids[pid] = cent
        pair_mean[pid] = torch.from_numpy(mean_v)
        cn = l2n_t(cent)
        sim = cn @ cn.T
        off = ~torch.eye(K, dtype=torch.bool)
        gram.append(float(sim[off].mean()))

    gm = float(np.mean(gram))
    torch.save({"embedding": embedding, "centroids": centroids, "pair_mean": pair_mean,
                "pair_to_callig": cmap.get("pair_to_callig"),
                "n_pairs": n_pairs, "k_clusters": K, "dim": D,
                "counts": counts, "n_skipped": n_skip,
                "basis": "pca_orthogonal", "scale": SCALE,
                "gram_cos_mean": gm,
                "map_path": a.map_path, "npz_path": a.npz}, a.out)

    print(f"[done] {a.out}")
    print(f"  embedding {tuple(embedding.shape)} | 尺度 SCALE={SCALE} (与原 K-Means 版同) ")
    print(f"  ★ token 两两余弦: 均值={gm:.6f}  最大={max(gram) if gram else float('nan'):.6f}"
          f"   <- 目标 ≈ 0 (正交)")
    print(f"  对照 K-Means 版: 均值 0.900 / 最大 0.970  (高度共线)")
    print(f"  pair_mean 保留为 DINO 真均值 (style_anchor_mode=mean 的目标不变)")


if __name__ == "__main__":
    main()
