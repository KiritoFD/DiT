# -*- coding: utf-8 -*-
"""v15 Stage 0 — 多模态风格初始化: 87 个 (书家×书体) pair 的 DINO 特征 K-Means (K=4)。

## 动机（docs/919 §3 / doc 72）

单冻结 128 维向量装不下多模态风格: r(书家样本数, strict) = −0.62
（苏轼 3131 张 -> strict 0.3485 最低; 李邕 1168 张风格统一 -> 0.6300 最高）。
v15 给每个 pair K 个 style token；若随机初始化，K 个 token 在训练初期会
隐式塌缩成同一向量（v14 端到端训表塌缩 cos 0.323 的教训）。这里用该 pair
辖下全部样本的 DINO CLS 特征做 K-Means，把 K 个质心作为 token 初始化 ——
每个 token 天然对应一个风格模态，起点即分化。

## 降维对齐

`assets/dino_cls_50k.npz` 是 DINOv2 **ViT-S/14 CLS (384 维)**，直接聚类。
若换 768 维特征源（如 ViT-B），先用与 StdDinoCharEmbedder 相同的**固定线性
插值** (F.interpolate, mode="linear", align_corners=False) 降到 --dim（默认
384，= 模型 hidden）再聚类 —— 零可学习参数，构建时一次性算。

## 用法（远端）

    python tools/build_multistyle_k4.py \
        --npz assets/dino_cls_50k.npz \
        --map assets/callig_script_id_map.json \
        --out assets/multistyle_k4_pretrained.pt --k 4

## 产物 `assets/multistyle_k4_pretrained.pt`

    embedding (87, K*384)   <- 直接灌 MultiStyleEmbedder.embedding_table（全表, 无 null 行）
    centroids (87, K, 384)  <- K-Means 质心（体检/可视化用）
    pair_mean (87, 384)     <- pair 级 DINO 质心（--style-anchor-mode mean 的锚定目标）
    pair_to_callig / n_pairs / k_clusters / dim / counts / n_skipped / inertia_sum
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def dim_reduce_768_to_384(feat: np.ndarray, out_dim: int) -> np.ndarray:
    """StdDinoCharEmbedder 同款固定线性插值降维 (N, 768) -> (N, out_dim)。"""
    t = torch.from_numpy(feat).float().unsqueeze(1)              # (N, 1, 768)
    t = F.interpolate(t, size=out_dim, mode="linear", align_corners=False)
    return t.squeeze(1).numpy()


def l2n(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def kmeans_torch(x: torch.Tensor, k: int, seed: int, n_init: int = 4,
                 max_iter: int = 100) -> tuple:
    """无依赖 K-Means（kmeans++ 初始化 + 多重启取最优 inertia）。x: (n, d) 已 L2 归一化。

    kmeans++ 要求 n >= k；n < k 时按 n 截断（调用方负责把返回的 (n,d) 平铺到 K）。
    """
    k = min(k, x.shape[0])
    n = x.shape[0]
    best_c, best_i = None, float("inf")
    for init_id in range(n_init):
        g = torch.Generator().manual_seed(seed * 1000 + init_id)
        # ---- kmeans++ ----
        centers = [x[torch.randint(n, (1,), generator=g)].squeeze(0)]
        for _ in range(k - 1):
            d2 = torch.stack([((x - c) ** 2).sum(1) for c in centers]).min(0).values
            tot = d2.sum()
            if tot <= 0:
                # 剩余点与已有中心零距离(pair 内存在重复特征): 循环平铺填满,
                # 语义与 n<K 的退化路径一致 —— 不强造分裂
                base = list(centers)
                while len(centers) < k:
                    centers.append(base[len(centers) % len(base)])
                break
            prob = d2 / tot
            centers.append(x[torch.multinomial(prob, 1, generator=g)].squeeze(0))
        c = torch.stack(centers)                                  # (k, d)
        for _ in range(max_iter):
            assign = torch.cdist(x, c).argmin(1)                  # (n,)
            new_c = c.clone()
            for j in range(k):
                m = assign == j
                if m.any():
                    new_c[j] = x[m].mean(0)
            if torch.allclose(new_c, c, atol=1e-6):
                c = new_c
                break
            c = new_c
        assign = torch.cdist(x, c).argmin(1)
        inertia = ((x - c[assign]) ** 2).sum().item()
        if inertia < best_i:
            best_i, best_c = inertia, c
    return best_c, best_i


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--npz", default="assets/dino_cls_50k.npz",
                    help="DINO CLS 特征 (tools/extract_dino_cls_50k.py 产物)")
    ap.add_argument("--map", dest="map_path", default="assets/callig_script_id_map.json",
                    help="(书家×书体) pair 词表 (src/utils/callig_script_map.py 产物)")
    ap.add_argument("--out", default="assets/multistyle_k4_pretrained.pt")
    ap.add_argument("--k", type=int, default=4, help="每 pair 的风格 token 数")
    ap.add_argument("--dim", type=int, default=384,
                    help="聚类空间维度 (=模型 hidden; npz 已是 384 时原样使用)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if not os.path.exists(a.npz):
        a.npz = os.path.join("/root/Workspace/xy/DiT", a.npz)
    if not os.path.exists(a.map_path):
        a.map_path = os.path.join("/root/Workspace/xy/DiT", a.map_path)

    z = np.load(a.npz)
    feat, calligs, scripts = z["feat"], z["calligs"], z["scripts"]
    src_dim = feat.shape[1]
    if src_dim == a.dim:
        print(f"[dims] feat {feat.shape} 已是 {a.dim} 维, 直接聚类")
    elif src_dim == 768:
        print(f"[dims] feat {feat.shape} 为 768 维 -> StdDinoCharEmbedder 同款固定插值 -> {a.dim}")
        feat = dim_reduce_768_to_384(feat, a.dim)
    else:
        raise SystemExit(f"[dims] feat 维度 {src_dim} 既非 {a.dim} 也非 768, 请确认特征源")

    # ---- 垃圾特征守卫 (2026-09-19 教训: extract 漏 /255 -> 全行恒定) ----
    # 采样 2000 行查唯一值: 特征几乎恒定 = 提取管线坏了, 聚类毫无意义, 直接拒跑。
    _sub = feat[np.random.RandomState(0).choice(len(feat), min(2000, len(feat)),
                                                replace=False)]
    _uniq = len(np.unique(np.round(_sub, 4), axis=0))
    if _uniq < 100:
        raise SystemExit(
            f"[guard] ✗ 采样 {_sub.shape[0]} 行只有 {_uniq} 个唯一特征向量 —— "
            f"特征几乎恒定, 提取管线坏了 (历史教训: extract_dino_cls_*.py 漏 /255 -> "
            f"ViT attention 饱和 -> CLS 恒定)。请用修好的 extractor 重新提取 npz, "
            f"不要用坏特征聚类。")

    with open(a.map_path, encoding="utf-8") as f:
        cmap = json.load(f)
    pair_map = cmap["pair_map"]                                   # {"<callig>:<script>": pair_id}
    n_pairs = int(cmap["num_pairs"])

    feat_t = torch.from_numpy(l2n(feat.astype(np.float32)))       # L2 归一化后做欧氏聚类
    pair_ids = np.full(feat.shape[0], -1, dtype=np.int64)
    for i, (c, s) in enumerate(zip(calligs.tolist(), scripts.tolist())):
        pair_ids[i] = pair_map.get(f"{int(c)}:{int(s)}", -1)
    n_skip = int((pair_ids < 0).sum())
    if n_skip:
        print(f"[warn] {n_skip} 个样本的 (callig,script) 不在 pair_map 里, 跳过")

    K, D = a.k, feat_t.shape[1]
    embedding = torch.zeros(n_pairs, K * D)
    centroids = torch.zeros(n_pairs, K, D)
    pair_mean = torch.zeros(n_pairs, D)
    counts = torch.zeros(n_pairs, dtype=torch.int64)
    inertia_sum = 0.0
    degenerate = []

    for pid in range(n_pairs):
        m = torch.from_numpy(pair_ids == pid)
        n = int(m.sum())
        counts[pid] = n
        if n == 0:
            # 词表里有、特征里没有的 pair: 回退到该书家所有样本的均值质心
            cg = cmap["pair_to_callig"][pid] if "pair_to_callig" in cmap else None
            sel = torch.zeros(0, dtype=torch.long)
            if cg is not None:
                sel = torch.from_numpy((calligs == int(cg)).astype(np.int64)).nonzero().squeeze(1)
            src = feat_t[sel].mean(0, keepdim=True) if len(sel) else feat_t.mean(0, keepdim=True)
            cent = src.repeat(K, 1)
            degenerate.append((pid, "empty", int(len(sel))))
        elif n < K:
            # 样本数 < K: 只能聚 n 个簇, 循环平铺到 K（与 SupCon 预训练的质心回退同策略）
            cent, inr = kmeans_torch(feat_t[m], n, seed=a.seed + pid)
            cent = cent[torch.arange(K) % n]
            inertia_sum += inr
            degenerate.append((pid, f"n={n}<K", n))
        else:
            cent, inr = kmeans_torch(feat_t[m], K, seed=a.seed + pid)
            inertia_sum += inr
        embedding[pid] = cent.reshape(-1)
        centroids[pid] = cent
        pair_mean[pid] = F.normalize(feat_t[m].mean(0, keepdim=True), dim=1).squeeze(0) \
            if n > 0 else centroids[pid].mean(0)

    # ---- 体检报告 ----
    e2 = embedding.double()
    e2 = e2 - e2.mean(0, keepdim=True)
    _, sv, _ = torch.linalg.svd(e2, full_matrices=False)
    eff_rank = float((sv.sum() / sv.max()).item())
    # 簇间均衡: 每个 pair 内 K 个质心的最小簇占比（退化 pair 除外）
    bal = []
    for pid in range(n_pairs):
        if counts[pid] < K:
            continue
        m = torch.from_numpy(pair_ids == pid)
        assign = torch.cdist(feat_t[m], centroids[pid]).argmin(1)
        sz = torch.bincount(assign, minlength=K).float()
        bal.append(float(sz.min() / sz.sum()))
    cos_cc = F.cosine_similarity(centroids.unsqueeze(2), centroids.unsqueeze(1), dim=-1)
    intra = cos_cc[:, ~torch.eye(K, dtype=bool)].mean().item()

    torch.save({"embedding": embedding,              # (87, K*D)
                "centroids": centroids,              # (87, K, D)
                "pair_mean": pair_mean,              # (87, D)  <- style-anchor-mode=mean 的目标
                "pair_to_callig": cmap.get("pair_to_callig"),
                "n_pairs": n_pairs, "k_clusters": K, "dim": D,
                "counts": counts, "n_skipped": n_skip,
                "inertia_sum": inertia_sum,
                "degenerate": degenerate,
                "map_path": a.map_path, "npz_path": a.npz},
               a.out)

    print(f"[done] {a.out}")
    print(f"  embedding {tuple(embedding.shape)} | K={K} D={D} | 聚类总 inertia {inertia_sum:.1f}")
    print(f"  pair 样本数: min={counts.min().item()} med={counts.median().item():.0f} "
          f"max={counts.max().item()} | 空表回退 {sum(1 for d in degenerate if d[1]=='empty')} | "
          f"n<K 平铺 {sum(1 for d in degenerate if d[1]!='empty')}")
    if bal:
        print(f"  簇均衡(最小簇占比): med={sorted(bal)[len(bal)//2]:.3f} min={min(bal):.3f} "
              f"(过低=该 pair 风格模态单一, 属预期)")
    print(f"  质心健康: pair 内 K 质心平均 cos={intra:.3f} (<<1 即未塌缩) | "
          f"表有效秩 {eff_rank:.1f}/{n_pairs}")


if __name__ == "__main__":
    main()
