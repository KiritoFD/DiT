"""从已有海报里算 Vendi Score —— "有效不同变体数"。

Vendi Score (Friedman & Dieng, TMLR 2023):
    VS = exp(-sum_i λ_i log λ_i),  λ = eigvals(K/n),  K = 相似度核
  - 范围 [1, n]: VS=1 所有样本完全相同(完全坍塌); VS=n 两两完全不同
  - **不需要参考数据集** -> 正好绕开"真迹不可比"的问题
  - 可解释为"有效不同元素数"
海报布局: 行=条件, 列=K 个生成 + 最后一列 GT
"""
import os, sys
import numpy as np
from PIL import Image

os.chdir("/root/Workspace/xy/DiT")

TH = 256


def vs_from_sims(S):
    """S: n x n 相似度矩阵(对角=1)。返回 Vendi Score。"""
    n = S.shape[0]
    K = S / n
    lam = np.linalg.eigvalsh(K)
    lam = np.clip(lam, 1e-12, None)
    lam = lam / lam.sum()
    return float(np.exp(-(lam * np.log(lam)).sum()))


def ssim_sim(a, b, win=7):
    from scipy.ndimage import uniform_filter
    x, y = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mx, my = uniform_filter(x, win), uniform_filter(y, win)
    sx2 = uniform_filter(x * x, win) - mx ** 2
    sy2 = uniform_filter(y * y, win) - my ** 2
    sxy = uniform_filter(x * y, win) - mx * my
    s = (((2 * mx * my + c1) * (2 * sxy + c2)) /
         ((mx ** 2 + my ** 2 + c1) * (sx2 + sy2 + c2)))
    return float(np.clip(s.mean(), -1, 1))


def run(tag):
    p = f"assets/diversity_poster_{tag}.png"
    if not os.path.exists(p):
        print(f"  {p} 不存在, 跳过"); return
    im = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
    H, W = im.shape
    nrow, ncol = H // TH, W // TH
    k = ncol - 1                     # 最后一列是 GT
    print(f"=== {tag} ===  海报 {nrow} 条件 x {ncol} 列 (K={k} 生成 + GT)")
    vs_list, intra_list = [], []
    for r in range(nrow):
        cells = [im[r * TH:(r + 1) * TH, c * TH:(c + 1) * TH] for c in range(k)]
        S = np.ones((k, k))
        for i in range(k):
            for j in range(i + 1, k):
                S[i, j] = S[j, i] = ssim_sim(cells[i], cells[j])
        vs_list.append(vs_from_sims(S))
        intra_list.append(1.0 - (S.sum() - k) / (k * (k - 1)))
    vs = np.array(vs_list); it = np.array(intra_list)
    print(f"  Vendi Score (有效不同变体数, 上界 {k}): "
          f"mean={vs.mean():.3f}  med={np.median(vs):.3f}  "
          f"min={vs.min():.3f}  max={vs.max():.3f}")
    print(f"  归一化 VS/K: {vs.mean()/k:.3f}")
    print(f"  对照 intra (1-平均SSIM): mean={it.mean():.4f}")
    print(f"  坍塌条件 (VS < 1.1, 即 4 张几乎一样): {(vs < 1.1).sum()}/{nrow}")
    print(f"  高分条件 (VS > 3.0): {(vs > 3.0).sum()}/{nrow}")
    # 参考: 真迹能到多少? 用 GT 列与"随机配对"不可得, 这里只给模型的绝对解读
    return vs


print("Vendi Score 解读: 1.0 = 4 张完全相同(坍塌); 4.0 = 4 张两两完全不同")
print()
for t in ["v12_intra_cfg07", "v12_intra_cfg10", "v12_d8"]:
    run(t)
    print()
