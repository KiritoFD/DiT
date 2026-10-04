import os

import numpy as np
import json
import torch
import torch.nn.functional as F

# 便携: 默认相对仓库根, 可用环境变量覆盖 (原来写死另一台机器的绝对路径 /home/ds/...)
DIR = os.environ.get("TRIPLE_TABLE_DIR", "assets/triple_tables_best_minimal")

def audit_table(name, npy_file, idx_file):
    w = np.load(f"{DIR}/{npy_file}")
    with open(f"{DIR}/{idx_file}", encoding="utf-8") as f:
        meta = json.load(f)
    classes = meta.get("classes", [])
    
    t = torch.from_numpy(w).float()
    N, D = t.shape
    
    # 1. 模长分布
    norms = t.norm(dim=-1)
    norm_mean = norms.mean().item()
    norm_std = norms.std().item()
    norm_min = norms.min().item()
    norm_max = norms.max().item()
    
    # 2. SVD 与有效秩 (Roy & Vetterli 2007)
    t_centered = t - t.mean(dim=0, keepdim=True)
    U, S, V = torch.linalg.svd(t_centered, full_matrices=False)
    p = S / S.sum()
    p = p[p > 1e-12]
    entropy = -(p * torch.log(p)).sum().item()
    erank = np.exp(entropy)
    
    cum_var = (S**2).cumsum(dim=0) / (S**2).sum()
    d90 = (cum_var < 0.90).sum().item() + 1
    d95 = (cum_var < 0.95).sum().item() + 1
    d99 = (cum_var < 0.99).sum().item() + 1
    
    # 3. 类间余弦相似度矩阵
    t_norm = F.normalize(t, dim=-1)
    cos_mat = (t_norm @ t_norm.T)
    mask = ~torch.eye(N, dtype=torch.bool)
    off_diag = cos_mat[mask]
    cos_mean = off_diag.mean().item()
    cos_abs_mean = off_diag.abs().mean().item()
    cos_std = off_diag.std().item()
    cos_min = off_diag.min().item()
    cos_max = off_diag.max().item()
    
    print(f"\n==================================================")
    print(f"【最小维度黄金表全面体检验收: {name}】 (形: {N} 行 × {D} 列)")
    print(f"==================================================")
    print(f"1. 向量模长 (L2 Norm):")
    print(f"   均值={norm_mean:.4f} ± {norm_std:.4f} | 极值=[{norm_min:.4f}, {norm_max:.4f}]")
    print(f"2. 本征维度与信息纯度 (Intrinsic Dimensionality):")
    print(f"   数学有效秩 (Effective Rank): {erank:.2f} / {min(N, D)}")
    print(f"   解释 90% 方差所需维数:      {d90} 维 / 总 {D} 维")
    print(f"   解释 95% 方差所需维数:      {d95} 维 / 总 {D} 维")
    print(f"   解释 99% 方差所需维数:      {d99} 维 / 总 {D} 维")
    print(f"3. 类间正交与区分度 (Pairwise Cosine Similarity):")
    print(f"   类间平均余弦: {cos_mean:+.4f} (平均绝对余弦: {cos_abs_mean:.4f} ± {cos_std:.4f})")
    print(f"   类间相似度跨度: [{cos_min:+.4f}, {cos_max:+.4f}]")
    
    if N <= 10:
        print("   两两余弦详细矩阵:")
        for r_i in range(N):
            row_str = " ".join(f"{cos_mat[r_i, c_j].item():+.3f}" for c_j in range(N))
            print(f"     类 {r_i}: {row_str}")

    grade = "A+" if (cos_abs_mean < 0.15 and erank / min(N, D) > 0.8) else ("A" if cos_abs_mean < 0.25 else "B")
    print(f"   ★ 最终综合评级: 【{grade} 级】")
    return {
        "name": name, "shape": [N, D], "norm_mean": norm_mean, "erank": erank,
        "d90": d90, "d95": d95, "d99": d99, "cos_mean": cos_mean, "cos_abs_mean": cos_abs_mean, "grade": grade
    }

res_fnt = audit_table("最小书体表 (Script / Font Table)", "font_table.npy", "font_index.json")
res_cal = audit_table("最小书家表 (Calligrapher Table)", "callig_table.npy", "callig_index.json")
res_chr = audit_table("最小汉字表 (Character / Content Table)", "char_table.npy", "char_index.json")
