import os
import numpy as np
import json
import torch
import torch.nn.functional as F

# 便携: 默认相对仓库根, 可用环境变量覆盖 (原来写死另一台机器的绝对路径 /home/ds/...)
DIR = os.environ.get("TRIPLE_TABLE_DIR", "assets/triple_tables_best_minimal")

def audit_table(name, npy_file, idx_file):
    w = np.load(f"{DIR}/{npy_file}")
    info = json.load(open(f"{DIR}/{idx_file}", encoding="utf-8"))
    classes = info.get("classes", [])
    t = torch.from_numpy(w).float()
    norms = torch.norm(t, dim=-1)
    
    # 归一化后算余弦相似度矩阵
    t_norm = F.normalize(t, p=2, dim=-1)
    sim = torch.mm(t_norm, t_norm.t())
    
    N = sim.shape[0]
    mask = ~torch.eye(N, dtype=torch.bool)
    off_diag = sim[mask]
    
    # 有效秩 (Effective Rank / Shannon Entropy of singular values)
    _, S, _ = torch.svd(t)
    S_norm = S / S.sum()
    ent = -torch.sum(S_norm * torch.log(S_norm + 1e-12))
    erank = float(torch.exp(ent))
    
    print(f"=== Table: {name} ===")
    print(f"  Shape: {list(w.shape)} (Classes: {len(classes)})")
    print(f"  L2 Norms: Mean={float(norms.mean()):.4f}, Min={float(norms.min()):.4f}, Max={float(norms.max()):.4f}")
    print(f"  Off-diag Cosine: Mean={float(off_diag.mean()):.4f}, Max={float(off_diag.max()):.4f}, Min={float(off_diag.min()):.4f}")
    print(f"  Effective Rank: {erank:.2f} / {min(w.shape)}")
    
    # 如果类间余弦均值接近0且有效秩高，判定为优
    grade = "A+" if erank >= min(w.shape) * 0.7 and abs(float(off_diag.mean())) < 0.15 else "B"
    if name == "Font/Script" and abs(float(off_diag.mean())) < 0.05: grade = "A"
    print(f"  [Rating Grade]: {grade}\n")

if __name__ == "__main__":
    audit_table("Font/Script", "font_table.npy", "font_index.json")
    audit_table("Calligrapher", "callig_table.npy", "callig_index.json")
    audit_table("Character", "char_table.npy", "char_index.json")
