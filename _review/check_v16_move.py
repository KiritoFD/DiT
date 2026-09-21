"""决定性检查：v16 训了 4650 步后，新行到底移动了多少？+ 表行间的区分度。"""
import glob
import os

import torch

os.chdir("/root/Workspace/xy/DiT")

d = torch.load("assets/multistyle_k4_pretrained.pt", map_location="cpu",
               weights_only=False)
emb = d["embedding"].float()
new_init = emb.mean(0)                       # v16 用 row_pt，近似均值
init_norm = new_init.norm().item()
print(f"  新行初始化范数 ≈ {init_norm:.4f}")

fs = sorted(glob.glob("assets/results/v16_fs_*/**/checkpoints/*.pt",
                      recursive=True))
print(f"  找到 {len(fs)} 个 v16 ckpt")
for f in fs[-6:]:
    try:
        ck = torch.load(f, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  ✗ {f}: {e}")
        continue
    sd = ck.get("ema") or ck.get("model")
    if not sd:
        continue
    k = [x for x in sd if "callig_embedder.embedding_table" in x]
    if not k:
        continue
    w = sd[k[0]].float()
    if w.shape[0] < 88:
        print(f"  {os.path.basename(f)}: 表 {tuple(w.shape)} 没扩行")
        continue
    new = w[87]
    dist = (new - new_init).norm().item()
    print(f"  {os.path.basename(os.path.dirname(os.path.dirname(f)))}/"
          f"{os.path.basename(f)}")
    print(f"     新行范数={new.norm():.4f}  与初始化的距离={dist:.6f}  "
          f"({dist / init_norm * 100:.3f}% 相对变化)")

# 表行间区分度
en = emb / emb.norm(dim=1, keepdim=True)
cos = en @ en.T
off = cos[~torch.eye(87, dtype=bool)]
print(f"\n  === v15 表行间余弦相似度 ===")
print(f"     mean={off.mean():.4f}  max={off.max():.4f}  min={off.min():.4f}")
print(f"     -> 越高说明各行越相似、区分度越差")
