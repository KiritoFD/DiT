# -*- coding: utf-8 -*-
"""对比 45 表 vs 87 表的质量: pairwise cos / 有效秩 / 塌缩指标。"""
import torch
import torch.nn.functional as F

for name in ("assets/callig_emb_pretrained_50k.pt",       # v13: 45 书家
             "assets/callig_script_emb_pretrained.pt"):    # v14: 87 书家×书体
    e = torch.load(name, map_location="cpu", weights_only=False)
    if isinstance(e, dict):
        print(f"\n{name}: dict keys = {list(e.keys())}")
        for k in ("emb", "weight", "embedding", "table"):
            if k in e and hasattr(e[k], "shape"):
                e = e[k]
                break
        else:
            # 打印所有 value 的 type/shape 找到正确的
            for k, v in e.items():
                print(f"  {k}: {type(v).__name__} "
                      f"{v.shape if hasattr(v, 'shape') else ''}")
            continue
    if not hasattr(e, "shape"):
        print(f"\n{name}: 无法提取 tensor, type={type(e)}")
        continue
    print(f"\n{'=' * 60}")
    print(f"{name}   shape={tuple(e.shape)}")
    en = F.normalize(e.float(), dim=-1)
    n = e.shape[0]
    cos = en @ en.T
    off = cos[~torch.eye(n, dtype=torch.bool)]
    print(f"  pairwise cos: mean={off.mean():.4f}  std={off.std():.4f}  "
          f"max={off.max():.4f}  min={off.min():.4f}")
    # 有效秩 (entropy-based)
    sv = torch.linalg.svdvals(e.float())
    p = sv / sv.sum().clamp_min(1e-8)
    p = p.clamp_min(1e-12)
    erank = (-p * p.log()).sum().exp()
    print(f"  有效秩 (entropy): {erank:.1f} / {min(n, e.shape[1])}")
    print(f"  塌缩判定: cos_mean < 0.323 为正常; > 0.5 = 塌缩")
    # nearest neighbor cos (每行最近的其他行)
    cos_off = cos.clone()
    cos_off.fill_diagonal_(-2)
    nn_cos = cos_off.max(dim=1).values
    print(f"  最近邻 cos: mean={nn_cos.mean():.4f}  (高=有近似重复行)")
