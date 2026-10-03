"""风格资产校验 + K=4 token 共线性体检 (CPU, 秒级)。

目的: A/B 两路开跑前先确认
  A 路: exp-std/csv/callig_script_emb_top10.pt  -> 单向量 (C,128)
  B 路: assets/multistyle_k4_top10.pt           -> K=4 查表 (C, K*384)
并算出**最关键的先验指标**: K 个 token 之间的余弦相似度。
  仓库记录 (dit.py:723-745): 旧 v15b 的 K=4 token 余弦 0.884(几乎共线) -> "多模态"是假的,
  attention 没有可寻址内容 -> 退化成昂贵的全局调制, 实测只值 +0.0016。
  若我们的新 K4 资产同样共线, B 路大概率重演该结论 (可提前判读, 不必等 100k)。
"""
import os
import sys

import torch

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")

K_EXP, D_EXP, C_EXP = 4, 384, 23


def load_any(p):
    print(f"\n--- {p} ---")
    if not os.path.exists(p):
        print("  ✗ 不存在")
        return None
    d = torch.load(p, map_location="cpu")
    if isinstance(d, dict):
        for k, v in d.items():
            if torch.is_tensor(v):
                print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
            else:
                print(f"  {k}: {type(v)}")
    elif torch.is_tensor(d):
        print(f"  tensor: shape={tuple(d.shape)} dtype={d.dtype}")
    else:
        print(f"  {type(d)}")
    return d


def table_of(d):
    """取 (C, K*D) 或 (C, D) 的表。"""
    if torch.is_tensor(d):
        return d.float()
    if isinstance(d, dict):
        for k in ("embedding_table", "weight", "table", "emb"):
            if k in d and torch.is_tensor(d[k]):
                return d[k].float()
        # 退一步: 取第一个 2D tensor
        for v in d.values():
            if torch.is_tensor(v) and v.dim() == 2:
                return v.float()
    return None


def collinearity(tab, K, tag):
    """每类内部 K 个 token 的两两余弦; 返回 (全局均值, 全局最大, 类均值分布)。"""
    C, KD = tab.shape
    D = KD // K
    if D * K != KD:
        print(f"  [{tag}] ✗ 表宽 {KD} 不是 K={K} 的整数倍")
        return None
    if K < 2:
        print(f"  [{tag}] K=1 -> 无簇内两两余弦可算 (单向量路径, 跳过)")
        n1 = tab / tab.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        s1 = n1 @ n1.T
        off1 = ~torch.eye(C, dtype=torch.bool)
        print(f"  [{tag}] 跨类余弦(基线分离度): 均值={s1[off1].mean():.4f} "
              f"最大={s1[off1].max():.4f}  ← 应接近 0 (健康)")
        return None
    t = tab.view(C, K, D)
    n = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    sim = n @ n.transpose(1, 2)                      # (C,K,K)
    off = ~torch.eye(K, dtype=torch.bool)
    vals = sim[:, off]                               # (C, K*(K-1))
    print(f"  [{tag}] C={C} K={K} D={D}")
    print(f"  [{tag}] token 两两余弦: 均值={vals.mean():.4f}  最大={vals.max():.4f}  "
          f"中位={vals.median():.4f}")
    print(f"  [{tag}] token L2 范数: 均值={t.norm(dim=-1).mean():.4f} "
          f"(std={t.norm(dim=-1).std():.4f})")
    m = float(vals.mean())
    if m > 0.9:
        print(f"  [{tag}] ✗✗ 高度共线 (均值 {m:.3f} > 0.9) —— 与仓库记录的 0.884 同病,"
              f" K 个 token 不可区分, B 路大概率重演 '+0.0016'")
    elif m > 0.7:
        print(f"  [{tag}] ⚠ 明显共线 ({m:.3f}) —— 可寻址性弱, B 路预期收益低")
    else:
        print(f"  [{tag}] ✓ 有区分度 ({m:.3f}) —— 与历史 v15b/v15c 的情况不同, 值得一跑")
    return m


print("=" * 78)
print("[A 路] 单向量风格表 (SupCon 冻结)")
dA = load_any("exp-std/csv/callig_script_emb_top10.pt")
tA = table_of(dA)
if tA is not None:
    print(f"  表 shape = {tuple(tA.shape)}  (期望 ({C_EXP},128) 或含 null 行 ({C_EXP+1},128))")
    collinearity(tA.unsqueeze(1).expand(-1, 1, -1).reshape(tA.shape[0], -1), 1, "A单向量")
    if tA.shape[0] > 1:
        n = tA / tA.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        s = n @ n.T
        off = ~torch.eye(tA.shape[0], dtype=torch.bool)
        print(f"  [A路] 跨类余弦(基线分离度): 均值={s[off].mean():.4f} "
              f"最大={s[off].max():.4f}  ← 应接近 0 (健康)")

print("\n" + "=" * 78)
print("[B 路] K=4 多模态风格表")
dB = load_any("assets/multistyle_k4_top10.pt")
tB = table_of(dB)
if tB is not None:
    print(f"  表 shape = {tuple(tB.shape)}  (期望 ({C_EXP},{K_EXP}*{D_EXP})={C_EXP*K_EXP*D_EXP})")
    if tB.shape[1] == K_EXP * D_EXP:
        print("  ✓ 形状与 callig_multi_style_k=4 + callig_embed_dim=384 完全吻合")
    else:
        print(f"  ✗ 形状不符! 需要 K*D = {K_EXP*D_EXP}, 实得 {tB.shape[1]}; "
              f"若列数=K*128={K_EXP*128} 则应把 config 的 callig_embed_dim 改成 128*"
              f"(但那样拼进 KV 时维度会与 hidden 384 不符, 需要投影)")
    collinearity(tB, K_EXP, "B路K4")

print("\n" + "=" * 78)
print("判读口径: B 路的胜负取决于 K=4 token 是否真的可区分。")
print("  余弦 > 0.9  = 与仓库记录的 0.884 同病 -> 预期 +0.001 量级 (仅证明结论可复现)")
print("  余弦 < 0.7  = 与历史情况不同 -> 有真实可寻址内容, B 路有翻盘可能")
