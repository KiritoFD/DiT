#!/usr/bin/env python
"""诊断: 表里 cos≈1 的行对, 是「字本来就几乎一样」还是「训练搞坏了」。

判定逻辑:
  对每对塌缩行 (g_i, g_j), 取它们**原始 DINO 特征**的类中心 (不过投影头) 算余弦 c_raw;
  表行余弦 c_tab。
  - c_raw 也 ≈1  -> 字在 DINO 空间本来就几乎同一个东西, 表是忠实的 ✓
  - c_raw 很低但 c_tab≈1 -> 表把不同的字挤到一起 = 训练问题 ✗
再算全局: 表余弦 vs DINO 中心余弦的秩相关 (表是否忠实于特征空间)。
"""
import json
import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CH = 7026

idx = json.load(open("assets/char_script_supcon_index.json", encoding="utf-8"))
glyphs = idx["glyphs"]
W = torch.from_numpy(np.load("assets/char_script_supcon_table.npy")).float()[glyphs].to(DEV)
Wn = F.normalize(W, dim=-1)
lab = torch.tensor(glyphs, device=DEV)

# ---- 原始 DINO 特征类中心 (不过投影头; 投影头训练完已无意义) ----
z = np.load(idx["npz"])
if "glyph_ids" in z.files:
    g_np = z["glyph_ids"].astype(np.int64)
else:
    g_np = z["scripts"].astype(np.int64) * CH + z["chars"].astype(np.int64)
feat = torch.from_numpy(z["feat"].astype(np.float32)).to(DEV)

from collections import defaultdict
by = defaultdict(list)
for i, g in enumerate(g_np.tolist()):
    by[g].append(i)
centers, nsamp = {}, {}
for g, ii in by.items():
    centers[g] = F.normalize(feat[ii].mean(0), dim=-1)
    nsamp[g] = len(ii)
glist = sorted(centers)
C = torch.stack([centers[g] for g in glist])                      # (S, d)
gid = torch.tensor(glist, device=DEV)
print(f"[data] 表行 {len(glyphs)}, DINO 中心 {len(glist)}, feat_dim={feat.shape[1]}")

# ---- 全表行间余弦 (分块, 7193x7193 没问题但省点显存) ----
S = Wn @ Wn.T
S.fill_diagonal_(-1)
iu = torch.triu_indices(S.shape[0], S.shape[0], 1)
pair_cos = S[iu[0], iu[1]]
_s = pair_cos[torch.randperm(pair_cos.shape[0], device=DEV)[:2000000]]
print(f"\n[全局] 表行间余弦 ({pair_cos.shape[0]} 对; 分位数基于 200 万对抽样): "
      f"mean={pair_cos.mean():.4f} p99={_s.quantile(0.99):.4f} "
      f"p999={_s.quantile(0.999):.4f} max={pair_cos.max():.4f}")
order = pair_cos.argsort(descending=True)

def decode(g):
    return g // CH, g % CH

print("\n[Top-25 塌缩行对]  表cos | DINO中心cos | 判定 | glyph对 (script,char) 样本数")
n_bad = 0
rows = []
for t in order[:25].tolist():
    i, j = int(iu[0][t]), int(iu[1][t])
    gi, gj = glist[i], glist[j]
    c_tab = float(pair_cos[t])
    c_raw = float((centers[gi] * centers[gj]).sum())
    si, ci = decode(gi); sj, cj = decode(gj)
    verdict = "✓ 字本来就一样" if c_raw > 0.9 else ("⚠ 有点像" if c_raw > 0.7 else "✗✗ 表挤错了")
    if c_raw <= 0.7:
        n_bad += 1
    rows.append(f"  {c_tab:.4f} | {c_raw:.4f} | {verdict} | "
                f"({si},{ci})[{nsamp[gi]}张] vs ({sj},{cj})[{nsamp[gj]}张]"
                + ("  ← 同字不同书家!" if ci == cj else ""))
print("\n".join(rows))

# ---- 全局忠实度: 表余弦 vs DINO 中心余弦 的相关性 (抽样 20 万对) ----
iu = iu.to(DEV)
sel = torch.randperm(pair_cos.shape[0], device=DEV)[:200000]
i_s, j_s = iu[0][sel], iu[1][sel]
c_tab_s = pair_cos[sel]
c_raw_s = (C[i_s] * C[j_s]).sum(-1)
pear = torch.corrcoef(torch.stack([c_tab_s, c_raw_s]))[0, 1].item()
print(f"\n[忠实度] 表cos vs DINO中心cos (20万对): Pearson r = {pear:.4f}")
print("  r 接近 1 = 表忠实复刻了 DINO 空间的相似结构 (训练没毛病);")

# ---- 数据侧重复字槽统计: DINO 中心 cos>0.999 的 glyph 对 (与表无关, 纯数据) ----
Cm = C @ C.T
iu2 = torch.triu_indices(C.shape[0], C.shape[0], 1, device=DEV)
raw_pair = Cm[iu2[0], iu2[1]]
dup_mask = raw_pair > 0.999
n_dup = int(dup_mask.sum().item())
inv = torch.unique(torch.stack([iu2[0][dup_mask], iu2[1][dup_mask]]))
n_g = int(inv.numel())
print(f"\n[数据重复] DINO中心cos>0.999 的 glyph 对: {n_dup} 对, 涉及 {n_g} 个 glyph 槽位")
if n_dup:
    ex = []
    for t in raw_pair.argsort(descending=True)[:10].tolist():
        gi, gj = glist[int(iu2[0][t])], glist[int(iu2[1][t])]
        si, ci = decode(gi); sj, cj = decode(gj)
        ex.append(f"    ({si},{ci}) vs ({sj},{cj}) raw={float(raw_pair[t]):.6f}")
    print("\n".join(ex))

# ---- 高余弦对的总量统计 ----
for thr in (0.9, 0.99, 0.999):
    n = int((pair_cos > thr).sum().item())
    m_raw = float(c_raw_s[(c_tab_s > thr)].mean()) if n and thr >= 0.9 and int((c_tab_s > thr).sum()) else float("nan")
    print(f"  表cos>{thr}: {n} 对" + (f"  (这些对的 DINO 中心 cos 均值 = {m_raw:.4f})" if n <= 5000 else ""))

print(f"\n[结论] Top-25 里 '表挤错了' (DINO 中心 cos<=0.7) 的对数: {n_bad}")
