# -*- coding: utf-8 -*-
"""验证插值/降维后 DINO 信号是否保留（外形一致性判别性 AUC）。

对比 768 表 vs 各降维方式(到384)：
  - fixed linear interpolate (当前方案)
  - truncate 前384维
  - 相邻平均 (768->384, 每2维平均)
  - PCA 投影到 384 (固定矩阵, 构建时算好, 冻结)
指标: 形近字对 vs 随机对 余弦 AUC (与 docs/25 相同方法)。
"""
import os, sys, json, csv
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)

# 加载 768 表
table = np.load("_sync_work/std_dino_char_table_768.npy").astype(np.float32)
print("table:", table.shape)

# char_id -> char
charid2char = {}
for r in csv.DictReader(open("assets/train_fame.csv", encoding="utf-8")):
    charid2char[int(r["character_id"])] = r["character"]

def norm_row(E):
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)

def to_chardict(E):
    d = {}
    for cid, v in enumerate(E):
        ch = charid2char.get(cid)
        if ch and len(ch) == 1:
            d[ch] = v
    return d

SIM = [("土","士"),("大","太"),("人","入"),("日","曰"),("天","夫"),("刀","力"),
       ("申","由"),("王","玉"),("牛","午"),("己","已"),("未","末"),("木","林"),
       ("休","体"),("侯","候"),("风","凤"),("几","凡"),("戊","戍"),("己","巳"),
       ("亳","毫"),("宋","宗"),("东","车"),("冈","同"),("三","王"),("十","干"),
       ("大","天"),("夫","天"),("犬","太"),("人","个"),("手","毛"),("毛","手"),
       ("子","孑"),("戊","戌"),("戍","戌"),("刀","刁"),("万","方"),("鸟","乌"),
       ("贝","见"),("龙","尤"),("失","矢"),("句","向"),("因","困"),("同","回"),
       ("问","间"),("门","们"),("口","曰"),("田","由"),("甲","申"),("白","百"),
       ("吉","古"),("夫","夭"),("天","夭"),("王","主"),("玉","主"),("人","入")]

def auc(pos, neg):
    s = np.concatenate([pos, neg]); lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(s); lab_s = lab[order]
    ranks = np.where(lab_s == 1)[0] + 1
    n_p, n_n = len(pos), len(neg)
    return float((ranks.sum() - n_p*(n_p+1)/2) / (n_p*n_n + 1e-9))

def eval_auc(E):
    d = to_chardict(norm_row(E))
    chars = list(d.keys())
    pos = np.array([d[a] @ d[b] for a, b in SIM if a in d and b in d])
    rng = np.random.default_rng(0)
    neg = []
    while len(neg) < len(pos)*10:
        a, b = rng.choice(chars, 2, replace=False)
        if (a, b) in SIM or (b, a) in SIM: continue
        neg.append(d[a] @ d[b])
    neg = np.array(neg)
    return auc(pos, neg), len(pos)

T = torch.from_numpy(table).float()

# 1. 原始 768
E768 = table
a, n = eval_auc(E768)
print(f"768 原始         : AUC={a:.4f} (pairs={n})")

# 2. linear interp
ti = T.unsqueeze(1)
ti = F.interpolate(ti, size=384, mode="linear", align_corners=False).squeeze(1).numpy()
a, n = eval_auc(ti)
print(f"768->384 linear  : AUC={a:.4f} (pairs={n})")

# 3. truncate 前384
a, n = eval_auc(E768[:, :384])
print(f"768->384 truncate: AUC={a:.4f} (pairs={n})")

# 4. 相邻平均 每2维
E_avg = E768.reshape(E768.shape[0], 384, 2).mean(axis=2)
a, n = eval_auc(E_avg)
print(f"768->384 avg2    : AUC={a:.4f} (pairs={n})")

# 5. PCA 投影到384 (固定矩阵)
Ec = E768 - E768.mean(0, keepdims=True)
# 直接在行维度做主成分不合适；这里对特征维做PCA: 保留前384主成分
# X (N,768), 计算协方差特征向量 -> 投影
U, S, Vt = np.linalg.svd(Ec, full_matrices=False)  # Vt (384_or_768, 768)
k = 384
P = Vt[:k].T  # (768, k)
E_pca = (Ec - 0) @ P
# 白化可选
a, n = eval_auc(E_pca)
print(f"768->384 PCA     : AUC={a:.4f} (pairs={n})")

# 形近字对余弦对比
print("\n形近字对余弦 (768 vs interp384):")
for a_, b_ in [("土","士"),("王","玉"),("水","火"),("未","末")]:
    ca = next((c for c,cc in charid2char.items() if cc==a_), None)
    cb = next((c for c,cc in charid2char.items() if cc==b_), None)
    if ca is None or cb is None: continue
    c768 = float(E768[ca] @ E768[cb] / (np.linalg.norm(E768[ca])*np.linalg.norm(E768[cb])+1e-9))
    c384 = float(ti[ca] @ ti[cb] / (np.linalg.norm(ti[ca])*np.linalg.norm(ti[cb])+1e-9))
    print(f"  {a_}/{b_}: 768={c768:.3f}  interp384={c384:.3f}")
