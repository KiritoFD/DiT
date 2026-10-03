# -*- coding: utf-8 -*-
"""生成 PCA 降维的标准字形 DINO char 表 (7026 x 384)。

背景
----
docs/system/25 结论：DINO 标准字形特征可直接直通注入，无需可训练投影。
DINO 原始 768d，模型 hidden=384，需零参数降维。实测判别性 AUC（形近/随机）：
  - 768 原始          : 0.820
  - linear interpolate: 0.824
  - truncate 前384    : 0.833
  - PCA 投影到 384    : 0.910   <-- 最好（保留主方差方向，去噪）

本脚本在「有真值覆盖的行」上算 PCA（避免 fallback 均值行污染主成分），
再把投影应用到整表（含 fallback），输出 7026x384 冻结表。

产出: _sync_work/std_dino_char_table_384_pca.npy (7026, 384)
      _sync_work/std_dino_pca_meta.json (记录投影矩阵 P 与均值, 便于复现/复用)
"""
import os, sys, csv, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
SRC = "_sync_work/std_dino_char_table_768.npy"
OUT = "_sync_work/std_dino_char_table_384_pca.npy"
META = "_sync_work/std_dino_pca_meta.json"
K = 384

table = np.load(SRC).astype(np.float32)          # (7026, 768)
print("src table:", table.shape)

# 标记哪些行是「有真值覆盖」(非 fallback)：由构建脚本记录，这里用简单启发：
# fallback 行在构建时 = 全局均值向量（几乎所有 fallback 行相同）。
# 更稳：直接检测行是否高度接近全局均值。
mean_all = table.mean(0, keepdims=True)
# 覆盖行：与该行自身是否属于"所有覆盖行的均值"判断不可靠。
# 改为：用原始 768 表（构建时记录）已知覆盖统计。这里用近似：行与全局均值余弦 >0.99 视为 fallback。
import numpy.linalg as la
cos = (table * mean_all).sum(1) / (la.norm(table, axis=1) * la.norm(mean_all) + 1e-9)
fallback_mask = cos > 0.999
covered = ~fallback_mask
print(f"detected covered rows: {covered.sum()}/{len(table)}  (fallback: {fallback_mask.sum()})")

X = table[covered]                                # (n_covered, 768)
Xc = X - X.mean(0, keepdims=True)                 # center
# SVD on (n,768): 保留前 K 个右奇异向量 -> 投影矩阵 P (768, K)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False) # Vt: (768, 768) 取前K行
P = Vt[:K].T                                      # (768, K)
center = X.mean(0)                                # (768,)

# 白化可去相关，但会改变幅度/方向可比性。这里不做白化，仅投影保方差。
# 投影后按行归一化（embedder 期望单位向量）。
E_pca = (table - center) @ P                       # (7026, 384)
E_pca = E_pca / (np.linalg.norm(E_pca, axis=1, keepdims=True) + 1e-8)

np.save(OUT, E_pca.astype(np.float32))
with open(META, "w") as f:
    json.dump({"src": SRC, "k": K, "center": center.tolist(),
               "P": P.tolist(), "n_covered": int(covered.sum())}, f)
print(f"saved {OUT} shape={E_pca.shape}  meta={META}")

# 校验：判别性 AUC
charid2char = {}
for r in csv.DictReader(open("assets/train_fame.csv", encoding="utf-8")):
    charid2char[int(r["character_id"])] = r["character"]
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
    s=np.concatenate([pos,neg]); lab=np.concatenate([np.ones(len(pos)),np.zeros(len(neg))])
    o=np.argsort(s); ls=lab[o]; r=np.where(ls==1)[0]+1
    return float((r.sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)+1e-9))
def ev(E):
    d={}
    for cid,v in enumerate(E):
        ch=charid2char.get(cid)
        if ch and len(ch)==1: d[ch]=v
    pos=np.array([d[a]@d[b] for a,b in SIM if a in d and b in d])
    rng=np.random.default_rng(0); neg=[]
    chs=list(d.keys())
    while len(neg)<len(pos)*10:
        a,b=rng.choice(chs,2,replace=False)
        if (a,b) in SIM or (b,a) in SIM: continue
        neg.append(d[a]@d[b])
    return auc(pos,np.array(neg))
print(f"PCA-384 AUC = {ev(E_pca):.4f}")
for a,b in [("土","士"),("王","玉"),("未","末")]:
    ca=next((c for c,cc in charid2char.items() if cc==a),None)
    cb=next((c for c,cc in charid2char.items() if cc==b),None)
    if ca is None or cb is None: continue
    print(f"  {a}/{b}: cos={float(E_pca[ca]@E_pca[cb]):.3f}")
