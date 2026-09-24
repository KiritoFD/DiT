#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比学习可行性探针 v2：向量化 InfoNCE，让 GPU 真正跑满。

v1 的三个问题（都记在注释里，避免再犯）：
  ① 下标混用：池里存训练行下标，却直接索引按缓存行排的 X -> 特征张冠李戴，
     命中率掉到 0.75x（比随机还低）。
  ② loss 写成 mean(logsumexp(pos, negs))，**漏了 − pos/τ** —— 那不是 InfoNCE，
     而是把 query 和正负样本一起推开（loss 停在 4.1 不降）。
  ③ 性能：逐样本 Python 循环 -> 每步 1000+ 个小 kernel，GPU 13% / 70W（上限 450W）。

本版：正/负样本各一次 gather + 一次 bmm，batch 1024，特征常驻显存。
"""
import csv
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.chdir('/root/Workspace/xy/DiT')
sys.path.insert(0, '.')

CACHE = 'data/dino_cache/50k_v1'
FEAT_NPY = 'assets/dino_meanstd_50k.npy'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
MAXC = 12
NB, KNEG, STEPS, TAU = 1024, 8, 1500, 0.1
torch.manual_seed(0)
np.random.seed(0)


def rid(p):
    m = re.search(r'(\d+)\.png$', str(p))
    return int(m.group(1)) if m else None


ids = np.load(CACHE + '/ids.npy')
pos = {int(i): k for k, i in enumerate(ids)}
X = np.load(FEAT_NPY)
print('[1] 特征 %s' % (X.shape,), flush=True)

tr = list(csv.DictReader(open('assets/train_50k_v2_fixed.csv', encoding='utf-8')))
ev = list(csv.DictReader(open('assets/eval_v13_strict_fixed.csv', encoding='utf-8')))[:249]


def build(rows):
    idx, cal, ch = [], [], []
    for r in rows:
        i = rid(r.get('image_path', ''))
        k = pos.get(i) if i is not None else None
        if k is None:
            continue
        idx.append(k)
        cal.append(r.get('calligrapher'))
        ch.append(str(r.get('character', '')))
    return np.array(idx), np.array(cal), np.array(ch)


TR_I, TR_C, TR_H = build(tr)
EV_I, EV_C, EV_H = build(ev)
print('[1] 训练 %d / 留出 %d' % (len(TR_I), len(EV_I)), flush=True)

# ⚠ 两套下标必须分开：训练循环用训练行(配合 XT=X[TR_I])，检索评分用缓存行(直接索引 X)
CAL_OF, by_cal, by_char, by_char_cache = {}, {}, {}, {}
for j in range(len(TR_I)):
    cr = int(TR_I[j])
    CAL_OF[cr] = TR_C[j]
    by_cal.setdefault(TR_C[j], []).append(j)
    by_char.setdefault(TR_H[j], []).append(j)
    by_char_cache.setdefault(TR_H[j], []).append(cr)

POS, NEG = [], []
for j in range(len(TR_I)):
    pc = [x for x in by_cal.get(TR_C[j], []) if TR_H[x] != TR_H[j]]
    POS.append(np.array(pc) if pc else np.array([j]))
    nc = [x for x in by_char.get(TR_H[j], []) if TR_C[x] != TR_C[j]]
    NEG.append(np.array(nc) if nc else np.array([j]))
print('[2] 配对表建好（书家 %d / 字 %d）' % (len(by_cal), len(by_char)), flush=True)


class Proj(nn.Module):
    def __init__(self, din=768, d=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, 512), nn.GELU(),
                                 nn.Linear(512, 256), nn.GELU(), nn.Linear(256, d))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def make_eval():
    qs, pools, qcal = [], [], []
    for a in range(len(EV_I)):
        ch = EV_H[a]
        c = [j for j in by_char_cache.get(ch, []) if j != EV_I[a]]
        if len(c) > MAXC:
            st = len(c) / MAXC
            c = [c[int(t * st)] for t in range(MAXC)]
        if len(c) < 2:
            continue
        qs.append(EV_I[a])
        pools.append(np.array(c))
        qcal.append(EV_C[a])
    return np.array(qs), pools, np.array(qcal)


QID, POOLS, QCAL = make_eval()
print('[3] 可评估 %d 列, 池均 %.1f' % (len(QID), np.mean([len(p) for p in POOLS])), flush=True)


def score(Z, tag):
    hit = tot = 0
    base = 0.0
    for k in range(len(QID)):
        cand = POOLS[k]
        best = cand[int(np.argmax(Z[cand] @ Z[QID[k]]))]
        hit += int(CAL_OF.get(int(best)) == QCAL[k])
        base += sum(1 for c in cand if CAL_OF.get(int(c)) == QCAL[k]) / len(cand)
        tot += 1
    h, b = hit / tot, base / tot
    print('  %-24s 命中 %6.1f%%  基线 %5.1f%%  富集 %.2fx  (n=%d)'
          % (tag, h * 100, b * 100, h / b, tot), flush=True)
    return h / b


def norm(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)


print('\n[3] 免训练基线（留出集, GT->GT）')
e_raw = score(norm(X), 'DINO mean+std 原始')

print('\n[4] 训投影（向量化 content-matched InfoNCE, batch=%d, %d 步）...' % (NB, STEPS), flush=True)
model = Proj().to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
XT = torch.from_numpy(X[TR_I]).to(DEV)          # 常驻显存
t0 = time.time()
for step in range(STEPS):
    bi = np.random.randint(0, len(TR_I), NB)
    # 正样本: 每个 query 抽 1 个「同书家异字」
    pj = np.array([POS[j][np.random.randint(len(POS[j]))] for j in bi])
    # 负样本: 每个 query 抽 K 个「同字异书家」 -> (NB, K)
    nj = np.stack([NEG[j][np.random.choice(len(NEG[j]), KNEG, replace=len(NEG[j]) < KNEG)]
                   for j in bi])
    zq = model(XT[bi])                                    # (NB, D)
    zp = model(XT[pj])                                    # (NB, D)
    zn = model(XT[nj.reshape(-1)]).view(NB, KNEG, -1)     # (NB, K, D)
    s_pos = (zq * zp).sum(-1) / TAU                       # (NB,)
    s_neg = torch.bmm(zn, zq.unsqueeze(-1)).squeeze(-1) / TAU       # (NB, K)
    # ★ InfoNCE = logsumexp(pos, negs) − pos/τ
    loss = (torch.logsumexp(torch.cat([s_pos.unsqueeze(-1), s_neg], 1), 1) - s_pos).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if (step + 1) % 250 == 0:
        print('    step %4d  loss %.4f  %.0fs' % (step + 1, float(loss), time.time() - t0),
              flush=True)

model.eval()
with torch.no_grad():
    Zall = norm(model(torch.from_numpy(X).to(DEV)).cpu().numpy())
print('\n[5] 对比学习后（留出集, GT->GT）')
e_proj = score(Zall, 'DINO + InfoNCE 投影')

print('\n=== 结论 ===')
print('  免训练 %.2fx  ->  对比学习后 %.2fx  （提升 %.2fx）' % (e_raw, e_proj, e_proj / e_raw))
print('  参考: 模型当前输出的富集 ~1.5x（见 93_samechar_nn_diag.md）')
print('  判读: >4x = 这条设计可用; ~2.5x = 表示里没有可学的风格结构, 设计要改')
torch.save(model.state_dict(), 'assets/style_proj_dino_50k.pt')
print('  投影已存 assets/style_proj_dino_50k.pt')
