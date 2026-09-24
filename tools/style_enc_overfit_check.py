#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""过拟合检查：加载已训的 latent 编码器，对比 train vs 留出准确率。

动机: 训练时 loss 掉到 0.11（≈99% 训练准确率），而留出只有 30%/24% —— 落差极大。
  若是过拟合，那"latent 信息不够"就不是唯一解释，换输入也不一定是正解；
  缩小模型 / 加强正则 / 换监督形式都可能更有效。
"""
import csv
import glob
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.chdir('/root/Workspace/xy/DiT')
sys.path.insert(0, '.')

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
np.random.seed(0)


def rid(p):
    m = re.search(r'(\d+)\.png$', str(p))
    return int(m.group(1)) if m else None


NMAX = 52457
arr = np.zeros((NMAX, 4, 32, 32), np.float16)
has = np.zeros(NMAX, bool)
for f in sorted(glob.glob('data/50k/shards_img/*.npz')):
    z = np.load(f)
    arr[z['img_ids']] = z['latents']
    has[z['img_ids']] = True

tr = list(csv.DictReader(open('assets/train_50k_v2_fixed.csv', encoding='utf-8')))
CALLS = sorted({r.get('calligrapher') for r in tr})
C2I = {c: i for i, c in enumerate(CALLS)}
IDX, CAL, CH = [], [], []
for r in tr:
    i = rid(r.get('image_path', ''))
    if i is None or not has[i]:
        continue
    IDX.append(i)
    CAL.append(C2I[r.get('calligrapher')])
    CH.append(str(r.get('character', '')))
IDX, CAL, CH = np.array(IDX), np.array(CAL), np.array(CH)

rng = np.random.RandomState(0)
chars = np.array(sorted(set(CH)))
rng.shuffle(chars)
test_chars = set(chars[:int(len(chars) * 0.15)].tolist())
m_char = np.array([c in test_chars for c in CH])
ridx = rng.permutation(len(IDX))
m_rand = np.zeros(len(IDX), bool)
m_rand[ridx[:int(len(IDX) * 0.1)]] = True
tr_final = (~m_char) & (~m_rand)


class Enc(nn.Module):
    def __init__(self, nc=len(CALLS), d=256, ch=96):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(4, ch, 3, 1, 1), nn.GELU(), nn.GroupNorm(8, ch),
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.GELU(), nn.GroupNorm(8, ch * 2),
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1), nn.GELU(), nn.GroupNorm(16, ch * 4),
            nn.Conv2d(ch * 4, ch * 4, 3, 1, 1), nn.GELU(), nn.GroupNorm(16, ch * 4),
            nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(ch * 4, d)
        self.clf = nn.Linear(d, nc)

    def emb(self, x):
        return F.normalize(self.head(self.body(x).flatten(1)), dim=-1)


model = Enc().to(DEV)
sd = torch.load('assets/style_enc_latent.pt', map_location=DEV)
model.load_state_dict(sd)
model.eval()
print('已加载 assets/style_enc_latent.pt (参数 %s)' % format(sum(p.numel() for p in model.parameters()), ','))


@torch.no_grad()
def acc_of(mask, tag, n=None):
    ii = IDX[mask]
    if n:
        ii = ii[:n]
    zs = []
    for s in range(0, len(ii), 2048):
        x = torch.from_numpy(arr[ii[s:s + 2048]].astype(np.float32)).to(DEV)
        zs.append(model.emb(x).cpu().numpy())
    Z = np.concatenate(zs)
    lg = model.clf(torch.from_numpy(Z).to(DEV)).cpu().numpy()
    p = lg.argmax(1)
    yy = CAL[mask][:len(ii)]
    v = (p == yy).mean()
    print('  %-26s %6.2f%%  (n=%d)' % (tag, v * 100, len(ii)), flush=True)
    return v


print('\n准确率对比（随机基线 %.1f%%）' % (100 / len(CALLS)))
acc_of(tr_final, '【训练样本】', n=8000)
acc_of(m_rand & (~m_char), '留出·样本新/字见过')
acc_of(m_char, '留出·样本新/字没见过')
print('\n判读: 训练样本准确率接近 100% 而留出只有 ~30% => **过拟合**，不是"latent 信息不够"。')
print('      那么加容量/加步数都没用（已验证平台化）；该走 缩小模型/加强正则/换监督形式。')
