#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冒烟 Step 1: 用 2.9M latent 编码器给全量 50k 建特征 + 87 pair 质心。

产出:
  assets/rank_feat_latent_50k.npy   (51036, 256)  2.9M 编码器的 embedding（L2 归一化）
  assets/rank_cent87.npy            (87, 256)     pair 质心（L2 归一化）
  assets/rank_feat_meta.npz         ids 数组 + 质心->pair_key 对照
口径与 tools/rank_dino.py 完全一致（同划分、同 CSV），保证可复现。
"""
import csv, glob, os, re, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.chdir('/root/Workspace/xy/DiT')
DEV = 'cuda'


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
IDX, CAL, CH, PAIR = [], [], [], []
for r in tr:
    i = rid(r.get('image_path', ''))
    if i is None or not has[i]:
        continue
    IDX.append(i)
    CAL.append(C2I[r.get('calligrapher')])
    CH.append(str(r.get('character', '')))
    PAIR.append('%s:%s' % (r.get('calligrapher_id'), r.get('script_id')))
IDX, CAL, CH, PAIR = np.array(IDX), np.array(CAL), np.array(CH), np.array(PAIR)
print('[1] 样本 %d' % len(IDX), flush=True)


class Enc(nn.Module):
    def __init__(self, nc=45, d=256, ch=96):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(4, ch, 3, 1, 1), nn.GELU(), nn.GroupNorm(8, ch),
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.GELU(), nn.GroupNorm(8, ch * 2),
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1), nn.GELU(), nn.GroupNorm(16, ch * 4),
            nn.Conv2d(ch * 4, ch * 4, 3, 1, 1), nn.GELU(), nn.GroupNorm(16, ch * 4),
            nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(ch * 4, 256)
        self.clf = nn.Linear(256, nc)

    def emb(self, x):
        return F.normalize(self.head(self.body(x).flatten(1)), dim=-1)


m = Enc(ch=96).to(DEV)
sd = torch.load('assets/style_enc_latent.pt', map_location=DEV, weights_only=True)
m.load_state_dict(sd)
m.eval()
print('[2] 2.9M 编码器已加载, 推理全量 ...', flush=True)
feats = np.zeros((len(IDX), 256), np.float32)
with torch.no_grad():
    for s in range(0, len(IDX), 2048):
        ii = IDX[s:s + 2048]
        x = torch.from_numpy(arr[ii].astype(np.float32)).to(DEV)
        feats[s:s + 2048] = m.emb(x).cpu().numpy()
print('[2] feats %s' % (feats.shape,), flush=True)

pairs = sorted(set(PAIR.tolist()))
print('[3] pair 数 %d' % len(pairs), flush=True)
cent = np.zeros((len(pairs), 256), np.float32)
p2i = {p: j for j, p in enumerate(pairs)}
cnt = np.zeros(len(pairs))
for j, p in enumerate(PAIR):
    cent[p2i[p]] += feats[j]
    cnt[p2i[p]] += 1
cent = cent / np.maximum(cnt, 1)[:, None]
cent = cent / (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9)

np.save('assets/rank_feat_latent_50k.npy', feats)
np.save('assets/rank_cent87.npy', cent)
np.savez('assets/rank_feat_meta.npz', ids=IDX, callig=CAL, ch=CH, pair=PAIR,
         pair_keys=np.array(pairs))
print('[done] rank_feat_latent_50k.npy / rank_cent87.npy / rank_feat_meta.npz', flush=True)

# 自检: 质心间的余弦相似度分布
S = cent @ cent.T
iu = np.triu_indices(len(cent), 1)
print('[check] 质心对角外 cos: mean %.3f p95 %.3f max %.3f'
      % (S[iu].mean(), np.percentile(S[iu], 95), S[iu].max()), flush=True)