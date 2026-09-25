#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""风格编码器 f：**作用在 VAE latent 上**的书家编码器（第 0 步，必须先验证它）。

为什么必须是 latent:
  扩散训练要监督的是 x0_pred，它是 (4,32,32) 的 VAE latent。若 f 吃像素，
  每步都要 VAE decode（额外显存与耗时）。所以 f 必须直接在 latent 上工作。

本脚本回答三件事（按重要性）:
  ① **按字划分**的留出准确率 —— 关键。测试字符训练时完全没见过，模型只能靠"风格"答对。
     随机划分的准确率再高都可能是在背 (字,书家) 配对，不足以说明学到了风格。
  ② 同字池检索的富集倍数（只能在"字见过"的那部分做，因为按字划分的测试字没有同字池）
  ③ 对比学习（content-matched InfoNCE）相对纯 CE 的增量

⚠ 历史踩坑，都记在这里避免再犯:
  · 第一版用 train_m(只排除随机10%) 训练 -> 按字测试集泄漏 -> 出现"按字 92.7% > 随机 30.4%"
    这种不可能的结果。必须用 tr_m_char。
  · 第二版把 NCE 写成 **batch 内**配对。5750 字 / batch 512 -> 同字几乎不可能同批出现
    -> 负样本恒为空 -> NCE 是 no-op。必须按索引查表取配对。
  · InfoNCE 必须写全: logsumexp(pos, negs) − pos/τ；漏了 − pos/τ 就变成"把一切都推开"。

用法:
  python tools/style_encoder_latent.py [--steps 8000] [--batch 512] [--use-nce 1]
产出: assets/style_enc_latent.pt
"""
import argparse
import csv
import glob
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

ap = argparse.ArgumentParser()
ap.add_argument('--steps', type=int, default=8000)
ap.add_argument('--batch', type=int, default=512)
ap.add_argument('--kneg', type=int, default=8)
ap.add_argument('--use-nce', type=int, default=1)
ap.add_argument('--nce-w', type=float, default=1.0)
ap.add_argument('--lr', type=float, default=8e-4)
ap.add_argument('--ch', type=int, default=96, help='首层通道数(容量)')
ap.add_argument('--drop', type=float, default=0.0, help='embedding dropout')
ap.add_argument('--noise', type=float, default=0.0, help='latent 输入加噪(数据增强)')
ap.add_argument('--wd', type=float, default=0.02)
# ⚠ 对比实验必须分开存: 第二版曾把更好的大模型权重覆盖掉。
ap.add_argument('--out', type=str, default='assets/style_enc_latent.pt')
ap.add_argument('--eval-every', type=int, default=2000)
a = ap.parse_args()

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
np.random.seed(0)


def rid(p):
    m = re.search(r'(\d+)\.png$', str(p))
    return int(m.group(1)) if m else None


print('[1] 载入 latent ...', flush=True)
NMAX = 52457
arr = np.zeros((NMAX, 4, 32, 32), np.float16)
has = np.zeros(NMAX, bool)
for f in sorted(glob.glob('data/50k/shards_img/*.npz')):
    z = np.load(f)
    L, I = z['latents'], z['img_ids']
    arr[I] = L
    has[I] = True
print('[1] latent %s, 有效 %d' % (arr.shape, int(has.sum())), flush=True)

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
IDX = np.array(IDX)
CAL = np.array(CAL)
CH = np.array(CH)
print('[1] 样本 %d, 书家 %d, 字 %d' % (len(IDX), len(CALLS), len(set(CH))), flush=True)

# ---- 划分 ----
rng = np.random.RandomState(0)
chars = np.array(sorted(set(CH)))
rng.shuffle(chars)
test_chars = set(chars[:int(len(chars) * 0.15)].tolist())
m_char = np.array([c in test_chars for c in CH])            # 按字划分 test
ridx = rng.permutation(len(IDX))
m_rand = np.zeros(len(IDX), bool)
m_rand[ridx[:int(len(IDX) * 0.1)]] = True                   # 随机划分 test
# ⚠ 训练集必须**同时**排除两个测试集。
#   第三版只排除了 m_char -> m_rand 的样本仍在训练里 -> "样本新/字见过" 99.77% 其实是训练准确率。
tr_final = (~m_char) & (~m_rand)
print('[2] 按字 test %d (字 %d 个) / 随机 test %d / 训练 %d'
      % (m_char.sum(), len(test_chars), m_rand.sum(), tr_final.sum()), flush=True)


class Enc(nn.Module):
    """(4,32,32) -> 256 维归一化 embedding + 45 类 logits"""
    def __init__(self, nc=len(CALLS), d=256, ch=96, drop=0.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(4, ch, 3, 1, 1), nn.GELU(), nn.GroupNorm(8, ch),
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.GELU(), nn.GroupNorm(8, ch * 2),
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1), nn.GELU(), nn.GroupNorm(16, ch * 4),
            nn.Conv2d(ch * 4, ch * 4, 3, 1, 1), nn.GELU(), nn.GroupNorm(16, ch * 4),
            nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(ch * 4, d)
        self.drop = nn.Dropout(drop)
        self.clf = nn.Linear(d, nc)

    def emb(self, x):
        return F.normalize(self.head(self.drop(self.body(x).flatten(1))), dim=-1)

    def forward(self, x):
        z = self.emb(x)
        return z, self.clf(z)


model = Enc(ch=a.ch, drop=a.drop).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
print('[2] 参数 %s' % format(sum(p.numel() for p in model.parameters()), ','), flush=True)

TRX = torch.from_numpy(arr[IDX[tr_final]].astype(np.float32)).to(DEV)
TRC = CAL[tr_final]
TRH = CH[tr_final]
# ★ 按索引查表的配对。不能用 batch 内配对: 5750 字 / batch 512 -> 同字几乎不同批出现
by_cal, by_char = {}, {}
for j in range(len(TRC)):
    by_cal.setdefault(int(TRC[j]), []).append(j)
    by_char.setdefault(TRH[j], []).append(j)
POS = []
for j in range(len(TRC)):
    v = [x for x in by_cal.get(int(TRC[j]), []) if TRH[x] != TRH[j]]
    POS.append(np.array(v) if v else np.array([j]))
NEG = []
for j in range(len(TRC)):
    v = [x for x in by_char.get(TRH[j], []) if int(TRC[x]) != int(TRC[j])]
    NEG.append(np.array(v) if v else np.array([j]))
print('[2] 配对表: 正(同书家异字) 均 %.0f / 负(同字异书家) 均 %.1f'
      % (np.mean([len(p) for p in POS]), np.mean([len(n) for n in NEG])), flush=True)


@torch.no_grad()
def emb_of(idx):
    out = []
    for s in range(0, len(idx), 2048):
        x = torch.from_numpy(arr[idx[s:s + 2048]].astype(np.float32)).to(DEV)
        out.append(model.emb(x).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def acc(mask, tag):
    if mask.sum() == 0:
        print('  %-22s (空)' % tag, flush=True)
        return np.nan
    Z = emb_of(IDX[mask])
    lg = model.clf(torch.from_numpy(Z).to(DEV)).cpu().numpy()
    v = (lg.argmax(1) == CAL[mask]).mean()
    print('  %-22s 准确率 %6.2f%%  (n=%d, 随机 %.1f%%)'
          % (tag, v * 100, int(mask.sum()), 100 / len(CALLS)), flush=True)
    return v


def retrieval(mask, tag):
    """同字池检索富集。池只能来自按字划分的训练集 -> mask 必须是"字见过"的那部分。"""
    pool_all = {}
    for j in range(len(TRX)):
        pool_all.setdefault(TRH[j], []).append(int(IDX[tr_final][j]))
    pool_all = {k: np.array(v[:12]) for k, v in pool_all.items() if len(v) >= 2}
    CAL_OF = {int(IDX[tr_final][k]): int(TRC[k]) for k in range(len(TRX))}
    Zp = {k: emb_of(c) for k, c in pool_all.items()}
    qs, qc, qh = IDX[mask], CAL[mask], CH[mask]
    Zq = emb_of(qs)
    hit = tot = 0
    base = 0.0
    for k in range(len(qs)):
        ch_ = qh[k]
        if ch_ not in Zp:
            continue
        Zc = Zp[ch_]
        cand = pool_all[ch_]
        best = int(np.argmax(Zc @ Zq[k]))
        hit += int(CAL_OF[int(cand[best])] == int(qc[k]))
        base += sum(1 for c in cand if CAL_OF[int(c)] == int(qc[k])) / len(cand)
        tot += 1
    if not tot:
        print('  %-22s (无可用池)' % tag, flush=True)
        return np.nan
    h, b = hit / tot, base / tot
    print('  %-22s 富集 %.2fx (命中 %.1f%%, 基线 %.1f%%, n=%d)' % (tag, h / b, h * 100, b * 100, tot),
          flush=True)
    return h / b


print('\n[3] 训练 %d 步 (batch %d, K负 %d, NCE %d)' % (a.steps, a.batch, a.kneg, a.use_nce), flush=True)
t0 = time.time()
for step in range(a.steps):
    bi = np.random.randint(0, len(TRX), a.batch)
    x = TRX[bi]
    if a.noise > 0:      # latent 输入加噪 = 最便宜的数据增强，专为对抗过拟合
        x = x + a.noise * torch.randn_like(x)
    z, lg = model(x)
    loss = F.cross_entropy(lg, torch.from_numpy(TRC[bi]).to(DEV))
    if a.use_nce:
        pj = np.array([POS[j][np.random.randint(len(POS[j]))] for j in bi])
        nj = np.stack([NEG[j][np.random.choice(len(NEG[j]), a.kneg,
                                               replace=len(NEG[j]) < a.kneg)] for j in bi])
        zp = model.emb(TRX[pj])
        zn = model.emb(TRX[nj.reshape(-1)]).view(len(bi), a.kneg, -1)
        s_pos = (z * zp).sum(-1) / 0.1
        s_neg = torch.bmm(zn, z.unsqueeze(-1)).squeeze(-1) / 0.1
        # ★ InfoNCE = logsumexp(pos, negs) − pos/τ（漏了 − pos/τ 就变成"把一切都推开"）
        loss = loss + a.nce_w * (torch.logsumexp(
            torch.cat([s_pos.unsqueeze(-1), s_neg], 1), 1) - s_pos).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    sched.step()
    if (step + 1) % 500 == 0:
        print('    step %5d  loss %.4f  %.0fs' % (step + 1, float(loss), time.time() - t0), flush=True)
    if (step + 1) % a.eval_every == 0 and (step + 1) < a.steps:
        model.eval()
        print('    [eval @%d]' % (step + 1), flush=True)
        acc(m_rand & (~m_char), '  样本新/字见过')
        acc(m_char, '  样本新/字没见过')
        model.train()

model.eval()
print('\n[4] 最终留出准确率（随机基线 %.1f%%）' % (100 / len(CALLS)))
acc(m_rand & (~m_char), '样本没见过/字见过')
acc(m_char, '**样本没见过/字没见过**')
print('  -> 判读: "字没见过"必须显著高于 %.1f%% 才算学到的是风格而非字形' % (100 / len(CALLS)), flush=True)

print('\n[5] 同字池检索富集')
retrieval(m_rand & (~m_char), '样本没见过/字见过')

torch.save(model.state_dict(), 'assets/style_enc_latent.pt')
print('\n  编码器已存 assets/style_enc_latent.pt  (总耗时 %.0fs)' % (time.time() - t0), flush=True)
