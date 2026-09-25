#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单独训练 DeformSkel —— 不碰扩散模型，几分钟出结论。

## 为什么能单独训
形变模块是一个小网络: (g_std, 书家风格) -> g'。
目标**现成且稠密**: g' 应逼近"该书家写的那个字"的 GT 骨架（`shards_aux_skel3`，
实测逐样本比值 0.996，宽度 ~3px 最接近 std 的 ~4px）。就是一个监督回归任务。

## 三个判据
  ① **闭合率** = 1 − MSE(g',g_gt) / MSE(g_std,g_gt)
     >60% 说明形变确实把 g_std 拉近该书家的写法
  ② **style-follow（关键）**：同一个字，对每个书家 k 生成 g'(c,k)，
     再看 g'(c,k) 是否比 g'(c,k') 更接近 **k 自己的** g_gt(c,k)。
     这一项直接回答"同一个 skel 是否对不同书家产出对应 skel"。
  ③ **offset 幅值**（含风格底图那一份）: ≈0 = 退化成恒等

## ★ v2 的两处修正（v1 的问题）
  · batch 太小(512) -> 改成吃满显存
  · **同字分组采样**：每个 batch 由若干"字"组成，每个字取多个书家 ->
    同一个 g_std 在同批里对应多个不同的 g_gt，**逼模型必须用风格去区分**。
    v1 用随机采样时同批几乎不会出现同字，风格因此被忽略（correct≈shuffled）。

## 用法
  python tools/train_deform_standalone.py --steps 6000 --batch 4096 --group 32
产出 assets/deform_skel_standalone.pt
"""
import argparse
import collections
import csv
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.chdir('/root/Workspace/xy/DiT')
sys.path.insert(0, '.')
from src.model.deform_skel import DeformSkel  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--steps', type=int, default=6000)
ap.add_argument('--batch', type=int, default=4096)
ap.add_argument('--group', type=int, default=32, help='每个 batch 采多少个"字"')
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--std-dir', default='data/50k/shards_std_fixed')
ap.add_argument('--gt-dir', default='data/50k/shards_aux_skel3')
ap.add_argument('--out', default='assets/deform_skel_standalone.pt')
ap.add_argument('--eval-every', type=int, default=1000)
ap.add_argument('--residual', type=int, default=0, help='1=形变+加性残差')
ap.add_argument('--style-dim', type=int, default=128, dest='style_dim')
ap.add_argument('--style-emb', default='assets/callig_emb_pretrained_50k.pt',
                dest='style_emb',
                help='用**模型同一张**预训练书家表(冻结); 否则离线训的风格输入与'
                     '线上 _e_callig() 对不上, 训好的头接进去会失效')
ap.add_argument('--width', type=int, default=64)
ap.add_argument('--max-off', type=float, default=3.0, dest='max_off')
ap.add_argument('--res-cap', type=float, default=1.0, dest='res_cap')
ap.add_argument('--follow-n', type=int, default=400, help='style-follow 测多少个字')
a = ap.parse_args()

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
np.random.seed(0)
NCAL = 45


def load_bank(d):
    m = {}
    for f in sorted(glob.glob(os.path.join(d, 'shard_*.npz'))):
        z = np.load(f)
        L, I = z['latents'], z['img_ids']
        for j, i in enumerate(I):
            m[int(i)] = L[j].astype(np.float32)
        z.close()
    return m


print('[1] 载入 g_std 与 g_gt ...', flush=True)
A = load_bank(a.std_dir)
B = load_bank(a.gt_dir)
print(f'    g_std {len(A)} / g_gt {len(B)}', flush=True)

C2I = {}
rows = list(csv.DictReader(open('assets/train_50k_v2_fixed.csv', encoding='utf-8')))
for r in rows:
    C2I.setdefault(str(r.get('calligrapher', '')), len(C2I))
rec = {}
for r in rows:
    try:
        i = int(os.path.basename(r.get('image_path', '')).split('.')[0])
    except Exception:
        continue
    if i in A and i in B:
        rec[i] = (C2I.get(str(r.get('calligrapher', '')), 0), str(r.get('character', '')))
ids = sorted(rec)
print(f'    有效 {len(ids)} 条, 书家 {len(C2I)} 个', flush=True)

G = torch.from_numpy(np.stack([A[i] for i in ids])).to(DEV)
T = torch.from_numpy(np.stack([B[i] for i in ids])).to(DEV)
Y = torch.tensor([rec[i][0] for i in ids], device=DEV)
CH = [rec[i][1] for i in ids]
N = len(ids)

# 字 -> 该字的样本下标（用于分组采样）
by_char = collections.defaultdict(list)
for k, c in enumerate(CH):
    by_char[c].append(k)
# 只保留"至少 2 个不同书家"的字（style-follow 与分组采样都需要）
groups = [v for v in by_char.values()
          if len({int(Y[i]) for i in v}) >= 2]
print(f'[1] 可用于分组/跟随测试的字 {len(groups)} 个', flush=True)

base_mse = float((G - T).pow(2).mean())
print(f'\n[2] 基线 MSE(g_std, g_gt) = {base_mse:.5f}   <- 形变要打败的就是它', flush=True)

model = DeformSkel(cond_dim=a.style_dim, ch=4, grid=32, residual=a.residual,
                   width=a.width, max_off=a.max_off, res_cap=a.res_cap).to(DEV)
# ★ 风格源必须与模型 _e_callig() 一致: 那就是 callig_emb_pretrained_50k.pt 的行。
#   用自己学的 embedding 会导致"离线训好的头接进模型后风格输入分布不一致"而失效。
_emb = torch.load(a.style_emb, map_location='cpu', weights_only=False)
_tab = _emb['embedding'] if isinstance(_emb, dict) else _emb
_tab = _tab.float()
assert _tab.shape[0] == NCAL, _tab.shape
print(f'[2] 用预训练书家表 {a.style_emb} {tuple(_tab.shape)} (冻结)', flush=True)
style = nn.Embedding.from_pretrained(_tab, freeze=True).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
print(f'[2] 形变模块参数 {sum(p.numel() for p in model.parameters()):,}  '
      f'(含风格底图 {sum(p.numel() for p in model.style_off.parameters()):,})', flush=True)

n_val = max(1, int(N * 0.03))
perm = np.random.permutation(N)
val_i = torch.tensor(perm[:n_val], device=DEV)
tr_i = torch.tensor(perm[n_val:], device=DEV)

# style-follow 评测集：每个字取最多 6 个不同书家的样本
follow = []
for v in groups:
    byk = {}
    for i in v:
        byk.setdefault(int(Y[i]), i)
    if len(byk) >= 2:
        follow.append(list(byk.values())[:6])
    if len(follow) >= a.follow_n:
        break
print(f'[2] style-follow 用 {len(follow)} 个字', flush=True)


@torch.no_grad()
def style_follow():
    """同一个字、不同书家: g'(c,k) 是否比 g'(c,k') 更接近 k 自己的 g_gt(c,k)。"""
    model.eval()
    good = tot = 0
    same_g = []
    for v in follow:
        gs = G[v]                                  # 同字的多个样本（g_std 基本同图）
        same_g.append(float((gs - gs[0]).abs().mean()))
        for i in v:
            others = [j for j in v if int(Y[j]) != int(Y[i])]
            if not others:
                continue
            gp_self = model(G[i:i + 1], style(Y[i:i + 1]))[0]
            gp_oth = model(G[i:i + 1], style(Y[others[0]:others[0] + 1]))[0]
            d_self = float((gp_self - T[i]).pow(2).mean())
            d_oth = float((gp_oth - T[i]).pow(2).mean())
            good += int(d_self < d_oth)
            tot += 1
    model.train()
    return (good / tot if tot else float('nan')), (np.mean(same_g) if same_g else float('nan'))


@torch.no_grad()
def evaluate(tag):
    model.eval()
    vi = val_i
    g2 = model(G[vi], style(Y[vi]))
    mse = float((g2 - T[vi]).pow(2).mean())
    ysh = (Y[vi] + 3) % NCAL
    g2s = model(G[vi], style(ysh))
    off = model.offset_stats() or {}
    fr, sg = style_follow()
    print('  %-10s MSE=%.5f (基线 %.5f, 闭合 %5.1f%%) | correct %.5f / shuffled %.5f '
          '| 输出变化 %.4f | off %.4f (风格底图 %.4f) | **style-follow %.1f%%**'
          % (tag, mse, base_mse, 100 * (1 - mse / base_mse),
             float((g2 - T[vi]).pow(2).mean()), float((g2s - T[vi]).pow(2).mean()),
             float((g2 - g2s).abs().mean()), off.get('mean_abs', 0),
             off.get('style_part', 0), 100 * fr), flush=True)
    model.train()
    return mse


print('\n[3] 训练 %d 步 (batch %d, 每批 %d 个字 分组采样)' % (a.steps, a.batch, a.group),
      flush=True)
evaluate('step0')
t0 = time.time()
per = max(1, a.batch // a.group)
for step in range(a.steps):
    sel = []
    for _ in range(a.group):
        v = groups[np.random.randint(len(groups))]
        if len(v) <= per:
            sel.extend(v)
        else:
            sel.extend(np.random.choice(v, per, replace=False).tolist())
    bi = torch.tensor(sel, device=DEV)
    g2 = model(G[bi], style(Y[bi]))
    loss = (g2 - T[bi]).pow(2).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    sched.step()
    if (step + 1) % 500 == 0:
        print('    step %5d  loss %.5f  %.0fs' % (step + 1, float(loss), time.time() - t0),
              flush=True)
    if (step + 1) % a.eval_every == 0:
        evaluate('step%d' % (step + 1))

print('\n[4] 最终', flush=True)
evaluate('final')
print()
print('=== 判读 ===')
print('  闭合率: >60% = 形变确实把 g_std 拉近该书家的写法')
print('  style-follow: 50% = 与随机无异(风格没驱动); >75% = 同一个 skel 对不同书家产出了对应 skel')
print('  off 的"风格底图"那一项: >0 说明风格专属的全局形变在起作用')
torch.save(dict(deform=model.state_dict(), style_emb=a.style_emb), a.out)
print('  已存', a.out, flush=True)
