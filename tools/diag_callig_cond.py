#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""书家条件诊断：correct / shuffled / null —— 模型到底有没有在用「书家」这个条件？

做法: **同一份噪声、同一个 seed、同一个 g**，只改 y_callig，生成三组：
  correct  = 该样本真实书家
  shuffled = 换成**另一个**书家（每个样本轮转不同偏移，避免只测一个 k'）
  null     = 书家条件置为 CFG 的无条件行（表外那一行）

四个判据（说服力从弱到强）：
  ① ssim(correct, GT) vs ssim(shuffled, GT)   —— 对的条件是否真的更贴近 GT
  ② ssim(correct, shuffled)                    —— 条件**有没有改变输出**（≈1.0 = 完全没用）
  ③ 同字池检索命中率（correct / shuffled）      —— 输出跟的是条件书家还是字形先验
  ④ **follow 率**：换成 k' 之后，输出的最近邻变成 k' 的比例 —— 最强判据

用法: python tools/diag_callig_cond.py --ckpt <pt> [--n 120]
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

os.chdir('/root/Workspace/xy/DiT')
sys.path.insert(0, '.')

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--n', type=int, default=120)
ap.add_argument('--cfg', type=float, default=0.7)
ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--device', default='cuda')
ap.add_argument('--nn-pool', type=int, default=16, help='同字池候选上限')
a = ap.parse_args()

from src.eval.in_mem_eval import _get_cache                     # noqa: E402
from src.eval.inference import (build_diffusion, load_eval_vae,  # noqa: E402
                                sample_latents, _ssim)
from src.eval.model_io import load_model_from_ckpt               # noqa: E402

DEV = torch.device(a.device)
CSVP = 'assets/eval_v13_strict.csv'

t0 = time.time()
model, args = load_model_from_ckpt(a.ckpt, device=DEV, use_ema=True, verbose=False)
model.eval()
print('[1] 模型载入 %.0fs' % (time.time() - t0), flush=True)

cache = _get_cache(CSVP, a.n, None, getattr(args, 'eval_skel_latent_shards_dir', None), args)
n = cache['n']
ev = list(csv.DictReader(open(CSVP, encoding='utf-8')))[:n]
print('[2] 评测 %d 条' % n, flush=True)

NCAL = int(getattr(args, 'num_calligraphers', 45))
conds = cache['conds']
cal = [int(c[0]) for c in conds]
glyph = [int(c[1]) for c in conds]

# index -> 书家名（用于判断"输出的最近邻是不是条件指定的书家"）
name_of = {}
for i in range(n):
    name_of[cal[i]] = ev[i].get('calligrapher', '')
print('[2] 覆盖书家 %d 个' % len(name_of), flush=True)

rng = np.random.RandomState(0)
shift = rng.randint(1, NCAL, size=n)
cal_sh = [(cal[i] + int(shift[i])) % NCAL for i in range(n)]
conds_sh = [(cal_sh[i], glyph[i]) for i in range(n)]
conds_nu = [(NCAL, glyph[i]) for i in range(n)]        # 表外 = CFG 无条件

diff = build_diffusion(int(getattr(args, 'eval_steps', 50)), 'flow',
                       flow_kwargs={'sampler': getattr(args, 'flow_sampler', 'heun')})
noise, skel = cache['noise'], cache.get('skels_latent')


def gen(cd, tag):
    t = time.time()
    L = sample_latents(model, diff, noise, cd, a.cfg, a.batch, DEV, skel=skel, seed=0)
    print('    %-9s 采样 %.0fs' % (tag, time.time() - t), flush=True)
    return L


L = {'correct': gen(conds, 'correct'),
     'shuffled': gen(conds_sh, 'shuffled'),
     'null': gen(conds_nu, 'null')}

vae = load_eval_vae(DEV)
scaling = float(getattr(vae.config, 'scaling_factor', 0.18215))


@torch.no_grad()
def decode(Lt):
    out = []
    for i in range(0, len(Lt), 16):
        z = Lt[i:i + 16].to(DEV) / scaling
        im = vae.decode(z).sample
        out.append(((im.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).float().cpu().numpy())
    return np.concatenate(out)


P = {k: decode(v) for k, v in L.items()}
print('[3] decode 完成', flush=True)

GT = []
for i in range(n):
    p = ev[i].get('image_path', '')
    if not os.path.isabs(p):
        p = os.path.join(os.getcwd(), p)
    with Image.open(p) as f:
        GT.append(np.asarray(f.convert('RGB'), np.float32) / 255.0)
GT = np.stack(GT)

print('\n=== 判据 ① 对的条件是否更贴近 GT ===')
s = {k: float(np.mean([_ssim(P[k][i], GT[i]) for i in range(n)])) for k in P}
for k in ['correct', 'shuffled', 'null']:
    print('  %-9s ssim=%.4f   (Δ vs correct %+.4f)' % (k, s[k], s[k] - s['correct']))

print('\n=== 判据 ② 条件有没有改变输出 ===')
for k in ['shuffled', 'null']:
    d = float(np.mean([_ssim(P['correct'][i], P[k][i]) for i in range(n)]))
    print('  ssim(correct, %-8s) = %.4f   (≈1.0 -> 条件完全没用)' % (k, d))

print('\n=== 判据 ③④ 同字池检索 ===')
tr = list(csv.DictReader(open('assets/train_50k_v2_fixed.csv', encoding='utf-8')))
by_char = {}
for r in tr:
    by_char.setdefault(str(r.get('character', '')), []).append(r)


def nn_name(img, ch):
    cs = [c for c in by_char.get(str(ch), [])
          if c.get('image_path') and os.path.exists(c['image_path'])]
    if not cs:
        return None
    best, bs = None, -9.0
    for c in cs[:a.nn_pool]:
        with Image.open(c['image_path']) as f:
            t = np.asarray(f.convert('RGB'), np.float32) / 255.0
        sc = _ssim(img, t)
        if sc > bs:
            bs, best = sc, c.get('calligrapher')
    return best


stat = dict(tot=0, cor_hit_gt=0, sh_hit_gt=0, sh_follow=0)
for i in range(n):
    ch = ev[i].get('character', '')
    k_gt = name_of.get(cal[i])
    k_sh = name_of.get(cal_sh[i])
    if not k_gt or not k_sh:
        continue
    na, nb = nn_name(P['correct'][i], ch), nn_name(P['shuffled'][i], ch)
    if na is None or nb is None:
        continue
    stat['tot'] += 1
    stat['cor_hit_gt'] += int(na == k_gt)
    stat['sh_hit_gt'] += int(nb == k_gt)
    stat['sh_follow'] += int(nb == k_sh)

T = max(stat['tot'], 1)
print('  n=%d' % stat['tot'])
print('  correct  输出的最近邻 == **GT 书家**   : %5.1f%%' % (100 * stat['cor_hit_gt'] / T))
print('  shuffled 输出的最近邻 == **GT 书家**   : %5.1f%%   <- 若与上一行接近 => 输出没跟条件走'
      % (100 * stat['sh_hit_gt'] / T))
print('  shuffled 输出的最近邻 == **新条件书家**: %5.1f%%   <- follow 率, 越高说明条件越有效'
      % (100 * stat['sh_follow'] / T))

print('\n=== 读法 ===')
print('  ② 若 ssim(correct,shuffled) 接近 1.0  -> **书家条件对输出几乎无影响**（条件路径失效）')
print('  ① 若 shuffled 的 ssim 与 correct 接近 -> 对的条件没带来可测收益')
print('  ④ follow 率若接近随机(1/同字池候选数) -> 输出跟的是字形先验，不是书家条件')
