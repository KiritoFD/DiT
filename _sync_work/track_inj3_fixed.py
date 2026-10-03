#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inj3-fixed vs E0 的跟踪表 —— 用「229 条未改标签样本」绕开口径断裂。

背景: 新 run 在修复评测集上, E0 在旧评测集上, 不能直接相减。
但那 20 条被改标签的样本**之外**的 229 条, 其 character / std_path / 骨架 latent
在两套集子里**完全相同** -> 这 229 条上的数字**直接可比**, 现在就能比, 不用等回溯评估。

另外跟踪: 那 20 条的 gap(相对 229 条) —— 它是"模型有没有在听 g"的干净指标,
随修复数据训练推进应逐渐收窄。
"""
import csv
import os
import sys

import numpy as np

OLD = 'assets/eval_v13_strict.csv'
NEW = 'assets/eval_v13_strict_fixed.csv'

E0 = 'assets/results/v17_s2_s2z_baseline/eval_stdskel_batch.csv'
I3 = 'assets/results/v17_inj3_100k/eval_stdskel_batch.csv'
FX = 'assets/results/v17_inj3_fixed_100k/eval_stdskel_batch.csv'


def fixed_ids():
    a = list(csv.DictReader(open(OLD, encoding='utf-8')))
    b = list(csv.DictReader(open(NEW, encoding='utf-8')))
    return {ra['image_path'].split('/')[-1]
            for ra, rb in zip(a, b) if ra['character'] != rb['character']}


def load(path):
    """-> {(step, set): [row, ...]}"""
    out = {}
    if not os.path.exists(path):
        return out
    for r in csv.DictReader(open(path, encoding='utf-8')):
        out.setdefault((r['step'], r['set']), []).append(r)
    return out


def mean(rows, key):
    v = [float(r[key]) for r in rows if r.get(key) not in (None, '')]
    return float(np.mean(v)) if v else float('nan')


FXID = fixed_ids()
print(f'被改标签的样本: {len(FXID)} 条 -> 其余 {249 - len(FXID)} 条用于直接对比\n')

e0 = load(E0)
i3 = load(I3)
fx = load(FX)

print('=== 口径 A: 只用「229 条未改标签样本」—— 与 E0 直接可比 ===')
print(f'{"step":>7} | {"E0 ssim":>8} {"E0 frag":>8} | {"inj3-fix":>8} {"frag":>8} {"Δssim":>8}'
      f' | {"inj3(旧)":>9}')
print('-' * 72)
for step in sorted({s for s, t in fx if t == 'strict'}, key=int):
    ref = [r for r in e0.get((step, 'strict'), [])
           if r['img_id'].split('/')[-1] not in FXID]
    new = [r for r in fx.get((step, 'strict'), [])
           if r['img_id'].split('/')[-1] not in FXID]
    old = [r for r in i3.get((step, 'strict'), [])
           if r['img_id'].split('/')[-1] not in FXID]
    if not new:
        continue
    es, ef = mean(ref, 'ssim'), mean(ref, 'frag_ratio')
    ns, nf = mean(new, 'ssim'), mean(new, 'frag_ratio')
    os_ = mean(old, 'ssim')
    d = f'{ns - es:+.4f}' if ref else '   n/a'
    o = f'{os_:.4f}' if old else '   n/a'
    print(f'{step:>7} | {es:8.4f} {ef:8.3f} | {ns:8.4f} {nf:8.3f} {d:>8} | {o:>9}')

print()
print('=== 口径 B: 那 20 条改标签样本 —— 只在修复集内看趋势 ===')
print('   （旧集上它们分数虚高, 不可比; 修复集上才是真实水平）')
print(f'{"step":>7} | {"20条 ssim":>10} {"229条 ssim":>11} {"gap":>9} | {"20条 frag":>10}')
print('-' * 60)
for step in sorted({s for s, t in fx if t == 'strict'}, key=int):
    rows = fx.get((step, 'strict'), [])
    bad = [r for r in rows if r['img_id'].split('/')[-1] in FXID]
    good = [r for r in rows if r['img_id'].split('/')[-1] not in FXID]
    if not bad:
        continue
    bs, gs = mean(bad, 'ssim'), mean(good, 'ssim')
    print(f'{step:>7} | {bs:10.4f} {gs:11.4f} {bs - gs:+9.4f} | {mean(bad, "frag_ratio"):10.3f}')

print()
print('=== 参考: 旧集上这 20 条 vs 229 条 (E0, 说明修复前是"虚高") ===')
for step in ['40000', '100000']:
    rows = e0.get((step, 'strict'), [])
    bad = [r for r in rows if r['img_id'].split('/')[-1] in FXID]
    good = [r for r in rows if r['img_id'].split('/')[-1] not in FXID]
    if bad:
        print(f'  E0 step{step}: 20条 {mean(bad, "ssim"):.4f} | 229条 {mean(good, "ssim"):.4f} '
              f'| gap {mean(bad, "ssim") - mean(good, "ssim"):+.4f}')
