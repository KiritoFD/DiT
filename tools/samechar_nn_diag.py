#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同字最近邻诊断 v2：补三个关键对照。

v1 的问题: gap = max - mean 天然 >0, 不能证明"最近邻有判别力"。
v2 补:
  ① 错字对照: 拿 gen_i 去比**别的字**的候选 -> 得到"无信息"时的 gap/nn 参考值
  ② 配对检验: own_gt vs mean_cand 的逐列差 (同一个字, 目标 vs 随机同字样本)
  ③ 只用 229 条两集共有列 (旧集/修复集的行序一致, 只有 20 条 character 不同)
"""
import csv, glob, os, re, sys
import concurrent.futures as cf
import numpy as np
from PIL import Image

os.chdir('/root/Workspace/xy/DiT'); sys.path.insert(0, '.')
from src.eval.metrics import ssim as _ssim

tr = list(csv.DictReader(open('assets/train_50k_v2_fixed.csv', encoding='utf-8')))
by_char = {}
for r in tr:
    by_char.setdefault(str(r.get('character', '')), []).append(r)
MAXC, NW = 12, 16

_a = list(csv.DictReader(open('assets/eval_v13_strict.csv', encoding='utf-8')))
_b = list(csv.DictReader(open('assets/eval_v13_strict_fixed.csv', encoding='utf-8')))
CHANGED = {i for i, (x, y) in enumerate(zip(_a, _b)) if x['character'] != y['character']}


def img(p):
    with Image.open(p) as f:
        return np.asarray(f.convert('RGB'), dtype=np.float32) / 255.0


def cands_of(ch):
    cs = by_char.get(str(ch), [])
    if len(cs) > MAXC:
        stp = len(cs) / MAXC
        cs = [cs[int(k * stp)] for k in range(MAXC)]
    return [c for c in cs if os.path.exists(c.get('image_path', ''))]


def work(job):
    i, gp, gtp, ch, ch_other, cal = job
    if not (os.path.exists(gp) and os.path.exists(gtp)):
        return None
    g = img(gp)
    own = _ssim(g, img(gtp))
    cs = cands_of(ch)
    ss = [(_ssim(g, img(c['image_path'])), c.get('calligrapher')) for c in cs]
    cs_o = cands_of(ch_other)
    so = [_ssim(g, img(c['image_path'])) for c in cs_o]
    ncal = sum(1 for _, c2 in ss if c2 == cal)
    hit = (max([s for s, c2 in ss if c2 == cal] or [-9]) >= max([s for s, _ in ss] or [-9]) - 1e-12)
    return dict(own=own,
                nn=max([s for s, _ in ss]) if ss else np.nan,
                mean_cand=float(np.mean([s for s, _ in ss])) if ss else np.nan,
                nn_other=max(so) if so else np.nan,
                mean_other=float(np.mean(so)) if so else np.nan,
                ncal=ncal, n=len(ss), hit=bool(hit))


RUNS = [
    ('E0 (100k, 旧数据)', 'assets/results/v17_s2_s2z_baseline', 'assets/eval_v13_strict.csv'),
    ('E1 (40k)', 'assets/results/v17_s2z_ada1_ln', 'assets/eval_v13_strict.csv'),
    ('gate (25k)', 'assets/results/v17_glyph_gate_f015', 'assets/eval_v13_strict.csv'),
    ('midskel20 (40k)', 'assets/results/v17_s2_s2z_midskel20', 'assets/eval_v13_strict.csv'),
    ('inj3 (25k, 旧数据)', 'assets/results/v17_inj3_100k', 'assets/eval_v13_strict.csv'),
    ('inj3-fixed (55k)', 'assets/results/v17_inj3_fixed_100k', 'assets/eval_v13_strict_fixed.csv'),
    ('inj3-aug (100k)', 'assets/results/v17_inj3_fixed_aug_100k', 'assets/eval_v13_strict_fixed.csv'),
    ('gq (70k)', 'assets/results/v17_gq_100k', 'assets/eval_v13_strict_fixed.csv'),
]
print(f'{"run":>20} | {"step":>6} {"n":>4} | {"own_gt":>7} {"nn":>7} {"mean_c":>7} '
      f'{"gap":>7} | {"mean_错字":>8} {"gap错字":>7} | {"own-mean_c":>10} {"t":>6} | '
      f'{"同书家":>7} {"基线":>6} {"富集":>5}')
print('-' * 116)
for nm, rd, evp in RUNS:
    if not os.path.isdir(rd):
        print(f'{nm:>20} | 无目录'); continue
    sd = os.path.join(rd, 'eval_samples_ctrl')
    st = None
    for d in sorted(glob.glob(os.path.join(sd, 'step*'))):
        m = re.search(r'step(\d+)', os.path.basename(d))
        sub = os.path.join(d, 'strict')
        if m and os.path.isdir(sub) and os.path.exists(os.path.join(sub, 'g0.png')):
            st = (int(m.group(1)), sub)
    if st is None:
        print(f'{nm:>20} | 无逐样本图'); continue
    step, sub = st
    ev = list(csv.DictReader(open(evp, encoding='utf-8')))
    jobs = []
    for i, r in enumerate(ev):
        if i in CHANGED:
            continue
        jobs.append((i, os.path.join(sub, f'g{i}.png'), os.path.join(sub, f'gt{i}.png'),
                     r.get('character'), ev[(i + 37) % len(ev)].get('character'),
                     r.get('calligrapher')))
    with cf.ThreadPoolExecutor(max_workers=NW) as ex:
        rs = [x for x in ex.map(work, jobs) if x]
    own = np.array([x['own'] for x in rs]); nn = np.array([x['nn'] for x in rs])
    mc = np.array([x['mean_cand'] for x in rs]); nn2 = np.array([x['nn_other'] for x in rs])
    mo = np.array([x['mean_other'] for x in rs])
    # ★ 有些列训练集里没有同字样本 -> nn/mc 为 nan。必须按列剔除，
    #   否则 nn.mean() 会被单个 nan 污染成 nan（v1 就是这么错的）。
    ok = ~np.isnan(nn) & ~np.isnan(mc) & ~np.isnan(own)
    own, nn, mc, nn2, mo = own[ok], nn[ok], mc[ok], nn2[ok], mo[ok]
    d = own - mc
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float('nan')
    av = np.array([x['ncal'] > 0 for x in rs])[ok]
    bs = np.array([x['ncal'] / max(x['n'], 1) for x in rs])[ok]
    hit = np.array([x['hit'] for x in rs])[ok]
    print(f'{nm:>20} | {step:>6} {len(rs):>4} | {own.mean():7.4f} {nn.mean():7.4f} '
          f'{mc.mean():7.4f} {nn.mean()-mc.mean():+7.4f} | {np.nanmean(mo):8.4f} '
          f'{np.nanmean(nn2)-np.nanmean(mo):+7.4f} | {d.mean():+10.4f} '
          f'{d.mean()/se if se and se==se else 0:+6.2f} | '
          f'{hit[av].mean():6.1%} {bs[av].mean():5.1%} '
          f'{hit[av].mean()/max(bs[av].mean(),1e-9):4.2f}x', flush=True)

print()
print('读法:')
print('  gap      = nn - mean_cand  (同字候选里挑最大)  -> 天然 >0')
print('  gap错字  = 拿 gen 去比**别的字**的候选        -> **无信息参考值**; 两者接近 = 最近邻没判别力')
print('  own-mean_c 配对 t: >2 才说明"比随机同字训练样本更接近自己的目标"')
print('  同书家/基线: 富集倍数 = 书家风格信号强度 (1.0x = 完全没学到)')
