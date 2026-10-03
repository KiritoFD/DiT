#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨 run 的「同字最近邻」诊断。

对每个 run 的**最后一个 step**，逐列算四个量：
  own_gt    : ssim(生成, 本列自己的 GT)          <- 模型对目标的拟合
  nn        : max over 同字训练候选 的 ssim      <- "最像谁"
  mean_cand : 同字训练候选的 ssim 均值            <- 随机挑一张的期望
  gap       : nn - mean_cand                     <- 最近邻的**判别力**（gap 小 = 挑谁都是噪声）
外加: 有同书家候选时命中同书家的比例 vs 随机基线。
"""
import csv, glob, json, os, re, sys
import numpy as np
from PIL import Image

os.chdir('/root/Workspace/xy/DiT'); sys.path.insert(0, '.')
from src.eval.metrics import ssim as _ssim

TRAIN = 'assets/train_50k_v2_fixed.csv'
tr = list(csv.DictReader(open(TRAIN, encoding='utf-8')))
by_char = {}
for r in tr:
    by_char.setdefault(str(r.get('character', '')), []).append(r)
MAXC = 12


def img(p):
    with Image.open(p) as f:
        return np.asarray(f.convert('RGB'), dtype=np.float32) / 255.0


def diag_run(rd, evcsv, ncol):
    sd = os.path.join(rd, 'eval_samples_ctrl')
    steps = []
    for d in sorted(glob.glob(os.path.join(sd, 'step*'))):
        m = re.search(r'step(\d+)', os.path.basename(d))
        sub = os.path.join(d, 'strict')
        if m and os.path.isdir(sub):
            k = 0
            while os.path.exists(os.path.join(sub, f'g{k}.png')):
                k += 1
            if k:
                steps.append((int(m.group(1)), sub, k))
    if not steps:
        return None
    st, sub, n = steps[-1]
    ev = list(csv.DictReader(open(evcsv, encoding='utf-8')))[:min(ncol, n)]
    own, nn_, mn, hit, avail, base = [], [], [], [], [], []
    for i, r in enumerate(ev):
        gp = os.path.join(sub, f'g{i}.png')
        gtp = os.path.join(sub, f'gt{i}.png')
        if not (os.path.exists(gp) and os.path.exists(gtp)):
            continue
        g = img(gp)
        own.append(_ssim(g, img(gtp)))
        cs = by_char.get(str(r.get('character', '')), [])
        if len(cs) > MAXC:
            stp = len(cs) / MAXC
            cs = [cs[int(k * stp)] for k in range(MAXC)]
        ss = []
        for c in cs:
            p = c.get('image_path', '')
            if os.path.exists(p):
                t = img(p)
                if t.shape == g.shape:
                    ss.append(_ssim(g, t))
        if ss:
            nn_.append(max(ss)); mn.append(float(np.mean(ss)))
        cal = r.get('calligrapher')
        ncal = sum(1 for c in cs if c.get('calligrapher') == cal)
        avail.append(ncal > 0); base.append(ncal / max(len(cs), 1))
        if ncal > 0:
            hit.append(bool(ss) and float(max(
                [_ssim(g, img(c['image_path'])) for c in cs
                 if c.get('calligrapher') == cal and os.path.exists(c.get('image_path', ''))] or [-9])) >=
                max(ss) - 1e-12)
    if not nn_:
        return None
    av, bs = np.array(avail), np.array(base)
    return dict(step=st, n=len(own),
                own=float(np.mean(own)), nn=float(np.mean(nn_)),
                mean_cand=float(np.mean(mn)),
                gap=float(np.mean(nn_) - np.mean(mn)),
                hit=float(np.mean(hit)) if hit else float('nan'),
                base=float(bs[av].mean()) if av.any() else float('nan'),
                n_avail=int(av.sum()))


RUNS = [
    ('E0 (旧数据/无增强)', 'assets/results/v17_s2_s2z_baseline', 'assets/eval_v13_strict.csv', 249),
    ('E1 (style LN)', 'assets/results/v17_s2z_ada1_ln', 'assets/eval_v13_strict.csv', 249),
    ('gate (骨架压0.15)', 'assets/results/v17_glyph_gate_f015', 'assets/eval_v13_strict.csv', 249),
    ('midskel20', 'assets/results/v17_s2_s2z_midskel20', 'assets/eval_v13_strict.csv', 249),
    ('inj3 (旧数据)', 'assets/results/v17_inj3_100k', 'assets/eval_v13_strict.csv', 249),
    ('inj3-fixed', 'assets/results/v17_inj3_fixed_100k', 'assets/eval_v13_strict_fixed.csv', 249),
    ('inj3-aug', 'assets/results/v17_inj3_fixed_aug_100k', 'assets/eval_v13_strict_fixed.csv', 249),
    ('gq (本次)', 'assets/results/v17_gq_100k', 'assets/eval_v13_strict_fixed.csv', 249),
]
print(f'{"run":>22} | {"step":>7} {"n":>4} | {"own_gt":>7} {"nn":>7} {"mean_cand":>9} {"gap":>6} '
      f'| {"同书家命中":>10} {"基线":>6}')
print('-' * 96)
for nm, rd, ev, nc in RUNS:
    if not os.path.isdir(rd):
        print(f'{nm:>22} | 目录不存在'); continue
    try:
        d = diag_run(rd, ev, nc)
    except Exception as e:
        print(f'{nm:>22} | ERR {type(e).__name__}: {str(e)[:40]}'); continue
    if not d:
        print(f'{nm:>22} | 无逐样本图'); continue
    print(f'{nm:>22} | {d["step"]:>7} {d["n"]:>4} | {d["own"]:7.4f} {d["nn"]:7.4f} '
          f'{d["mean_cand"]:9.4f} {d["gap"]:+6.4f} | {d["hit"]:9.1%} {d["base"]:5.1%}',
          flush=True)
