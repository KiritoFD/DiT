#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""style-rank 是否**确定**有效：配对 McNemar 检验。

为什么不能用 "1.11x -> 1.66x" 下结论:
  n_cal 只有 28 列, 两个比值的差约等于 1-2 次命中 —— 完全可能是噪声。

本脚本做两件正确的事:
  ① **配对**：同一列在两个 run 间比较（消掉列间变异），只统计"一边命中一边没命中"的列
     -> McNemar 精确检验（二项分布）
  ② **不抽样候选**：用**全部**同字候选而不是抽 12 个（n_cal 从 ~28 提到 ~50）
"""
import csv
import os
import sys
from math import comb

import numpy as np
from PIL import Image

os.chdir('/root/Workspace/xy/DiT')
sys.path.insert(0, '.')
from src.eval.metrics import ssim as _ssim  # noqa: E402

_a = list(csv.DictReader(open('assets/eval_v13_strict.csv', encoding='utf-8')))
_b = list(csv.DictReader(open('assets/eval_v13_strict_fixed.csv', encoding='utf-8')))
CHANGED = {i for i, (x, y) in enumerate(zip(_a, _b)) if x['character'] != y['character']}

tr = list(csv.DictReader(open('assets/train_50k_v2_fixed.csv', encoding='utf-8')))
by_char = {}
for r in tr:
    by_char.setdefault(str(r.get('character', '')), []).append(r)

CASES = [
    ('inj3-aug @100k', 'assets/results/v17_inj3_fixed_aug_100k', 100000,
     'assets/eval_v13_strict_fixed.csv'),
    ('v18 @100k', 'assets/results/v18_style_rank_200k', 100000,
     'assets/eval_v13_strict_fixed.csv'),
    ('v18 @125k', 'assets/results/v18_style_rank_200k', 125000,
     'assets/eval_v13_strict_fixed.csv'),
]


def img(p):
    with Image.open(p) as f:
        return np.asarray(f.convert('RGB'), dtype=np.float32) / 255.0


def hits(rd, step, evp):
    """返回 {列下标: 1/0}，1 = 同书家候选在**全部**同字候选里 SSIM 最高。"""
    d = os.path.join(rd, 'eval_samples_ctrl', 'step%07d' % step, 'strict')
    if not os.path.isdir(d):
        return None
    ev = list(csv.DictReader(open(evp, encoding='utf-8')))
    out = {}
    for i in range(min(249, len(ev))):
        if i in CHANGED:
            continue
        gp = os.path.join(d, 'g%d.png' % i)
        if not os.path.exists(gp):
            continue
        r = ev[i]
        ch, cal = str(r.get('character', '')), r.get('calligrapher')
        cs = [c for c in by_char.get(ch, [])
              if c.get('image_path') and os.path.exists(c['image_path'])]
        if not cs:
            continue
        ncal = sum(1 for c in cs if c.get('calligrapher') == cal)
        if ncal == 0:
            continue                       # 只统计"有同书家候选"的列
        g = img(gp)
        best, bs, bs_cal = None, -9.0, -9.0
        for c in cs:
            try:
                s = float(_ssim(g, img(c['image_path'])))
            except Exception:
                continue
            if s > bs:
                bs = s
            if c.get('calligrapher') == cal and s > bs_cal:
                bs_cal = s
        out[i] = int(bs_cal >= bs - 1e-12)
    return out


res = {}
for tag, rd, st, evp in CASES:
    h = hits(rd, st, evp)
    if h is None:
        print('%-16s 缺图' % tag)
        continue
    res[tag] = h
    print('%-16s 有同书家候选的列 %d, 命中 %d (%.1f%%)'
          % (tag, len(h), sum(h.values()), 100 * sum(h.values()) / max(len(h), 1)), flush=True)

print()


def mcnemar(a, b, na, nb):
    keys = sorted(set(a) & set(b))
    b01 = sum(1 for k in keys if a[k] == 0 and b[k] == 1)   # a 没中、b 中
    b10 = sum(1 for k in keys if a[k] == 1 and b[k] == 0)
    n = b01 + b10
    if n == 0:
        print('  %s vs %s: 无差异列' % (nb, na))
        return
    k = min(b01, b10)
    p = sum(comb(n, i) for i in range(k + 1)) / (2 ** n) * 2      # 双尾精确二项
    print('  %-14s -> %-14s  配对列 %d' % (na, nb, len(keys)))
    print('     %s 中/%s 没中: %d     %s 没中/%s 中: %d'
          % (nb, na, b01, nb, na, b10))
    print('     McNemar 双尾 p = %.4f  %s' % (min(p, 1.0),
                                             '★显著' if p < 0.05 else '(不显著)'))


tags = [t for t in ['inj3-aug @100k', 'v18 @100k', 'v18 @125k'] if t in res]
for i in range(len(tags) - 1):
    mcnemar(res[tags[i]], res[tags[i + 1]], tags[i], tags[i + 1])
