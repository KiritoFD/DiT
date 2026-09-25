#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用**连续统计量 + 配对检验**重判 style-rank 是否有效。

为什么不用 "命中率 17.1% -> 22.0%"：
  那是二值量，只有 41 列可用 -> 2 列翻转就是全部信息，power 极低。

改用两个连续量（每个样本都给出一个实数，power 高得多）：
  ① tgt_spec = ssim(gen, 自己的 GT) − mean(ssim(gen, 同字其他 GT))
     —— 对**全部 218 列**都有定义，配对数多
  ② margin   = max(ssim(gen, 同书家候选)) − max(ssim(gen, 异书家候选))
     —— 直接量"是不是更像对的那个书家"，只对有同书家候选的 41 列有定义
两者都做**逐列配对 t 检验**（同列在两 run 间比较，消掉列间变异）。
"""
import csv
import os
import sys

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
    ('inj3-aug @100k', 'assets/results/v17_inj3_fixed_aug_100k', 100000),
    ('v18 @100k', 'assets/results/v18_style_rank_200k', 100000),
    ('v18 @125k', 'assets/results/v18_style_rank_200k', 125000),
    ('E0 @100k', 'assets/results/v17_s2_s2z_baseline', 100000),
]


def img(p):
    with Image.open(p) as f:
        return np.asarray(f.convert('RGB'), dtype=np.float32) / 255.0


def percol(rd, step):
    d = os.path.join(rd, 'eval_samples_ctrl', 'step%07d' % step, 'strict')
    if not os.path.isdir(d):
        return None
    ev = list(csv.DictReader(open('assets/eval_v13_strict_fixed.csv', encoding='utf-8')))
    tgt, marg = {}, {}
    for i in range(min(249, len(ev))):
        if i in CHANGED:
            continue
        gp, gtp = os.path.join(d, 'g%d.png' % i), os.path.join(d, 'gt%d.png' % i)
        if not (os.path.exists(gp) and os.path.exists(gtp)):
            continue
        r = ev[i]
        ch, cal = str(r.get('character', '')), r.get('calligrapher')
        cs = [c for c in by_char.get(ch, [])
              if c.get('image_path') and os.path.exists(c['image_path'])]
        if not cs:
            continue
        g = img(gp)
        own = float(_ssim(g, img(gtp)))
        same, other = [], []
        for c in cs:
            try:
                s = float(_ssim(g, img(c['image_path'])))
            except Exception:
                continue
            (same if c.get('calligrapher') == cal else other).append(s)
        # ⚠ tgt_spec 只需要**同字**候选（不要求同书家）-> 覆盖 218 列，power 最高
        if same or other:
            tgt[i] = own - float(np.mean(same + other))
        # margin 才需要同书家候选 -> 只覆盖 41 列
        if same and other:
            marg[i] = max(same) - max(other)
    return tgt, marg


res = {}
for tag, rd, st in CASES:
    r = percol(rd, st)
    if r is None:
        print('%-16s 缺图' % tag)
        continue
    res[tag] = r
    t, m = r
    print('%-16s tgt_spec 列数 %3d 均值 %+.5f | margin 列数 %3d 均值 %+.5f'
          % (tag, len(t), np.mean(list(t.values())), len(m),
             np.mean(list(m.values())) if m else float('nan')), flush=True)

print()


def paired(a, b, na, nb, lbl):
    keys = sorted(set(a) & set(b))
    if len(keys) < 5:
        print('  %s: 配对列太少 (%d)' % (lbl, len(keys)))
        return
    d = np.array([b[k] - a[k] for k in keys])
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / se if se > 0 else 0.0
    win = int((d > 0).sum())
    print('  %-10s %s -> %s' % (lbl, na, nb))
    print('     配对列 %d | 均值差 %+.5f | SE %.5f | t = %+.2f | %s 胜 %d/%d'
          % (len(keys), d.mean(), se, t, nb, win, len(keys)))
    print('     %s' % ('★ 显著 (|t|>2)' if abs(t) > 2 else '(不显著, |t|<=2)'))


tags = [t for t in [c[0] for c in CASES] if t in res]
for i in range(len(tags) - 1):
    paired(res[tags[i]][0], res[tags[i + 1]][0], tags[i], tags[i + 1], 'tgt_spec')
for i in range(len(tags) - 1):
    paired(res[tags[i]][1], res[tags[i + 1]][1], tags[i], tags[i + 1], 'margin')

print()
print('参考: 连续量 tgt_spec 覆盖 ~218 列，比二值命中率的 41 列 power 高得多。')
print('      若这里也不显著，说明效应本身就在噪声水平 -> 必须重建更大的评测集。')
