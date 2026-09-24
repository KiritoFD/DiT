#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""每个 run 的**每个 step** 都算同书家富集 + own-mean_c，用于匹配 step 对照。

输出一张长表: run, step, n, own_gt, mean_c, own-mean_c, 同书家命中, 基线, 富集
只用 229 条共有列。
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
    gp, gtp, ch, cal = job
    if not (os.path.exists(gp) and os.path.exists(gtp)):
        return None
    g = img(gp)
    own = _ssim(g, img(gtp))
    cs = cands_of(ch)
    if not cs:
        return None
    ss = [(_ssim(g, img(c['image_path'])), c.get('calligrapher')) for c in cs]
    vals = [s for s, _ in ss]
    ncal = sum(1 for _, c2 in ss if c2 == cal)
    hit = (max([s for s, c2 in ss if c2 == cal] or [-9]) >= max(vals) - 1e-12)
    return own, float(np.mean(vals)), ncal, len(ss), bool(hit)


RUNS = [
    ('E0', 'assets/results/v17_s2_s2z_baseline', 'assets/eval_v13_strict.csv'),
    ('E1', 'assets/results/v17_s2z_ada1_ln', 'assets/eval_v13_strict.csv'),
    ('gate', 'assets/results/v17_glyph_gate_f015', 'assets/eval_v13_strict.csv'),
    ('midskel20', 'assets/results/v17_s2_s2z_midskel20', 'assets/eval_v13_strict.csv'),
    ('inj3old', 'assets/results/v17_inj3_100k', 'assets/eval_v13_strict.csv'),
    ('inj3fixed', 'assets/results/v17_inj3_fixed_100k', 'assets/eval_v13_strict_fixed.csv'),
    ('inj3aug', 'assets/results/v17_inj3_fixed_aug_100k', 'assets/eval_v13_strict_fixed.csv'),
    ('gq', 'assets/results/v17_gq_100k', 'assets/eval_v13_strict_fixed.csv'),
]
out = []
for nm, rd, evp in RUNS:
    if not os.path.isdir(rd):
        continue
    ev = list(csv.DictReader(open(evp, encoding='utf-8')))
    for d in sorted(glob.glob(os.path.join(rd, 'eval_samples_ctrl', 'step*'))):
        m = re.search(r'step(\d+)', os.path.basename(d)); sub = os.path.join(d, 'strict')
        if not (m and os.path.isdir(sub) and os.path.exists(os.path.join(sub, 'g0.png'))):
            continue
        step = int(m.group(1))
        jobs = [(os.path.join(sub, f'g{i}.png'), os.path.join(sub, f'gt{i}.png'),
                 ev[i].get('character'), ev[i].get('calligrapher'))
                for i in range(min(len(ev), 249)) if i not in CHANGED]
        with cf.ThreadPoolExecutor(max_workers=NW) as ex:
            rs = [x for x in ex.map(work, jobs) if x]
        own = np.array([x[0] for x in rs]); mc = np.array([x[1] for x in rs])
        ncal = np.array([x[2] for x in rs]); nn = np.array([x[3] for x in rs])
        hit = np.array([x[4] for x in rs])
        av = ncal > 0
        bs = ncal[av] / np.maximum(nn[av], 1)
        out.append(dict(run=nm, step=step, n=len(rs),
                        own=own.mean(), mc=mc.mean(), d=(own - mc).mean(),
                        hit=hit[av].mean() if av.any() else np.nan,
                        base=bs.mean() if av.any() else np.nan,
                        enr=(hit[av].mean() / bs.mean()) if av.any() and bs.mean() > 0 else np.nan))
        print(f'{nm:>10} step{step:>7}: n={len(rs)} own={own.mean():.4f} mean_c={mc.mean():.4f} '
              f'd={out[-1]["d"]:+.4f} 富集={out[-1]["enr"]:.2f}x', flush=True)

with open('assets/samechar_nn_by_step.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader(); w.writerows(out)
print('-> assets/samechar_nn_by_step.csv')

print('\n=== 匹配 step 横向对照（富集倍数）===')
steps = sorted({r['step'] for r in out})
runs = [r[0] for r in RUNS if any(x['run'] == r[0] for x in out)]
idx = {(r['run'], r['step']): r for r in out}
print(f'{"step":>7} | ' + ' '.join(f'{x:>10}' for x in runs))
for s in steps:
    row = []
    for x in runs:
        v = idx.get((x, s))
        row.append(f'{v["enr"]:10.2f}' if v and v['enr'] == v['enr'] else ' ' * 10)
    print(f'{s:>7} | ' + ' '.join(row))
print('\n=== 匹配 step: own - mean_c（目标特异性）===')
print(f'{"step":>7} | ' + ' '.join(f'{x:>10}' for x in runs))
for s in steps:
    row = []
    for x in runs:
        v = idx.get((x, s))
        row.append(f'{v["d"]:+10.4f}' if v else ' ' * 10)
    print(f'{s:>7} | ' + ' '.join(row))
