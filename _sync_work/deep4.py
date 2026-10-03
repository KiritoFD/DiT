#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnosis part 4: script axis, char complexity, style-hit rate, stroke breaks."""
import csv, collections, statistics as st, os, glob
import numpy as np

B = 'assets/results/{}/eval_stdskel_batch.csv'
EXPS = ['v21_skelnet_200k','v22_aug_skelnet_200k','v23_splitnorm','v24_frozenskel','v25_stdskel']
NUM = ('ssim','ink_ssim','ink_iou','skel_iou','frag_ratio','hole_pred','hole_gt')

def rows(exp):
    f = B.format(exp)
    if not os.path.exists(f): return []
    out = []
    for r in csv.DictReader(open(f)):
        for k in NUM:
            try: r[k] = float(r[k])
            except Exception: r[k] = None
        out.append(r)
    return out

def latest(rs, s):
    sub = [r for r in rs if r['set'] == s]
    if not sub: return []
    sp = max(int(r['step']) for r in sub)
    return [r for r in sub if int(r['step']) == sp]

def mean(v):
    v = [x for x in v if x is not None]
    return st.mean(v) if v else float('nan')

print('==== A. 书体维度 (strict, 各模型最新 ckpt) ====')
train_script = collections.Counter()
for f in ('assets/train_50k_v2_fixed.csv','assets/train_top10_style23.csv'):
    if os.path.exists(f):
        for r in csv.DictReader(open(f)):
            train_script[(os.path.basename(f)[:12], r['script'])] += 1
print('  训练集书体分布:')
for (f, s), c in sorted(train_script.items()):
    print(f'    {f:12s} {s}: {c}')
for e in EXPS:
    rs = rows(e)
    if not rs: continue
    last = latest(rs, 'strict')
    bys = collections.defaultdict(list)
    for r in last: bys[r['script']].append(r)
    line = ' '.join(f"{s}:ink={mean([r['ink_ssim'] for r in v]):.3f}/iou={mean([r['ink_iou'] for r in v]):.3f}/n={len(v)}"
                    for s, v in sorted(bys.items(), key=lambda kv: -len(kv[1])))
    print(f'  {e:24s} {line}')

print('\n==== B. 字复杂度 vs strict ink_ssim (GT 墨量 / 连通块) ====')
try:
    from PIL import Image
    rs = [r for r in rows('v22_aug_skelnet_200k') if r['set'] == 'strict']
    sp = max(int(r['step']) for r in rs)
    rs = [r for r in rs if int(r['step']) == sp]
    feats = {}
    for r in rs:
        p = r['img_id']
        if not os.path.exists(p): continue
        try:
            a = np.asarray(Image.open(p).convert('L'), dtype=np.float32) / 255.0
        except Exception: continue
        ink = (a < 0.5)
        feats.setdefault(r['char'], []).append((ink.mean(), ink.sum()))
    xs, ys = [], []
    for c, v in feats.items():
        dens = st.mean([x[0] for x in v])
        m = [r['ink_ssim'] for r in rs if r['char'] == c and r['ink_ssim'] is not None]
        if m: xs.append(dens); ys.append(st.mean(m))
    if len(xs) > 5:
        mx, my = st.mean(xs), st.mean(ys)
        cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / len(xs)
        r_ = cov / (st.pstdev(xs)*st.pstdev(ys)+1e-9)
        print(f'  GT 墨面占比 vs ink_ssim: Pearson r = {r_:+.3f} (n={len(xs)} 字, v22 strict)')
        print(f'  墨量: min={min(xs):.3f} med={st.median(xs):.3f} max={max(xs):.3f}')
except Exception as ex:
    print('  skipped:', ex)

print('\n==== C. 风格命中率 (poster: 最近邻书家 == 目标书家) ====')
for d in sorted(glob.glob('assets/results/*/posters/strict_train_samechar_nn.csv')):
    exp = d.split('/')[2]
    if not any(k in exp for k in ('v18_style','v21','v22','v23','v24','v25')): continue
    rr = list(csv.DictReader(open(d)))
    if not rr: continue
    hit = sum(1 for r in rr if r.get('nn_calligrapher') == r.get('eval_calligrapher'))
    ns = [float(r['nn_ssim']) for r in rr if r.get('nn_ssim')]
    print(f'  {exp:34s} n={len(rr):3d} 风格命中={hit/len(rr)*100:5.1f}%  nn_ssim={st.mean(ns):.4f}')
for d in sorted(glob.glob('assets/results/*/posters/seen_train_samechar_nn.csv')):
    exp = d.split('/')[2]
    if not any(k in exp for k in ('v22','v24_frozen','v25')): continue
    rr = list(csv.DictReader(open(d)))
    if not rr: continue
    hit = sum(1 for r in rr if r.get('nn_calligrapher') == r.get('eval_calligrapher'))
    print(f'  [seen] {exp:28s} n={len(rr):3d} 风格命中={hit/len(rr)*100:5.1f}%')

print('\n==== D. 笔画断裂 frag_ratio: seen vs strict, 随步数 ====')
for e in EXPS:
    rs = rows(e)
    if not rs: continue
    for s in ('seen','strict'):
        subs = [r for r in rs if r['set'] == s]
        steps = sorted({int(r['step']) for r in subs})
        pts = []
        for sp in [steps[0], steps[len(steps)//2], steps[-1]]:
            cur = [r for r in subs if int(r['step']) == sp]
            pts.append(f'{sp//1000}k:{mean([r["frag_ratio"] for r in cur]):.2f}')
        print(f'  {e:24s} {s:6s} ' + ' -> '.join(pts))

print('\n==== E. seen 集构成 ====')
for f in ('assets/eval_top10_seen_20.csv','assets/eval_top10_strict_subset84.csv',
          'assets/eval_v13_seen_fixed.csv'):
    if not os.path.exists(f): continue
    rr = list(csv.DictReader(open(f)))
    ch = [r.get('character') for r in rr]
    print(f'  {f}: n={len(rr)} uniq_char={len(set(ch))} chars={ch[:10]}')
