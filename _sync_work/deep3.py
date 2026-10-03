#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnosis part 3: matched-step compare, per-char determinism across models,
   style-table vs sample count, seen/strict char identity."""
import csv, collections, statistics as st, os

B = 'assets/results/{}/eval_stdskel_batch.csv'
NUM = ('ssim','ink_ssim','ink_iou','frag_ratio','hole_pred')

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

def agg(rs, k):
    v = [r[k] for r in rs if r[k] is not None]
    return st.mean(v) if v else float('nan')

print('==== A. 同步长对照: v24_frozenskel(SkelNet冻结) vs v25_stdskel(无SkelNet) ====')
a, b = rows('v24_frozenskel'), rows('v25_stdskel')
for sp in (5000, 10000, 15000, 20000, 25000):
    for s in ('seen','strict'):
        ra = [r for r in a if int(r['step'])==sp and r['set']==s]
        rb = [r for r in b if int(r['step'])==sp and r['set']==s]
        if ra and rb:
            print(f'  step={sp:>6} {s:6s} v24fs ink={agg(ra,"ink_ssim"):.4f} iou={agg(ra,"ink_iou"):.4f} frag={agg(ra,"frag_ratio"):.2f}'
                  f' | v25 ink={agg(rb,"ink_ssim"):.4f} iou={agg(rb,"ink_iou"):.4f} frag={agg(rb,"frag_ratio"):.2f}')

print('\n==== B. 同一字在多个模型上是否稳定地差（跨实验最差字交集） ====')
exps = ['v21_skelnet_200k','v22_aug_skelnet_200k','v23_splitnorm','v24_frozenskel','v25_stdskel']
per_char = collections.defaultdict(dict)
for e in exps:
    rs = [r for r in rows(e) if r['set']=='strict']
    if not rs: continue
    sp = max(int(r['step']) for r in rs)
    for r in rs:
        if int(r['step'])==sp and r['ink_ssim'] is not None:
            per_char[r['char']][e] = r['ink_ssim']
scored = []
for c, d in per_char.items():
    if len(d) >= 4:
        scored.append((st.mean(d.values()), c, len(d), d))
scored.sort()
print('  跨 4-5 个模型一致最差的 12 个字:')
for m, c, n, d in scored[:12]:
    print(f'    {c}  mean={m:.3f} n={n}  ' + ' '.join(f"{e.split('_')[0]}{e.split('_')[1][:4]}={v:.2f}" for e,v in d.items()))
print('  跨模型一致最好的 8 个字:')
for m, c, n, d in scored[-8:]:
    print(f'    {c}  mean={m:.3f} n={n}')
vals = [m for m,_,_,_ in scored]
print(f'  字间离散: p10={sorted(vals)[len(vals)//10]:.3f} med={st.median(vals):.3f} p90={sorted(vals)[9*len(vals)//10]:.3f} '
      f'(n={len(vals)} 字在>=4个模型上都出现)')

print('\n==== C. 书家维度: 训练样本数 vs strict ink_ssim (v22) ====')
rs = [r for r in rows('v22_aug_skelnet_200k') if r['set']=='strict']
sp = max(int(r['step']) for r in rs)
rs = [r for r in rs if int(r['step'])==sp]
byk = collections.defaultdict(list)
for r in rs:
    if r['ink_ssim'] is not None: byk[r['calligrapher']].append(r['ink_ssim'])
# train csv
cnt = collections.Counter()
for f in ('assets/train_50k_v2_fixed_augmented.csv','assets/train_50k_v2_fixed.csv','assets/train_top10_style23.csv'):
    if os.path.exists(f):
        rd = list(csv.DictReader(open(f)))
        col = None
        for cand in ('calligrapher','callig','style','author'):
            if cand in rd[0]: col = cand; break
        if col:
            c2 = collections.Counter(r[col] for r in rd)
            print(f'  {f}: col={col} rows={len(rd)} uniq={len(c2)}')
            if len(c2) > 40: cnt = c2
if cnt:
    pairs = [(k, sum(v)/len(v), len(v), cnt.get(k, 0)) for k, v in byk.items()]
    xs = [p[3] for p in pairs]; ys = [p[1] for p in pairs]
    if len(xs) > 3:
        mx, my = st.mean(xs), st.mean(ys)
        cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / len(xs)
        r_ = cov / (st.pstdev(xs)*st.pstdev(ys))
        print(f'  Pearson r(样本数, strict ink_ssim) = {r_:+.3f}  n={len(xs)}')
    print('  书家(样本数, ink):', ', '.join(f"{k}:{n}张/{m:.2f}" for k,m,c,n in sorted(pairs, key=lambda p:-p[3])[:10]))
    print('  最差5:', ', '.join(f"{k}:{m:.2f}(n={c},train={n})" for k,m,c,n in sorted(pairs, key=lambda p:p[1])[:5]))

print('\n==== D. seen 集的字是否 = 高频字 ====')
for f in ('assets/eval_v13_seen_fixed.csv','assets/eval_v13_strict_fixed.csv'):
    if os.path.exists(f):
        rd = list(csv.DictReader(open(f)))
        ch = [r.get('char') for r in rd]
        print(f'  {f}: n={len(rd)} uniq_char={len(set(ch))} sample={ch[:12]}')

print('\n==== E. ssim 与 ink_ssim 的相关 + ssim 的区分力 ====')
allr = []
for e in exps:
    for r in rows(e):
        if r['ssim'] is not None and r['ink_ssim'] is not None:
            allr.append((r['ssim'], r['ink_ssim'], r['ink_iou']))
xs = [a for a,_,_ in allr]; ys = [b for _,b,_ in allr]
mx, my = st.mean(xs), st.mean(ys)
cov = sum((x-mx)*(y-my) for x,y in zip(xs,ys))/len(xs)
print(f'  n={len(allr)}  r(ssim, ink_ssim) = {cov/(st.pstdev(xs)*st.pstdev(ys)):+.3f}')
lo = [x for x,y,_ in allr if y < 0.25]
print(f'  ink_ssim<0.25 (很差) 的样本, 其全图 ssim 均值 = {st.mean(lo):.3f}  -> 全图 ssim 看不出这些失败')
