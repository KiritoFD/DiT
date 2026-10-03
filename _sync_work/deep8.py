#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnosis part 8: 评测泄漏体检 + 记忆化 vs 泛化的真正拆分."""
import csv, os, collections, statistics as st

B = 'assets/results/{}/eval_stdskel_batch.csv'

def train_index(csvf):
    paths, pairs, chars = set(), set(), set()
    for r in csv.DictReader(open(csvf)):
        p = r.get('image_path') or r.get('img_path') or r.get('img_id')
        paths.add(p); paths.add(os.path.basename(p))
        pairs.add((r['character'], r['calligrapher']))
        chars.add(r['character'])
    return paths, pairs, chars

def key_of(r):
    p = r.get('img_id') or r.get('image_path')
    return p, os.path.basename(p)

print('==== A. top10 训练/评测 泄漏细查 ====')
tp, tpair, tchar = train_index('assets/train_top10_style23.csv')
ev = list(csv.DictReader(open('assets/eval_top10_strict_subset84.csv')))
leak = 0
for r in ev:
    a, b = key_of(r)
    if a in tp or b in tp: leak += 1
print(f'  eval_top10_strict_subset84: {len(ev)} 条, 训练集中出现同图路径 {leak} 条 ({100*leak/len(ev):.0f}%)')
rows = list(csv.DictReader(open('assets/train_top10_style23.csv')))
print(f'  train_top10: rows={len(rows)} unique image_path={len(set(r["image_path"] for r in rows))} '
      f'unique (char,callig)={len(set((r["character"],r["calligrapher"]) for r in rows))}')
print('  训练 csv 里 aug 列取值:', collections.Counter(r.get('aug','') for r in rows).most_common(5))

print('\n==== B. v24_frozenskel / v25 的 strict: 训练内图 vs 训练外图 分开算 ====')
for exp in ('v24_frozenskel', 'v25_stdskel'):
    f = B.format(exp)
    if not os.path.exists(f): continue
    rs = [r for r in csv.DictReader(open(f)) if r['set'] == 'strict']
    sp = max(int(r['step']) for r in rs)
    cur = [r for r in rs if int(r['step']) == sp]
    inn, out = [], []
    for r in cur:
        a, b = key_of(r)
        (inn if (a in tp or b in tp) else out).append(r)
    def m(g, k):
        v = [float(r[k]) for r in g if r[k] not in ('', None)]
        return st.mean(v) if v else float('nan')
    for name, g in (('训练集内(泄漏)', inn), ('训练集外(真泛化)', out)):
        if g:
            print(f'  {exp:16s} @90k {name:14s} n={len(g):3d} ssim={m(g,"ssim"):.4f} '
                  f'ink_ssim={m(g,"ink_ssim"):.4f} ink_iou={m(g,"ink_iou"):.4f} frag={m(g,"frag_ratio"):.2f}')

print('\n==== C. 50k strict (249): 同字训练样本数 vs ink_ssim (v22) ====')
tc = collections.Counter(r['character'] for r in csv.DictReader(open('assets/train_50k_v2_fixed.csv')))
tp50, pair50, _ = train_index('assets/train_50k_v2_fixed.csv')
f = B.format('v22_aug_skelnet_200k')
rs = [r for r in csv.DictReader(open(f)) if r['set'] == 'strict']
sp = max(int(r['step']) for r in rs)
cur = [r for r in rs if int(r['step']) == sp]
buckets = collections.defaultdict(list)
for r in cur:
    n = tc.get(r['char'], 0)
    bk = '0' if n == 0 else ('1-4' if n <= 4 else ('5-14' if n <= 14 else '15+'))
    buckets[bk].append(r)
for bk in ('0','1-4','5-14','15+'):
    g = buckets.get(bk, [])
    if g:
        v = lambda k: st.mean([float(r[k]) for r in g if r[k] not in ('', None)])
        print(f'  同字训练图 {bk:>4s} 张: n={len(g):3d} ink_ssim={v("ink_ssim"):.4f} ink_iou={v("ink_iou"):.4f} '
              f'ssim={v("ssim"):.4f} frag={v("frag_ratio"):.2f}')
pair_seen = [r for r in cur if (r['char'], r['calligrapher']) in pair50]
pair_new = [r for r in cur if (r['char'], r['calligrapher']) not in pair50]
for nm, g in (('配对见过(同字同书家有训练图)', pair_seen), ('配对没见过', pair_new)):
    if g:
        v = lambda k: st.mean([float(r[k]) for r in g if r[k] not in ('', None)])
        print(f'  {nm:28s} n={len(g):3d} ink_ssim={v("ink_ssim"):.4f} ink_iou={v("ink_iou"):.4f} frag={v("frag_ratio"):.2f}')

print('\n==== D. v25 目前进度 & 是否值得继续 ====')
f = B.format('v25_stdskel')
rs = [r for r in csv.DictReader(open(f))]
sp = sorted({int(r['step']) for r in rs})
for s in ('seen', 'strict'):
    line = []
    for st_ in sp:
        g = [r for r in rs if r['set'] == s and int(r['step']) == st_]
        if g:
            line.append(f'{st_//1000}k:{st.mean([float(r["ink_ssim"]) for r in g]):.3f}')
    print(f'  v25 {s:6s} ink_ssim ' + ' -> '.join(line))
