#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnosis part 7: 口径体检 —— strict 到底是"未见字"还是"未见配对"?"""
import csv, collections, statistics as st, os, re

TRAIN = {'50k': 'assets/train_50k_v2_fixed.csv', 'top10': 'assets/train_top10_style23.csv'}
for tag, f in TRAIN.items():
    if not os.path.exists(f): continue
    rr = list(csv.DictReader(open(f)))
    chars = set(r['character'] for r in rr)
    pairs = set((r['character'], r['calligrapher']) for r in rr)
    pc = collections.Counter((r['character'], r['calligrapher']) for r in rr)
    print(f'== {tag}: {len(rr)} 张, {len(chars)} 字, {len(pairs)} 个(字,书家)配对')
    print(f'   每配对样本数: med={st.median(pc.values()):.0f} mean={st.mean(pc.values()):.2f} '
          f'配对只 1 张的比例 {100*sum(1 for v in pc.values() if v==1)/len(pc):.0f}%')

print()
for tag, ev in (('50k', 'assets/eval_v13_strict_fixed.csv'),
                ('50k', 'assets/eval_v13_seen_fixed.csv'),
                ('top10', 'assets/eval_top10_strict_subset84.csv'),
                ('top10', 'assets/eval_top10_seen_20.csv')):
    if not os.path.exists(ev): continue
    rr = list(csv.DictReader(open(ev)))
    tr = list(csv.DictReader(open(TRAIN[tag])))
    chars = set(r['character'] for r in tr)
    pairs = set((r['character'], r['calligrapher']) for r in tr)
    img = set(os.path.basename(r['image_path']) for r in tr)
    ch_hit = sum(1 for r in rr if r['character'] in chars)
    pr_hit = sum(1 for r in rr if (r['character'], r['calligrapher']) in pairs)
    im_hit = sum(1 for r in rr if os.path.basename(r['image_path']) in img)
    print(f'{os.path.basename(ev):34s} [{tag}] n={len(rr)} '
          f'字在训练集={ch_hit}/{len(rr)} ({100*ch_hit/len(rr):.0f}%) '
          f'配对在训练集={pr_hit}/{len(rr)} ({100*pr_hit/len(rr):.0f}%) '
          f'同图在训练集={im_hit}/{len(rr)}')

print('\n== strict 249 条: 同字在训练集里有多少张(其它书家写的同一字) ==')
tr = collections.Counter(r['character'] for r in csv.DictReader(open('assets/train_50k_v2_fixed.csv')))
rr = list(csv.DictReader(open('assets/eval_v13_strict_fixed.csv')))
v = [tr.get(r['character'], 0) for r in rr]
v2 = [len(set()) for r in rr]
tc = collections.Counter()
tp = collections.Counter()
for r in csv.DictReader(open('assets/train_50k_v2_fixed.csv')):
    tc[r['character']] += 1
print(f'  同字训练样本数: min={min(v)} med={st.median(v):.0f} max={max(v)} '
      f'(=0 的条数 {sum(1 for x in v if x==0)})')
byc = collections.defaultdict(list)
for r in rr: byc[r['character']].append(r['calligrapher'])

print('\n== strict 84 条(top10) 同字训练样本数 ==')
tc2 = collections.Counter(r['character'] for r in csv.DictReader(open('assets/train_top10_style23.csv')))
rr2 = list(csv.DictReader(open('assets/eval_top10_strict_subset84.csv')))
v2 = [tc2.get(r['character'], 0) for r in rr2]
print(f'  min={min(v2)} med={st.median(v2):.0f} max={max(v2)} (=0 的条数 {sum(1 for x in v2 if x==0)})')

print('\n== v24_top10 崩坏的直接证据 ==')
p = 'logs/v24_top10_style23.log'
txt = open(p, errors='ignore').read()
off = [(int(m.group(1)), float(m.group(2))) for m in
       re.finditer(r'\(step=(\d{7})\)[^|]*\|[^|]*off=([\d.]+)', txt)]
dif = [(int(m.group(1)), float(m.group(2))) for m in
       re.finditer(r'\(step=(\d{7})\).*?Diff: ([\d.]+)', txt)]
if off:
    sel = off[::max(1, len(off)//8)]
    print('  Deform off:', ', '.join(f'{s//1000}k:{x:.2f}' for s, x in sel))
if dif:
    sel = dif[::max(1, len(dif)//8)]
    print('  Diff      :', ', '.join(f'{s//1000}k:{x:.3f}' for s, x in sel))
print('  NaN:', txt.lower().count('nan'), '| grad-norm 最后一条:',
      [l[:150] for l in txt.splitlines() if 'grad-norm' in l][-1:])
