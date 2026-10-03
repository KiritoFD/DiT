#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnosis part 6: aug effect attribution, char coverage, enrich trend, collapse trace."""
import csv, os, re, glob, collections, statistics as st

B = 'assets/results/{}/eval_stdskel_batch.csv'
NUM = ('ssim','ink_ssim','ink_iou','frag_ratio')

def rows(exp, s=None, step=None):
    f = B.format(exp)
    if not os.path.exists(f): return []
    out = []
    for r in csv.DictReader(open(f)):
        if s and r['set'] != s: continue
        if step is not None and int(r['step']) != step: continue
        for k in NUM:
            try: r[k] = float(r[k])
            except Exception: r[k] = None
        out.append(r)
    return out

def last_step(exp, s):
    f = B.format(exp)
    if not os.path.exists(f): return None
    ss = [int(r['step']) for r in csv.DictReader(open(f)) if r['set'] == s]
    return max(ss) if ss else None

print('==== A. v22(长尾增广) vs v21(无增广) 逐字差: 增广到底帮了谁 ====')
a18, a21 = {}, {}
for exp, dst in (('v21_skelnet_200k', a18), ('v22_aug_skelnet_200k', a21)):
    sp = last_step(exp, 'strict')
    for r in rows(exp, 'strict', sp):
        if r['ink_ssim'] is not None: dst[r['char']] = r['ink_ssim']
common = sorted(set(a18) & set(a21))
d = sorted(((a21[c] - a18[c], c) for c in common), reverse=True)
print(f'  共同 strict 字 n={len(common)}  平均差={st.mean([x for x,_ in d]):+.4f}')
print('  v22 帮最多的 8 字:', ', '.join(f'{c}{x:+.2f}' for x, c in d[:8]))
print('  v22 伤最多的 8 字:', ', '.join(f'{c}{x:+.2f}' for x, c in d[-8:]))
# are the helped chars rare chars? check train counts per char
cnt = collections.Counter()
for f in ('assets/train_50k_v2.csv', 'assets/train_50k_v2_fixed.csv'):
    if os.path.exists(f):
        for r in csv.DictReader(open(f)): cnt[r['character']] += 1
        break
hi = [x for x, c in d if cnt.get(c, 0) <= 2]
lo = [x for x, c in d if cnt.get(c, 0) > 2]
print(f'  生僻字(train<=2 张) 上的平均差 = {st.mean(hi):+.4f} (n={len(hi)})')
print(f'  常见字(train>2 张) 上的平均差 = {st.mean(lo):+.4f} (n={len(lo)})')

print('\n==== B. 字覆盖 vs strict 口径 ====')
tr_chars = collections.Counter()
for r in csv.DictReader(open('assets/train_50k_v2_fixed.csv')): tr_chars[r['character']] += 1
st_chars = [r['character'] for r in csv.DictReader(open('assets/eval_v13_strict_fixed.csv'))]
print(f'  训练集: {len(tr_chars)} 个不同字, {sum(tr_chars.values())} 张, 每字中位 {st.median(tr_chars.values()):.0f} 张')
print(f'  strict 集: {len(st_chars)} 条 / {len(set(st_chars))} 字, 其中在训练集出现过的: '
      f'{sum(1 for c in st_chars if c in tr_chars)} 条')
print(f'  训练集只出现 1 次的字: {sum(1 for v in tr_chars.values() if v == 1)} '
      f'({100*sum(1 for v in tr_chars.values() if v==1)/len(tr_chars):.0f}%)')

print('\n==== C. cal_enrich（风格跟随）随步数: seen vs strict ====')
PAT = re.compile(r'\[in-mem-eval\] step=(\d+) set=(\w+) n=\d+ ssim=[\d.]+.*?cal_enrich=([\d.]+)x')
for tag, pat in (('v22', 'logs/v22_aug_skelnet_200k.log'), ('v24fs', 'logs/v24_frozenskel.log'),
                 ('v21', 'logs/v21_skelnet_200k.log'), ('v24top10', 'logs/v24_top10_style23.log')):
    if not os.path.exists(pat): continue
    d2 = collections.defaultdict(dict)
    for m in PAT.finditer(open(pat, errors='ignore').read()):
        d2[m.group(2)][int(m.group(1))] = float(m.group(3))
    for s in ('seen', 'strict'):
        ks = sorted(d2[s])
        if not ks: continue
        pick = [ks[0], ks[len(ks)//2], ks[-1]]
        print(f'  {tag:8s} {s:6s} ' + ' -> '.join(f'{k//1000}k:{d2[s][k]:.2f}x' for k in pick))

print('\n==== D. v24_top10(可训 SkelNet) 崩坏的直接证据 ====')
p = 'logs/v24_top10_style23.log'
if os.path.exists(p):
    txt = open(p, errors='ignore').read()
    for m in re.finditer(r'\(step=(\d+)\)[^\n]*Deform[^A-Z]*off=([\d.]+)', txt):
        pass
    offs = [(int(m.group(1)), float(m.group(2))) for m in
            re.finditer(r'\(step=(\d{7})\).*?Deform\(w=1\.00\): ([\d.]+) off=([\d.]+)', txt)]
    offs = [(int(m.group(1)), float(m.group(3))) for m in
            re.finditer(r'\(step=(\d{7})\).*?off=([\d.]+)', txt)]
    if offs:
        print('  off (形变幅度) 轨迹:', ', '.join(f'{s//1000}k:{v:.2f}' for s, v in offs[::max(1, len(offs)//8)]))
    dif = [(int(m.group(1)), float(m.group(2))) for m in
           re.finditer(r'\(step=(\d{7})\).*?Diff: ([\d.]+)', txt)]
    print('  Diff:', ', '.join(f'{s//1000}k:{v:.3f}' for s, v in dif[::max(1, len(dif)//8)]),
          '| 末:', f'{dif[-1][0]//1000}k:{dif[-1][1]:.3f}' if dif else '-')
    print('  NaN 次数:', txt.lower().count('nan'), ' Inf 次数:', txt.count('Inf'))
    for kw in ('grad-norm', 'clip', 'corrupt', 'xavier'):
        lines = [l for l in txt.splitlines() if kw in l]
        if lines: print(f'  [{kw}] 最后一条:', lines[-1][:160])

print('\n==== E. 早停指标是什么（是否用了被白底抬高的 ssim） ====')
import json
for f in sorted(glob.glob('src/train/configs/v2[1-5]*.json')):
    c = json.load(open(f))
    print(f'  {os.path.basename(f):32s} early_stop={c.get("early_stop")} metric={c.get("early_stop_metric", "ssim")} '
          f'min_delta={c.get("early_stop_min_delta")}')
