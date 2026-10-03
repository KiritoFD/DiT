#!/usr/bin/env python3
import csv, os, sys
os.chdir('/root/Workspace/xy/DiT'); sys.path.insert(0, '.')
tr = list(csv.DictReader(open('assets/train_50k_v2_fixed.csv', encoding='utf-8')))
by = {}
for r in tr:
    by.setdefault(str(r.get('character', '')), []).append(r)
ev = list(csv.DictReader(open('assets/eval_v13_strict_fixed.csv', encoding='utf-8')))
sub = 'assets/results/v17_gq_100k/eval_samples_ctrl/step0070000/strict'
print('训练 csv 的 character 样例:', [r['character'] for r in tr[:5]])
print('评测 csv 的 character 样例:', [r['character'] for r in ev[:5]])
for i in [0, 1, 2, 8]:
    ch = ev[i].get('character')
    cs = by.get(str(ch), [])
    ex = [os.path.exists(c.get('image_path', '')) for c in cs[:3]]
    print(f'i={i} ch={ch!r} len(by[ch])={len(cs)} 前3个 path 存在={ex} '
          f'首path={cs[0].get("image_path") if cs else None!r}')
