#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnosis part 12: 泄漏结论复核 —— 全路径比对 + 像素级比对 (防止 basename 撞号假阳性)."""
import csv, os, collections, hashlib

TR = 'assets/train_top10_style23.csv'
rows = list(csv.DictReader(open(TR)))
full = set(r['image_path'] for r in rows)
pref = collections.Counter(r['image_path'].split('/')[1] for r in rows)
print('train_top10 路径前缀分布:', pref.most_common(5))
print('train_top10 全路径去重:', len(full), '/', len(rows))

for ev in ('assets/eval_top10_strict_subset84.csv', 'assets/eval_top10_seen_20.csv'):
    rr = list(csv.DictReader(open(ev)))
    pfx = collections.Counter(r['image_path'].split('/')[1] for r in rr)
    hit_full = [r for r in rr if r['image_path'] in full]
    hit_base = [r for r in rr if os.path.basename(r['image_path']) in
                set(os.path.basename(p) for p in full)]
    print(f'\n{os.path.basename(ev)} 前缀={pfx.most_common(3)}')
    print(f'  全路径命中 = {len(hit_full)}/{len(rr)}   basename 命中 = {len(hit_base)}/{len(rr)}')
    print('  全路径命中样例:', [r['image_path'] for r in hit_full[:3]])

print('\n==== 逐图字节比对 (全路径命中的前 5 张) ====')
rr = list(csv.DictReader(open('assets/eval_top10_strict_subset84.csv')))
m = [r for r in rr if r['image_path'] in full][:5]
for r in m:
    p = r['image_path']
    ok = os.path.exists(p)
    h = hashlib.md5(open(p, 'rb').read()).hexdigest()[:10] if ok else '-'
    print(f"  {p} exists={ok} md5={h} char={r['character']} callig={r['calligrapher']}")

print('\n==== 训练 csv 里这些图是否真的带 std/aux shard (即真的进过训练) ====')
idx = {r['image_path']: r for r in rows}
for r in m:
    t = idx.get(r['image_path'])
    if t:
        print(f"  {r['image_path']}: train_row std={t.get('std_path')} aug={t.get('aug','')!r} "
              f"slot={t.get('slot_name')}")

print('\n==== v26_gtskel 的评测集 (seen_pred 是什么) ====')
import json
c = json.load(open('src/train/configs/v26_gtskel.json'))
print('  ', c['in_mem_eval_sets'])
print('   std shards:', c.get('skel_latent_shards_dir'), '| img shards:', c.get('latent_shards_dir'))
