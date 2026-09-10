import csv
from collections import defaultdict, Counter

rows = list(csv.DictReader(open('/root/Workspace/xy/DiT/5script/train_fame3_clean_v8.csv', encoding='utf-8')))
# 组合 key = (calligrapher_id, glyph_id/character_id)
def key(r):
    return (int(r['calligrapher_id']), int(r.get('glyph_id', r.get('character_id', 0))))
combo = defaultdict(int)
for r in rows:
    combo[key(r)] += 1
n_total = len(combo)
n_single = sum(1 for v in combo.values() if v == 1)
print(f'(书家,字) 组合数 = {n_total}')
print(f'单样本组合 = {n_single} ({n_single/n_total*100:.1f}%)')
print(f'样本数分布: 1样本={sum(1 for v in combo.values() if v==1)}, 2={sum(1 for v in combo.values() if v==2)}, 3={sum(1 for v in combo.values() if v==3)}, >=4={sum(1 for v in combo.values() if v>=4)}')
# 单样本组合占总训练样本的比例
n_samples = len(rows)
single_samples = sum(1 for r in rows if combo[key(r)] == 1)
print(f'落在单样本组合里的训练样本 = {single_samples}/{n_samples} ({single_samples/n_samples*100:.1f}%)')