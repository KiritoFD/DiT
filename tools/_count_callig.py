import csv
from collections import Counter

rows = list(csv.DictReader(open('assets/train_fame3_clean_v8.csv', encoding='utf-8')))
c = Counter(int(r['calligrapher_id']) for r in rows)
print('train 样本数 =', len(rows))
print('train 唯一书家数 =', len(c))
print('calligrapher_id 范围 =', min(c), '..', max(c))
print('样本数 top15 书家:', c.most_common(15))

ev = list(csv.DictReader(open('assets/eval_seen_v10.csv', encoding='utf-8')))
ce = Counter(int(r['calligrapher_id']) for r in ev)
print('eval_seen 样本数 =', len(ev), '唯一书家数 =', len(ce))
print('eval 书家 id 集合:', sorted(ce))
print('eval 书家是否都在 train 内:', set(ce) <= set(c))

# 每个书家的平均样本数 / 是否极端长尾
import statistics
cnts = list(c.values())
print('train 每书家样本数: min=%d 中位=%d max=%d' % (min(cnts), statistics.median(cnts), max(cnts)))