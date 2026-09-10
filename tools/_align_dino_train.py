import numpy as np
import csv
from collections import defaultdict

d = np.load('/tmp/dino_cls_train.npz')
feat = d['feat']            # (51321, 384)
chars = d['chars']
calligs = [int(x) for x in d['calligs'].tolist()]  # 字符串转 int

# 当前训练集 41 书家
train_rows = list(csv.DictReader(open('/root/Workspace/xy/DiT/5script/train_fame3_clean_v8.csv', encoding='utf-8')))
train_ca = sorted(set(int(r['calligrapher_id']) for r in train_rows))
print('train 书家数:', len(train_ca))

# DINO 里每个书家的样本数 (只关心 train 的书家)
dino_count = defaultdict(int)
for c in calligs:
    dino_count[c] += 1
dino_ca = set(dino_count)
print('DINO 书家数:', len(dino_ca))

missing = [c for c in train_ca if c not in dino_ca]
print('train 有但 DINO 无样本的书家:', missing)
print('train 41 书家在 DINO 中的样本数分布:')
cnts = [dino_count[c] for c in train_ca]
print('  min=%d med=%d max=%d  sum=%d' % (min(cnts), sorted(cnts)[len(cnts)//2], max(cnts), sum(cnts)))

# 每个 train 书家在 DINO 里覆盖多少不同字 (对比学习正对可用性)
ca_chars = defaultdict(set)
for c, ch in zip(calligs, chars.tolist()):
    ca_chars[c].add(ch)
print('train 书家在 DINO 中覆盖字数: min=%d med=%d max=%d' % (
    min(len(ca_chars[c]) for c in train_ca),
    sorted(len(ca_chars[c]) for c in train_ca)[len(train_ca)//2],
    max(len(ca_chars[c]) for c in train_ca)))

# DINO 多出的书家 (44 - 41 = ?)
extra = sorted(dino_ca - set(train_ca))
print('DINO 多出的书家 id:', extra)