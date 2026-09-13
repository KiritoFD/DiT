import numpy as np
from collections import Counter

d = np.load('/tmp/dino_cls_train.npz')
feat = d['feat']
chars = d['chars']
calligs = d['calligs']
print('feat shape:', feat.shape, 'dtype', feat.dtype)
print('chars shape:', chars.shape, 'dtype', chars.dtype, 'uniq_chars:', len(set(chars.tolist())))
print('calligs shape:', calligs.shape, 'dtype', calligs.dtype)
ca = calligs.tolist()
cc = Counter(ca)
print('uniq_callig:', len(cc))
print('callig id 范围:', min(ca), '..', max(ca))
print('top10 书家样本数:', cc.most_common(10))
# 同书家但不同字 的样本对数 (对比学习正对可用性)
from collections import defaultdict
ca_chars = defaultdict(set)
for c, ch in zip(ca, chars.tolist()):
    ca_chars[c].add(ch)
n_multi = sum(1 for c, s in ca_chars.items() if len(s) >= 2)
print('有 >=2 个不同字的书家数:', n_multi, '/', len(ca_chars))
print('各书家覆盖字数: min=%d med=%d max=%d' % (
    min(len(s) for s in ca_chars.values()),
    sorted(len(s) for s in ca_chars.values())[len(ca_chars)//2],
    max(len(s) for s in ca_chars.values())))
# 与 train csv 的 41 书家是否一致
import csv
train_ca = set()
for r in csv.DictReader(open('/root/Workspace/xy/DiT/assets/train_fame3_clean_v8.csv', encoding='utf-8')):
    train_ca.add(int(r['calligrapher_id']))
print('DINO calligs 集合 == train csv 书家集合:', set(cc) == train_ca)
print('DINO 有但 train 没有:', sorted(set(cc) - train_ca))
print('train 有但 DINO 没有:', sorted(train_ca - set(cc)))