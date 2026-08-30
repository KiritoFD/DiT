# -*- coding: utf-8 -*-
"""fame 数据集结构探查：组合覆盖密度、eval 字/书家是否已见、未见字难度估计"""
import csv, sys
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def load(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))

tr = load("5script/train_fame.csv")
ev = load("5script/eval_fame_strict.csv")
print(f"train rows={len(tr)}  eval rows={len(ev)}")
print("train cols:", list(tr[0].keys()))
print("eval  cols:", list(ev[0].keys()))

C, K = "character", "calligrapher"
tr_chars = Counter(x[C] for x in tr)
tr_cal = Counter(x[K] for x in tr)
tr_pairs = set((x[K], x[C]) for x in tr)
# 每个字被几个书家写过
char2cal = defaultdict(set)
for x in tr:
    char2cal[x[C]].add(x[K])

nC, nK = len(tr_chars), len(tr_cal)
print(f"\n=== 规模 ===")
print(f"书家={nK}  字={nC}  样本={len(tr)}")
print(f"理论组合={nK*nC}  实际组合={len(tr_pairs)}  覆盖率={len(tr_pairs)/(nK*nC)*100:.2f}%")
print(f"每字平均样本={len(tr)/nC:.2f}  每书家平均样本={len(tr)/nK:.1f}")

print(f"\n=== 每字样本数分布 ===")
cc = Counter(tr_chars.values())
for k in sorted(cc)[:12]:
    print(f"  出现{k}次: {cc[k]}字")
print(f"  ...出现>=20次: {sum(v for k,v in cc.items() if k>=20)}字")
print(f"  仅1次的字: {cc.get(1,0)} ({cc.get(1,0)/nC*100:.1f}%)")
print(f"  仅1位书家写过的字: {sum(1 for c,v in char2cal.items() if len(v)==1)} "
      f"({sum(1 for c,v in char2cal.items() if len(v)==1)/nC*100:.1f}%)")

print(f"\n=== eval 可见性（组合泛化的真实难度）===")
ev_unseen_char = [x for x in ev if x[C] not in tr_chars]
ev_unseen_cal = [x for x in ev if x[K] not in tr_cal]
print(f"  eval n={len(ev)}")
print(f"  字完全未见: {len(ev_unseen_char)}")
print(f"  书家完全未见: {len(ev_unseen_cal)}")
# eval 每个字在 train 里被几个书家写过
ncal_seen = Counter(len(char2cal.get(x[C], ())) for x in ev)
print(f"  eval 字在 train 中被几位书家写过 -> 分布:")
for k in sorted(ncal_seen):
    print(f"    {k}位书家: {ncal_seen[k]}个eval样本 ({ncal_seen[k]/len(ev)*100:.1f}%)")

print(f"\n=== eval 组合是否真未现 ===")
leak = [x for x in ev if (x[K], x[C]) in tr_pairs]
print(f"  组合已出现(泄漏): {len(leak)}")

print(f"\n=== 每书家样本数（前/后5）===")
for k, v in tr_cal.most_common(5):
    print(f"  {k}: {v}")
print("  ...")
for k, v in tr_cal.most_common()[-5:]:
    print(f"  {k}: {v}")
