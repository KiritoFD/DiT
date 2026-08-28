import sys
sys.path.insert(0, "tools")
from aug6 import count_combos
rows, combo = count_combos("5script/train_3top30_common.csv")
need = {k: 6 - n for k, n in combo.items() if n < 6}
cnt = {}
for k, n in combo.items():
    cnt[n] = cnt.get(n, 0) + 1
print("orig rows:", len(rows), "combos:", len(combo))
print("combo size dist:", dict(sorted(cnt.items())))
print("sparse combos:", len(need), "total aug:", sum(need.values()))
print("augmented rows:", len(rows) + sum(need.values()))