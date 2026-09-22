# -*- coding: utf-8 -*-
import csv, collections
rows = list(csv.DictReader(open("local_posters/eval_low_ink.csv", encoding="utf-8")))
rows.sort(key=lambda r: float(r["ink"]))  # ink_ssim 升序 (最差在前)
def dist(rs):
    c = collections.Counter(r["script"] for r in rs)
    return c, sum(c.values())
print("全部 249 条 书体分布:", dist(rows))
for k in (20, 40, 60):
    c, n = dist(rows[:k])
    print(f"最差 {k} 条 书体分布: {dict(c)}  隶占比={c.get('隶',0)/n*100:.0f}%")
# 各书体的 ink_ssim 中位
import statistics
for sc in ("楷", "行", "隶"):
    vs = [float(r["ink"]) for r in rows if r["script"] == sc]
    if vs:
        print(f"{sc}: n={len(vs)} ink_ssim 中位={statistics.median(vs):.3f} "
              f"最差10%均值={statistics.mean(sorted(vs)[:max(1,len(vs)//10)]):.3f}")
