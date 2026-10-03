"""分析二阶段复核结果。"""
import csv
import sys
from collections import Counter

f = sys.argv[1] if len(sys.argv) > 1 else "/tmp/_s2t3.csv"
rows = list(csv.DictReader(open(f, encoding="utf-8")))
n = len(rows)
print(f"  {f}: n={n}")
print(f"  verdict: {dict(Counter(r['verdict'] for r in rows))}")
print(f"  s2_conf: {dict(Counter(r['s2_conf'] for r in rows).most_common(5))}")
print(f"  s2_pred='?' 的比例: "
      f"{sum(1 for r in rows if r['s2_pred'] in ('?',''))/n*100:.1f}%")
print()
print(f"  {'char':<4}{'s1':<5}{'s2':<5}{'conf':<7}{'verdict':<9}raw")
for r in rows[:25]:
    print(f"  {r['char']:<4}{r['s1_pred']:<5}{r['s2_pred']:<5}"
          f"{r['s2_conf']:<7}{r['verdict']:<9}{r['raw'][:26]!r}")

# 高置信的 strong
st = [r for r in rows if r["verdict"] == "strong"
      and r["s2_conf"] and float(r["s2_conf"] or 0) >= 0.9]
print(f"\n  ★ strong 且 conf>=0.9: {len(st)}/{n}")
for r in st[:20]:
    print(f"    {r['char']} -> s1={r['s1_pred']} s2={r['s2_pred']} "
          f"conf={r['s2_conf']}")
