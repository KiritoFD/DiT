"""按 idx 分区统计二阶段结果，看不同区段的行为差异。"""
import csv
import sys
from collections import Counter

f = sys.argv[1] if len(sys.argv) > 1 else "/tmp/_s2samp.csv"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10   # 分成几个区
rows = list(csv.DictReader(open(f, encoding="utf-8")))
if not rows:
    raise SystemExit("空文件")
mx = max(int(r["idx"]) for r in rows)
step = mx / N
print(f"  {f}: n={len(rows)}  idx 范围 0..{mx}  分 {N} 区")
print(f"\n  总 verdict: {dict(Counter(r['verdict'] for r in rows))}")

bins = {}
for r in rows:
    b = min(N - 1, int(int(r["idx"]) / step))
    bins.setdefault(b, []).append(r)

print(f"\n  {'区':<5}{'idx 范围':<16}{'n':<6}{'reject':<9}{'weak':<8}"
      f"{'strong':<8}{'s2=?'}")
for b in range(N):
    g = bins.get(b, [])
    if not g:
        continue
    c = Counter(x["verdict"] for x in g)
    lo, hi = int(b * step), int((b + 1) * step)
    q = sum(1 for x in g if x["s2_pred"] in ("?", ""))
    print(f"  {b:<5}{f'{lo}-{hi}':<16}{len(g):<6}{c.get('reject',0):<9}"
          f"{c.get('weak',0):<8}{c.get('strong',0):<8}{q}")

# 高置信 strong 的样本（真候选）
st = [r for r in rows if r["verdict"] == "strong"
      and r["s2_conf"] and float(r["s2_conf"] or 0) >= 0.9]
print(f"\n  ★★ strong 且 conf>=0.9（高置信真候选）: {len(st)}/{len(rows)}"
      f"  ({len(st)/len(rows)*100:.1f}%)")
for r in st[:30]:
    print(f"    idx={r['idx']:>6} {r['char']} -> s1={r['s1_pred']} "
          f"s2={r['s2_pred']} conf={r['s2_conf']} "
          f"({r['callig']}/{r['script']})")
