"""分析 strict 的失败模式：按 script 分组 + 看最差/最好的样本。"""
import csv
import os
import statistics as S
import sys

p = sys.argv[1]
rows = [r for r in csv.DictReader(open(p, encoding="utf-8")) if r["set"] == "strict"]
print(f"  strict {len(rows)} 条")

by_script = {}
for r in rows:
    by_script.setdefault(r["script"] or "?", []).append(float(r["ssim"]))
print()
print("  === 按 script ===")
for k in sorted(by_script):
    v = by_script[k]
    print(f"    {k:<6} n={len(v):>3}  mean={S.mean(v):.4f}  med={S.median(v):.4f}  "
          f"min={min(v):.4f}")

sv = sorted(rows, key=lambda r: float(r["ssim"]))
print()
print("  === 最差 12 条 ===")
for r in sv[:12]:
    print(f"    ssim={float(r['ssim']):.4f}  lpips={float(r['lpips']):.3f}  "
          f"{r['script']} {r['char']}  img={r['img_id']}")
print()
print("  === 最好 8 条 ===")
for r in sv[-8:]:
    print(f"    ssim={float(r['ssim']):.4f}  lpips={float(r['lpips']):.3f}  "
          f"{r['script']} {r['char']}  img={r['img_id']}")

vals = [float(r["ssim"]) for r in rows]
print()
print(f"  === 分布 ===")
for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
    n = sum(1 for v in vals if v < t)
    print(f"    ssim < {t}: {n:>3} / {len(vals)}  ({n/len(vals)*100:.1f}%)")
