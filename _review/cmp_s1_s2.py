"""看 strict 上两阶段的准确率。"""
import csv
import sys
from collections import Counter

s1f, s2f = sys.argv[1], sys.argv[2]
s1 = {int(r["idx"]): r for r in csv.DictReader(open(s1f, encoding="utf-8"))}
s2 = {int(r["idx"]): r for r in csv.DictReader(open(s2f, encoding="utf-8"))}

n = len(s1)
ex1 = sum(1 for r in s1.values() if r["pred"] == r["char"])
rl1 = sum(int(r["relaxed_eq"]) for r in s1.values())
print(f"  === 一阶段 PaddleOCR-VL 裸 OCR: ===")
print(f"    n={n}  exact={ex1/n:.4f}  relaxed={rl1/n:.4f}")
print(f"    可疑(relaxed 不等): {n-rl1} 条")

n2 = len(s2)
ex2 = sum(1 for r in s2.values() if r["s2_pred"] == r["char"])
rl2 = sum(int(r["s2_relaxed_eq"]) for r in s2.values())
print(f"\n  === 二阶段 Qwen3-VL-4B + 锚点 ===")
print(f"    n={n2}  exact={ex2/n2:.4f}  relaxed={rl2/n2:.4f}")
print(f"    verdict: {dict(Counter(r['verdict'] for r in s2.values()))}")

print(f"\n  两阶段对比（二阶段覆盖一阶段的判定）:")
print(f"    一阶段错、二阶段对 -> 二阶段修正 ✓")
print(f"    一阶段对、二阶段错 -> 二阶段引入错误 ✗")

fix = wrong = 0
for i, r in s1.items():
    if i not in s2:
        continue
    ok1 = r["pred"] == r["char"]
    ok2 = s2[i]["s2_pred"] == s2[i]["char"]
    if not ok1 and ok2:
        fix += 1
    if ok1 and not ok2:
        wrong += 1
print(f"    修正: {fix}  引入错误: {wrong}")

# 最佳组合：任一模型对就算对
both = sum(1 for i, r in s1.items()
           if i in s2 and (r["pred"] == r["char"]
                           or s2[i]["s2_pred"] == s2[i]["char"]))
print(f"\n  ★ 组合（任一模型对就算对）: {both}/{n} = {both/n:.4f}")

print(f"\n  前 20 条对比:")
print(f"    {'char':<4}{'s1':<5}{'s2':<5}{'verdict':<9}raw2")
for i in sorted(s1)[:20]:
    r = s1[i]
    r2 = s2.get(i, {})
    print(f"    {r['char']:<4}{r['pred']:<5}{r2.get('s2_pred',''):<5}"
          f"{r2.get('verdict',''):<9}{r2.get('raw','')[:24]!r}")
