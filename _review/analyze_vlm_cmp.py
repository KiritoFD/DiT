"""分析 VLM 的 std-vs-GT 比较结果。"""
import csv
import os
import sys
from collections import Counter

os.chdir("/root/Workspace/xy/DiT")
f = sys.argv[1] if len(sys.argv) > 1 else "assets/vlm_cmp/strict__7b.csv"
rows = list(csv.DictReader(open(f, encoding="utf-8")))
print(f"  {f}: {len(rows)} 条\n")

print(f"  SAME 分布: {dict(Counter(r['same'] for r in rows))}")
print(f"  conf 分布: {dict(Counter(r['conf'] for r in rows).most_common(6))}")
print(f"  GT_CHAR 非'？' 的比例: "
      f"{sum(1 for r in rows if r['gt_char'] not in ('', '？', '?'))}/{len(rows)}\n")

# 关键：GT_CHAR 和 character 不一致的
mism = [r for r in rows if r["gt_char"] not in ("", "？", "?")
        and r["gt_char"] != r["char"]]
print(f"  ★ GT_CHAR != character 的: {len(mism)}/{len(rows)} "
      f"({len(mism)/max(len(rows),1)*100:.1f}%)")
for r in mism[:40]:
    print(f"    {r['char']} -> vlm={r['gt_char']}  same={r['same']} "
          f"conf={r['conf']}  ({r['callig']}/{r['script']})")

# 特别看「升」
print("\n  === 「升」(idx 8, id 044447) ===")
for r in rows:
    if r["char"] == "升" or "044447" in r.get("src", ""):
        print(f"    {r}")

# B 且 GT_CHAR 也不同的（双证据）
both = [r for r in mism if r["same"] == "B"]
print(f"\n  ★★ 双证据（same=B 且 gt_char 不同）: {len(both)}")
for r in both[:20]:
    print(f"    {r['char']} -> {r['gt_char']}  conf={r['conf']}")
