"""按**正确判据**分析简繁错配。

## 关键修正
rapidocr 是**简体字表**的模型 -> 它对繁体 GT 会输出简体，
所以 `csv=繁体 -> OCR=简体` 是 **OCR 的局限，不是数据错配** ✗

**真错配 = csv 标简体，但 GT 实际写繁体** -> OCR 会输出繁体
判据: s2t(char_csv) == char_ocr  且  char_csv != char_ocr
      （即 OCR 输出的是 csv 的**繁体版**）
"""
import csv
from collections import Counter

from opencc import OpenCC

s2t = OpenCC("s2t")
t2s = OpenCC("t2s")

rows = list(csv.DictReader(open("assets/ocr_gpu_scan.csv", encoding="utf-8")))
print(f"  n={len(rows)}")

# 真错配: csv 简体 -> OCR 繁体（OCR 输出 = csv 的繁体版）
real = []
for r in rows:
    a, b = r["char_csv"], r["char_ocr"]
    if a == b:
        continue
    if s2t.convert(a) == b and s2t.convert(a) != a:
        real.append(r)

print(f"\n  ★ 真错配候选 (csv 简体 -> OCR 繁体): {len(real)} = "
      f"{len(real)/len(rows)*100:.2f}%")

hi = [r for r in real if float(r["conf"]) >= 0.9]
print(f"    其中高置信度(>=0.9): {len(hi)}")

print(f"\n  === 按书体 ===")
print(f"    {dict(Counter(r['script'] for r in real))}")
print(f"\n  === 按书家 Top12 ===")
for k, v in Counter(r["calligrapher"] for r in real).most_common(12):
    print(f"    {k}: {v}")

print(f"\n  === 字符对 Top25 ===")
for (a, b), v in Counter((r["char_csv"], r["char_ocr"])
                         for r in real).most_common(25):
    print(f"    {a} -> {b}: {v}")

print(f"\n  === 高置信度样例 20 ===")
for r in hi[:20]:
    print(f"    {r['char_csv']} -> {r['char_ocr']}  conf={r['conf']}  "
          f"[{r['script']}/{r['calligrapher']}]")

# 导出给 VLM 精查
with open("assets/mismatch_real_candidates.csv", "w", newline="",
          encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["idx", "image_path", "script", "calligrapher",
                "char_csv", "char_ocr", "conf"])
    for r in real:
        w.writerow([r["idx"], r["image_path"], r["script"], r["calligrapher"],
                    r["char_csv"], r["char_ocr"], r["conf"]])
print(f"\n  written assets/mismatch_real_candidates.csv ({len(real)} 条)")

# 反向: csv 繁体 -> OCR 简体（OCR 局限，但也说明 csv 是繁体、GT 可能也是繁体 -> 正常）
rev = [r for r in rows if r["char_csv"] != r["char_ocr"]
       and t2s.convert(r["char_csv"]) == r["char_ocr"]]
print(f"  (对照) csv 繁体 -> OCR 简体: {len(rev)} 条 —— 这是 OCR 的局限，不是错配")
