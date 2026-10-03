import csv
from collections import Counter

rows = list(csv.DictReader(open("assets/simp_trad_mismatch.csv", encoding="utf-8")))
print(f"  总不一致: {len(rows)}")
c = Counter(r["relation"] for r in rows)
print(f"  按关系: {dict(c)}")

rel = [r for r in rows if r["relation"] == "简繁"]
print(f"\n  ★ 简繁错配: {len(rel)}")
print(f"  {'csv':<5}{'OCR':<5}{'conf':>7}   书体/书家")
for r in rel[:50]:
    print(f"  {r['char_csv']:<5}{r['char_ocr']:<5}{r['conf']:>7}   "
          f"{r['script']}/{r['calligrapher']}")

print(f"\n  === 其他不一致(非简繁, 前15) ===")
oth = [r for r in rows if r["relation"] != "简繁"]
for r in oth[:15]:
    print(f"  csv={r['char_csv']} OCR={r['char_ocr']} conf={r['conf']} "
          f"[{r['script']}/{r['calligrapher']}]")
