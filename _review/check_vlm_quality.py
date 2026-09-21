import csv
from collections import Counter

rows = list(csv.DictReader(open("assets/vlm_full_scan.csv", encoding="utf-8")))
print(f"  n={len(rows)}")
ok = sum(int(r["exact"]) for r in rows)
c = Counter(r["relation"] for r in rows)
print(f"  完全一致: {ok}/{len(rows)} = {ok/len(rows)*100:.1f}%")
print(f"  关系: {dict(c)}")

print(f"\n  === 简繁关系（真错配候选）===")
rel = [r for r in rows if r["relation"] == "简繁"]
for r in rel[:30]:
    print(f"    csv={r['char_csv']} -> VLM={r['char_vlm']}  "
          f"[{r['script']}/{r['calligrapher']}]")

print(f"\n  === 其他不一致（前 15）===")
oth = [r for r in rows if r["relation"] == "other"]
for r in oth[:15]:
    print(f"    csv={r['char_csv']} -> VLM={r['char_vlm']}  "
          f"[{r['script']}/{r['calligrapher']}]")
