"""分析 rapidocr 全量结果：简繁错配的分布 + 高置信度子集。"""
import csv
from collections import Counter, defaultdict

rows = list(csv.DictReader(open("assets/ocr_gpu_scan.csv", encoding="utf-8")))
print(f"  n={len(rows)}")
c = Counter(r["relation"] for r in rows)
print(f"  关系: {dict(c)}")

rel = [r for r in rows if r["relation"] == "简繁"]
print(f"\n  ★ 简繁错配: {len(rel)} = {len(rel)/len(rows)*100:.1f}%")

# 按置信度分层
def conf_bin(f):
    return ("0-0.3" if f < 0.3 else "0.3-0.6" if f < 0.6 else
            "0.6-0.9" if f < 0.9 else "0.9-1.0")


print(f"\n  === 简繁错配的置信度分布 ===")
cb = Counter(conf_bin(float(r["conf"])) for r in rel)
for k in ("0-0.3", "0.3-0.6", "0.6-0.9", "0.9-1.0"):
    n = cb.get(k, 0)
    tot = sum(1 for r in rows if conf_bin(float(r["conf"])) == k)
    print(f"    {k}: {n}/{tot} = {n/max(tot,1)*100:.1f}% 是该档内的错配率")

hi = [r for r in rel if float(r["conf"]) >= 0.9]
print(f"\n  ★ 高置信度(>=0.9)简繁错配: {len(hi)}")

print(f"\n  === 按书体 ===")
print(f"    {dict(Counter(r['script'] for r in rel))}")
print(f"\n  === 按书家 Top15 ===")
for k, v in Counter(r["calligrapher"] for r in rel).most_common(15):
    print(f"    {k}: {v}")

print(f"\n  === 错配的字符对 Top25 ===")
pairs = Counter((r["char_csv"], r["char_ocr"]) for r in rel)
for (a, b), v in pairs.most_common(25):
    print(f"    {a} -> {b}: {v}")

print(f"\n  === 高置信度错配样例 20 ===")
for r in hi[:20]:
    print(f"    {r['char_csv']} -> {r['char_ocr']}  conf={r['conf']}  "
          f"[{r['script']}/{r['calligrapher']}]")

# 写出高置信度子集给 VLM 精查
with open("assets/mismatch_hi_conf.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["idx", "image_path", "script", "calligrapher",
                "char_csv", "char_ocr", "conf"])
    for r in hi:
        w.writerow([r["idx"], r["image_path"], r["script"], r["calligrapher"],
                    r["char_csv"], r["char_ocr"], r["conf"]])
print(f"\n  written assets/mismatch_hi_conf.csv ({len(hi)} 条)")

# 低置信度子集（给 VLM）
lo = [r for r in rows if float(r["conf"]) < 0.6]
with open("assets/ocr_low_conf.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["idx", "image_path", "script", "calligrapher",
                "char_csv", "char_ocr", "conf"])
    for r in lo[:5000]:
        w.writerow([r["idx"], r["image_path"], r["script"], r["calligrapher"],
                    r["char_csv"], r["char_ocr"], r["conf"]])
print(f"  written assets/ocr_low_conf.csv ({len(lo)} 条, 前5000 已导出)")
