# -*- coding: utf-8 -*-
"""per-calligrapher comparison: fame vs tongji (by script)."""
import collections
import csv
import glob
import os
import zipfile

out = open("callig_compare.txt", "w", encoding="utf-8")

fame = collections.defaultdict(lambda: collections.Counter())
for path in ("assets/train_fame3_clean_v8.csv", "assets/train_fame3_e_full.csv"):
    seen = set()
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            iid = r["image_path"].rsplit("/", 1)[-1]
            if (r["calligrapher"], iid) in seen:
                continue
            seen.add((r["calligrapher"], iid))
            fame[r["calligrapher"]][r["script"]] += 1

z = zipfile.ZipFile(r"E:\Calli-Tongji.zip")
tongji = collections.defaultdict(lambda: collections.Counter())
for n in z.namelist():
    try:
        nn = n.encode("cp437").decode("gbk")
    except Exception:
        nn = n
    parts = n.split("/")
    if len(parts) >= 3 and parts[-1].endswith(".png"):
        callig, script = parts[1].rsplit("-", 1)
        tongji[callig][script] += 1

out.write(f"{'书家':<12} | fame(楷/行/隶 原始图数) | tongji(体:张)\n")
all_names = sorted(set(fame) | set(tongji))
for c in all_names:
    f = fame.get(c)
    f_str = f"{f['楷']}/{f['行']}/{f['隶']}" if f else "-"
    t = tongji.get(c)
    t_str = " ".join(f"{k}{v}" for k, v in sorted(t.items())) if t else "-"
    mark = "  <== tongji 独有" if not f else ""
    out.write(f"{c:<12} | {f_str:>15} | {t_str}\n")
out.write(f"\nfame 原始图数 (clean_v8 去重): {sum(sum(c.values()) for c in fame.values())}\n")
out.write(f"tongji 总: {sum(sum(v.values()) for v in tongji.values())} "
          f"({len(tongji)} 书家)\n")
out.close()
print("done")
