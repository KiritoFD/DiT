# -*- coding: utf-8 -*-
"""Calli-Tongji vs fame: correct pairing (dir = calligrapher-script)."""
import collections
import csv
import zipfile

z = zipfile.ZipFile(r"E:\Calli-Tongji.zip")
names = []
for n in z.namelist():
    try:
        nn = n.encode("cp437").decode("gbk")
    except Exception:
        nn = n
    names.append(nn)
chars = {}
for n in names:
    parts = n.split("/")
    if len(parts) >= 3 and parts[2].endswith(".png"):
        callig, _, rest = n.partition("/")
        script = rest.split("/", 1)[0] if "/" in rest else ""
        script = parts[2].rsplit("/", 1)[0] if False else script
        # parts = ['Calli-Tongji', '王羲之-行', '的.png', ...] flatten dirs; char = last part
        ch = parts[-1][:-4]
        callig = parts[1]
        chars.setdefault(callig, set()).add(ch)

out = open("calli_tongji_report.txt", "w", encoding="utf-8")
for d in sorted(chars):
    sc = d.rsplit("-", 1)
    out.write(f"{d}: {len(chars[d])} chars\n")
calligs = collections.Counter(d.rsplit("-", 1)[0] for d in chars)
scripts = collections.Counter(d.rsplit("-", 1)[1] for d in chars)
out.write("\nscripts (体): " + ", ".join(f"{k}:{v}" for k, v in scripts.most_common()) + "\n")
out.write(f"calligraphers ({len(calligs)}): " + ", ".join(f"{k}({v}dir)" for k, v in calligs.most_common()) + "\n")

fame_calligs = set()
fame_cs_pairs = set()
fame_csc = set()
for path in (r"assets\train_fame3_clean_v8.csv", r"assets\train_fame3_e_full.csv"):
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fame_calligs.add(r["calligrapher"])
            fame_csc.add((r["calligrapher"], r["script"], r["character"]))

tongji_csc = set()
tongji_cc = set()
for d, chs in chars.items():
    callig, script = d.rsplit("-", 1)
    for ch in chs:
        tongji_csc.add((callig, script, ch))
        tongji_cc.add((callig, ch))
out.write(f"\nfame calligraphers ({len(fame_calligs)}): " + ", ".join(sorted(fame_calligs)) + "\n")
inter_calligs = {d.rsplit('-', 1)[0] for d in chars} & fame_calligs
out.write(f"calligrapher overlap with fame: {len(inter_calligs)} -> {sorted(inter_calligs)}\n")
inter_cc = tongji_cc & {(c, ch) for c, _, ch in fame_csc}
out.write(f"(callig,char) overlap with fame: {len(inter_cc)} of {len(tongji_cc)}\n")
out.write(f"(callig,script,char) exact overlap with fame: {len(tongji_csc & fame_csc)} of {len(tongji_csc)}\n")
# script id mapping check: fame script names
scripts_fame = collections.Counter()
with open(r"assets\train_fame3_clean_v8.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        scripts_fame[r["script"]] += 1
out.write("fame scripts: " + ", ".join(f"{k}:{v}" for k, v in scripts_fame.most_common()) + "\n")
out.close()
print("written calli_tongji_report.txt")
