# -*- coding: utf-8 -*-
"""fame-3 raw data stats + tongji merge plan for fame-tj-kxl (楷/行/隶)."""
import collections
import csv
import zipfile

out = open("fame_tj_plan.txt", "w", encoding="utf-8")

# fame raw per script/callig
by_script = collections.Counter()
by_callig_script = collections.Counter()
strict_pairs = set()
seen_pairs = set()
for path, store in ((r"assets\train_fame3_clean_v8.csv", None),
                    (r"assets\train_fame3_e_full.csv", None)):
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_script[r["script"]] += 1
            by_callig_script[(r["calligrapher"], r["script"])] += 1
with open(r"assets\eval_fame3_strict_clean_v9.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        strict_pairs.add((r["calligrapher"], r["script"], r["character"]))
with open(r"assets\eval_seen_v10.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        seen_pairs.add((r["calligrapher"], r["script"], r["character"]))

out.write("fame3_clean_v8 + e_full scripts: " +
          ", ".join(f"{k}:{v}" for k, v in by_script.most_common()) + "\n\n")
out.write("fame (callig, script) counts:\n")
for (c, s), v in sorted(by_callig_script.items()):
    out.write(f"  {c} {s}: {v}\n")
out.write(f"\nstrict eval pairs: {len(strict_pairs)}, scripts: "
          + str(collections.Counter(s for _, s, _ in strict_pairs)) + "\n")

# tongji
z = zipfile.ZipFile(r"E:\Calli-Tongji.zip")
names = []
for n in z.namelist():
    try:
        nn = n.encode("cp437").decode("gbk")
    except Exception:
        nn = n
    names.append(nn)
tongji = {}
for n in names:
    parts = n.split("/")
    if len(parts) >= 3 and parts[-1].endswith(".png"):
        callig_script = parts[1]
        ch = parts[-1][:-4]
        callig, script = callig_script.rsplit("-", 1)
        tongji.setdefault((callig, script), set()).add(ch)

out.write("\ntongji (callig, script) counts:\n")
for (c, s), chs in sorted(tongji.items()):
    out.write(f"  {c}-{s}: {len(chs)}\n")

# plan: fame kai+xing + tongji kai+xing+li
keep_scripts = {"楷", "行"}
fame_keep = {(c, s): v for (c, s), v in by_callig_script.items() if s in keep_scripts}
tj_keep = {(c, s): len(v) for (c, s), v in tongji.items() if s in keep_scripts}
tj_li = {(c, s): len(v) for (c, s), v in tongji.items() if s == "隶"}
out.write(f"\n=== fame-tj-kxl plan ===\n")
out.write(f"fame keep (楷+行): {sum(fame_keep.values())} rows in ({len(fame_keep)} callig-script)\n")
out.write(f"tongji keep (楷+行): {sum(tj_keep.values())} rows in {len(tj_keep)} dirs\n")
out.write(f"tongji 隶 (new script): {sum(tj_li.values())} rows in {len(tj_li)} dirs\n")
out.write(f"tongji 篆 (excluded): {sum(len(v) for (c, s), v in tongji.items() if s == '篆')}\n")

# leakage check: tongji rows colliding with strict eval pairs
leak = 0
for (c, s), chs in tongji.items():
    if s not in keep_scripts and s != "隶":
        continue
    for ch in chs:
        if (c, s, ch) in strict_pairs:
            leak += 1
out.write(f"tongji rows colliding with strict eval pairs: {leak} (must exclude from train)\n")
leak_seen = sum(1 for (c, s), chs in tongji.items()
                for ch in chs if (c, s, ch) in seen_pairs)
out.write(f"tongji rows colliding with seen eval pairs: {leak_seen}\n")
out.close()
print("written fame_tj_plan.txt")
