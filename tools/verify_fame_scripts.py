# -*- coding: utf-8 -*-
"""verify fame script_id -> script mapping + lishu coverage."""
import collections
import csv

out = open("fame_script_verify.txt", "w", encoding="utf-8")
sid = collections.Counter()
sc_by_id = collections.defaultdict(collections.Counter)
for path in (r"assets\train_fame3_clean_v8.csv", r"assets\train_fame3_e_full.csv"):
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sid[r["script_id"]] += 1
            sc_by_id[r["script_id"]][r["script"]] += 1
for k in sorted(sc_by_id):
    out.write(f"script_id={k}: total={sid[k]} names={dict(sc_by_id[k])}\n")

# lishu calligraphers in fame
lishu = collections.Counter()
seen_cs = set()
for path in (r"assets\train_fame3_clean_v8.csv", r"assets\train_fame3_e_full.csv"):
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["script"] == "隶":
                lishu[r["calligrapher"]] += 1
                seen_cs.add((r["calligrapher"], r["character"]))
out.write("\nfame 隶书 calligraphers: " +
          ", ".join(f"{k}:{v}" for k, v in lishu.most_common()) + "\n")
out.write(f"fame 隶书 unique chars: {len(seen_cs)}\n")

# tongji lishu calligraphers vs fame
z_names = ["伊秉绶", "吴让之", "林散之", "金农"]
tongji_li = {c: c in lishu for c in z_names}
out.write(f"\ntongji 隶书 calligraphers already in fame: {tongji_li}\n")

# what are tongji's 15 new calligraphers vs fame?
fame_calligs = set()
for path in (r"assets\train_fame3_clean_v8.csv", r"assets\train_fame3_e_full.csv"):
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fame_calligs.add(r["calligrapher"])
z = __import__("zipfile")
zn = z.ZipFile(r"E:\Calli-Tongji.zip")
names = []
for n in zn.namelist():
    try:
        nn = n.encode("cp437").decode("gbk")
    except Exception:
        nn = n
    names.append(nn)
tongji_calligs = set()
for n in names:
    parts = n.split("/")
    if len(parts) >= 3 and parts[-1].endswith(".png"):
        tongji_calligs.add(parts[1].rsplit("-", 1)[0])
new_calligs = tongji_calligs - fame_calligs
out.write(f"\ntongji calligraphers NOT in fame ({len(new_calligs)}): {sorted(new_calligs)}\n")
new_dirs = [p.split('/')[1] for p in names
            if len(p.split('/')) >= 3 and p.split('/')[1].rsplit('-', 1)[0] in new_calligs]
out.write(f"their dirs: {sorted(set(new_dirs))}\n")
out.close()
print("done")
