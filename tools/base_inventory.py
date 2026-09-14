# -*- coding: utf-8 -*-
"""base full inventory: calligraphers, chars, std skel status."""
import collections as C
import csv
import glob
import os
import re

rows = list(csv.DictReader(open("assets/train_base_noaug.csv", encoding="utf-8")))
out = open("base_inventory.txt", "w", encoding="utf-8")
by_callig = C.Counter(r["calligrapher"] for r in rows)
by_src = C.Counter("unicalli" if "unicalli" in r["image_path"] else
                   ("tongji" if "tongji" in r["image_path"] else "fame") for r in rows)
chars = set()
pairs = set()
for r in rows:
    chars.add(r["character"])
    pairs.add((r["script"], r["character"]))
out.write(f"total rows: {len(rows)} (fame {by_src['fame']}, tongji {by_src['tongji']}, "
          f"unicalli {by_src['unicalli']})\n")
out.write(f"calligraphers: {len(by_callig)}\n")
out.write(f"unique chars: {len(chars)}\n")
out.write(f"unique (script,char) pairs: {len(pairs)}\n\n")
for c, v in by_callig.most_common():
    uc = len({r["character"] for r in rows if r["calligrapher"] == c})
    srcs = sorted({("unicalli" if "unicalli" in r["image_path"] else
                    ("tongji" if "tongji" in r["image_path"] else "fame"))
                   for r in rows if r["calligrapher"] == c})
    out.write(f"  {c}: {v} (chars {uc}; {'+'.join(srcs)})\n")

# std skel 覆盖: base 的 (script,char) pairs -> 渲染过的 uid + fame 现有 shard 覆盖
key2uid = {}
for r in csv.DictReader(open("data/skel/std_skel3_base_key2uid.csv", encoding="utf-8")):
    key2uid[(r["script"], r["character"])] = int(r["uid"])
rendered = len(glob.glob("data/skel/std_skel3_base_png/*.png"))
covered_fame = set()
for sp in glob.glob("data/skel/std_skel3_latents_fame_sym/shard_*.npz"):
    with __import__("numpy").load(sp) as d:
        covered_fame.update(int(i) for i in d["img_ids"])
id2key = {}
for r in rows:
    iid = int(re.search(r"(\d+)\.png", r["image_path"]).group(1))
    id2key[iid] = (r["script"], r["character"])
n_fame_cov = sum(1 for iid, k in id2key.items() if iid in covered_fame)
n_new = sum(1 for iid, k in id2key.items() if iid not in covered_fame)
have_png = sum(1 for iid, k in id2key.items() if key2uid.get(k) and
               os.path.isfile(f"data/skel/std_skel3_base_png/{key2uid[k]}.png"))
out.write(f"\nstd skel: fame 已覆盖 img_ids {n_fame_cov}, 需新渲染行 {n_new}, "
          f"渲染 PNG 就绪 {have_png}/{n_new} (rendered files {rendered})\n")
out.close()
print("done")
