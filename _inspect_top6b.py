import csv, json, sys
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8")

# 1) train_top6 structure
with open("5script/train_top6.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
scripts = Counter(r["script"] for r in rows)
callis = set(r["calligrapher"] for r in rows)
chars = set(r["character"] for r in rows)
print("train_top6: rows=%d scripts=%s callis=%d chars=%d" % (
    len(rows), dict(scripts), len(callis), len(chars)))
sig = Counter((r["script"], r["script_id"]) for r in rows)
print("  script/script_id:", dict(sig))
prefix = Counter(r["image_path"].split("/")[0] for r in rows)
print("  img prefix:", dict(prefix))
cssid = Counter((r["script"], r["calligrapher"], r["calligrapher_id"]) for r in rows)
print("  (script,calli)->calli_id:", dict(list(cssid.items())[:12]))

# 2) eval100_top6 structure
with open("5script/eval100_top6.csv", encoding="utf-8") as f:
    erows = list(csv.DictReader(f))
scripts_e = Counter(r["script"] for r in erows)
callis_e = set(r["calligrapher"] for r in erows)
print("eval100_top6: rows=%d scripts=%s callis=%d" % (len(erows), dict(scripts_e), len(callis_e)))
print("  head:", erows[0])

# 3) manifest script names sample
man = json.load(open("archive/final_manifest.json", encoding="utf-8"))
snames = Counter(e.get("orig_script", "") for e in man)
print("manifest orig_script dist:", dict(snames))
# check image exists for a few train_top6 ids in final_imgs_256
import os
for r in rows[:5]:
    iid = int(r["image_path"].rsplit("/", 1)[1].replace(".png", ""))
    p = f"final_imgs_256/{iid}.png"
    print("  img exists in final_imgs_256:", p, os.path.exists(p))