import csv, sys
sys.stdout.reconfigure(encoding="utf-8")

def ids(path):
    s = set()
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s.add(int(r["image_path"].rsplit("/", 1)[1].replace(".png", "")))
    return s

tr = ids("5script/train_top6.csv")
for p in ["5script/eval500_top6.csv", "5script/eval_unseen_top6.csv"]:
    ev = ids(p)
    print(p, "rows:", len(ev), "overlap_with_train:", len(ev & tr))
    # also check path prefix consistency
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print("   path prefixes:", set(r["image_path"].split("/")[0] for r in rows))
with open("5script/train_top6.csv", encoding="utf-8") as f:
    trr = list(csv.DictReader(f))
print("train prefixes:", set(r["image_path"].split("/")[0] for r in trr))

# Are eval ids a strict subset distribution? show a few eval ids and whether in manifest+train
import json
man = json.load(open("archive/final_manifest.json", encoding="utf-8"))
id2e = {e["img_id"]: e for e in man}
with open("5script/eval500_top6.csv", encoding="utf-8") as f:
    er = list(csv.DictReader(f))
print("first 5 eval500_top6 ids in manifest:", [(int(r['image_path'].rsplit('/',1)[1].replace('.png','')), r['script'], r['character'], r['calligrapher']) for r in er[:5]])
