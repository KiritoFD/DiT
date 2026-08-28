import json, csv, os, sys
sys.stdout.reconfigure(encoding="utf-8")

man = json.load(open("archive/final_manifest.json", encoding="utf-8"))
print("manifest entries:", len(man))
print("sample keys:", list(man[0].keys()))
import collections
snames = collections.Counter(e.get("orig_script", "") for e in man)
print("orig_script dist:", dict(snames))

# dino index coverage for top6 rows
idx = json.load(open("pretrained_models/dino_embeddings/glyph_dino_index.json", encoding="utf-8"))
print("dino index entries:", len(idx))
if isinstance(idx, dict):
    sample = list(idx.items())[:3]
    print("dino index sample:", sample)
else:
    print("dino index sample:", idx[:3])

with open("5script/train_top6.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
gids = set(int(r["glyph_id"]) for r in rows)
print("train_top6 unique glyph_id:", len(gids), "range:", min(gids), max(gids))
if isinstance(idx, dict):
    missing = [g for g in gids if str(g) not in idx]
    print("glyph_ids missing from dino index:", len(missing), missing[:10])
elif isinstance(idx, list):
    # maybe list of {id: ...}?
    keys = set()
    for e in idx:
        if isinstance(e, dict):
            keys.add(e.get("id") or e.get("glyph_id") or e.get("char_id"))
    print("dino list keys sample:", list(keys)[:5] if keys else "n/a")

# local imgs?
for p in ["final_imgs_256", "final_images", "archive/aug_vis"]:
    print(p, "exists locally:", os.path.isdir(p))
import glob
print("local final_imgs_256 count:", len(glob.glob("final_imgs_256/*.png")) if os.path.isdir("final_imgs_256") else "n/a")