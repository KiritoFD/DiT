"""Build eval500_3top30.csv: 500 eval images disjoint from train (img_id).

Selection criteria:
  - img_id NOT in train_3top30_nobeike.csv (strictly disjoint images)
  - ~balanced scripts: 楷/行/隶 (~180/180/140)
  - maximize glyph & calligrapher coverage
  - reuse existing eval100 images (they are already disjoint) as seed
Output: 5script/eval500_3top30.csv with same schema as eval100.
"""
import os, csv, sys, json, random
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
MANIFEST = os.path.join(ROOT, "archive", "final_manifest.json")
TRAIN_CSV = os.path.join(ROOT, "5script", "train_3top30_nobeike.csv")
EVAL100 = os.path.join(ROOT, "5script", "eval100_3top30.csv")
OUT = os.path.join(ROOT, "5script", "eval500_3top30.csv")

SCRIPT_ID_NAME = {0: "楷", 3: "行", 4: "隶"}  # 3top30 subset

print("Loading manifest...")
manifest = json.load(open(MANIFEST, encoding='utf-8'))
id2entry = {e['img_id']: e for e in manifest}
print(f"  manifest: {len(id2entry)} entries")

print("Loading train ids...")
train_ids = set()
with open(TRAIN_CSV, encoding='utf-8') as f:
    for r in csv.DictReader(f):
        iid = int(os.path.basename(r['image_path']).replace('.png', ''))
        train_ids.add(iid)
print(f"  train: {len(train_ids)}")

# Candidate eval images: in manifest, script in {0,3,4}, not in train
cand = []
for iid, e in id2entry.items():
    sid = int(e['script_id'])
    if sid in SCRIPT_ID_NAME and iid not in train_ids:
        cand.append((iid, e))
print(f"  candidates (script in 3top30, disjoint from train): {len(cand)}")

# Script distribution of candidates
sd = Counter(e['script_id'] for _, e in cand)
print(f"  candidate scripts: {dict(sd)}")

# Existing eval100 seeds
seed_rows = []
with open(EVAL100, encoding='utf-8') as f:
    for r in csv.DictReader(f):
        seed_rows.append(r)
seed_ids = {int(os.path.basename(r['image_path']).replace('.png', '')) for r in seed_rows}
print(f"  eval100 seeds: {len(seed_ids)}")

random.seed(42)
# Exclude seeds from candidates (they'll be included as-is)
cand = [(i, e) for i, e in cand if i not in seed_ids]

# Sort candidates by (script, glyph) to diversify glyph coverage
# Target counts: 楷 180, 行 180, 隶 140 (500 total)
targets = {0: 180, 3: 180, 4: 140}

# Stratify: for each script, sample to target - existing seeds in that script
selected = []
for sid, tgt in targets.items():
    script_cand = [(i, e) for i, e in cand if int(e['script_id']) == sid]
    seed_in_script = sum(1 for r in seed_rows if int(r['script_id']) == sid)
    need = tgt - seed_in_script
    if need <= 0:
        continue
    # shuffle for diversity
    random.shuffle(script_cand)
    # prefer unique glyphs first
    by_glyph = defaultdict(list)
    for i, e in script_cand:
        by_glyph[(int(e['script_id']), int(e['char_id']))].append((i, e))
    chosen = []
    # round 1: one per glyph
    glyphs = list(by_glyph.keys())
    random.shuffle(glyphs)
    for g in glyphs:
        if len(chosen) >= need:
            break
        chosen.append(random.choice(by_glyph[g]))
    # round 2: fill remainder from remaining candidates
    if len(chosen) < need:
        remaining = [c for c in script_cand if c not in chosen]
        random.shuffle(remaining)
        chosen.extend(remaining[:need - len(chosen)])
    selected.extend(chosen)
    print(f"  script {SCRIPT_ID_NAME[sid]}: target {tgt}, seed {seed_in_script}, need {need}, got {len(chosen)}")

print(f"\nselected new: {len(selected)}, seeds: {len(seed_rows)}")

# Build output rows
out_rows = []
for iid, e in selected:
    out_rows.append({
        'image_path': f"final_imgs_256/{iid}.png",
        'calligrapher': e.get('orig_calli', ''),
        'script': SCRIPT_ID_NAME[int(e['script_id'])],
        'character': e.get('orig_char', ''),
        'calligrapher_id': int(e['calli_id']),
        'script_id': int(e['script_id']),
        'character_id': int(e['char_id']),
        'glyph_id': int(e['script_id']) * 7026 + int(e['char_id']),
    })
# append seeds (dedupe by img_id)
seen = {int(os.path.basename(r['image_path']).replace('.png', '')) for r in out_rows}
for r in seed_rows:
    iid = int(os.path.basename(r['image_path']).replace('.png', ''))
    if iid not in seen:
        out_rows.append(r)
        seen.add(iid)

# Verify disjointness
out_ids = {int(os.path.basename(r['image_path']).replace('.png', '')) for r in out_rows}
assert not (out_ids & train_ids), "eval500 overlaps train!"
assert len(out_ids) == len(out_rows), "duplicate img_ids in eval500"

with open(OUT, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader()
    w.writerows(out_rows)

print(f"\nDone: {len(out_rows)} rows -> {OUT}")
print(f"scripts: {dict(Counter(r['script'] for r in out_rows))}")
print(f"unique glyphs: {len(set(r['glyph_id'] for r in out_rows))}")
print(f"unique calligs: {len(set(r['calligrapher_id'] for r in out_rows))}")
print(f"unique chars: {len(set(r['character_id'] for r in out_rows))}")
