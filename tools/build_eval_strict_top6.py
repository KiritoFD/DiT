"""Build strict eval for small (top6): same standard as eval500_3top30.

Criteria (mirror tools/build_eval500.py for mid):
  - (script, calligrapher) by NAME in train_top6.csv  -> calligrapher_id from train
  - (script, character)   by NAME in train_top6.csv  -> character_id from train
  - img_id NOT in train_top6.csv (image never trained)
  - script in {楷, 隶} (the two scripts of top6)

Because top6's domain is small, strict candidates = 286 (楷 179 + 隶 107),
so this eval is 286 rows (vs 500 for mid). Prefer unique glyphs first.
Output: 5script/eval_strict_top6.csv  (same schema as eval100_top6)
"""
import os, csv, json, random, sys
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
MANIFEST = os.path.join(ROOT, "archive", "final_manifest.json")
TRAIN_CSV = os.path.join(ROOT, "5script", "train_top6.csv")
OUT = os.path.join(ROOT, "5script", "eval_strict_top6.csv")

NUM_CHARACTERS = 7026
SCRIPT_NAME_TO_ID = {"楷": 0, "隶": 4}
SEED = 42

print("Loading manifest...")
manifest = json.load(open(MANIFEST, encoding='utf-8'))
id2entry = {e['img_id']: e for e in manifest}
print(f"  manifest: {len(id2entry)} entries")

print("Loading train_top6 (authoritative name->id maps)...")
train_ids = set()
train_char_map = {}   # (script, character) -> character_id
train_calli_map = {}  # (script, calligrapher) -> calligrapher_id
with open(TRAIN_CSV, encoding='utf-8') as f:
    for r in csv.DictReader(f):
        iid = int(os.path.basename(r['image_path']).replace('.png', ''))
        train_ids.add(iid)
        skey = (r['script'], r['character'])
        if skey not in train_char_map:
            train_char_map[skey] = int(r['character_id'])
        ckey = (r['script'], r['calligrapher'])
        if ckey not in train_calli_map:
            train_calli_map[ckey] = int(r['calligrapher_id'])
print(f"  train imgs: {len(train_ids)}, (script,char): {len(train_char_map)}, "
      f"(script,calli): {len(train_calli_map)}")

# Candidates: script in {楷,隶}, img NOT in train, (script,char) in train, (script,calli) in train
cand = []
for iid, e in id2entry.items():
    if iid in train_ids:
        continue
    sname = e.get('orig_script', '')
    if sname not in SCRIPT_NAME_TO_ID:
        continue
    skey = (sname, e.get('orig_char', ''))
    if skey not in train_char_map:
        continue
    ckey = (sname, e.get('orig_calli', ''))
    if ckey not in train_calli_map:
        continue
    cand.append((iid, e, skey, ckey))
print(f"  strict candidates: {len(cand)} "
      f"{dict(Counter(e['orig_script'] for _, e, _, _ in cand))}")

random.seed(SEED)
selected = []
by_script = defaultdict(list)
for c in cand:
    by_script[c[1]['orig_script']].append(c)
for sname in ["楷", "隶"]:
    sub = by_script[sname]
    by_glyph = defaultdict(list)
    for i, e, skey, ckey in sub:
        by_glyph[skey].append((i, e, skey, ckey))
    glyphs = list(by_glyph.keys())
    random.shuffle(glyphs)
    for g in glyphs:
        selected.append(random.choice(by_glyph[g]))
    print(f"  {sname}: {len(sub)} cand -> selected {len(glyphs)} unique-glyph rows")

# Build output rows from authoritative train ids
out_rows = []
for iid, e, skey, ckey in selected:
    sid = SCRIPT_NAME_TO_ID[skey[0]]
    char_id = train_char_map[skey]
    calli_id = train_calli_map[ckey]
    glyph_id = sid * NUM_CHARACTERS + char_id
    out_rows.append({
        'image_path': f"final_imgs_256/{iid}.png",
        'calligrapher': e.get('orig_calli', ''),
        'script': skey[0],
        'character': e.get('orig_char', ''),
        'calligrapher_id': calli_id,
        'script_id': sid,
        'character_id': char_id,
        'glyph_id': glyph_id,
    })

# Validation
out_ids = {int(os.path.basename(r['image_path']).replace('.png', '')) for r in out_rows}
assert not (out_ids & train_ids), f"eval overlaps train! {len(out_ids & train_ids)}"
assert len(out_ids) == len(out_rows), "duplicate img_ids"
for r in out_rows:
    sid = int(r['script_id'])
    cid = int(r['character_id'])
    gid = int(r['glyph_id'])
    caid = int(r['calligrapher_id'])
    assert sid in (0, 4), f"script_id bad: {r['script']}"
    assert 0 <= cid < NUM_CHARACTERS
    assert 0 <= gid < NUM_CHARACTERS * 5
    assert gid == sid * NUM_CHARACTERS + cid

with open(OUT, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader()
    w.writerows(out_rows)

print(f"\nDone: {len(out_rows)} rows -> {OUT}")
print(f"scripts: {dict(Counter(r['script'] for r in out_rows))}")
print(f"unique glyphs: {len(set(r['glyph_id'] for r in out_rows))}")
print(f"unique callis: {len(set(r['calligrapher_id'] for r in out_rows))}")
print(f"overlap with train: {len(out_ids & train_ids)}")