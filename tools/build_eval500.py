"""Build eval500_3top30.csv: 500 eval images disjoint from train (img_id).

Selection criteria:
  - img_id NOT in train_3top30_nobeike.csv (strictly disjoint images)
  - (script, character) MUST exist in train CSV -> per-script character_id
    from train's authoritative encoding (character_id is per-script, NOT the
    global manifest char_id!). glyph_id = script_id*7026 + per_script_char_id.
  - ~balanced scripts: 楷/行/隶 (~180/180/140)
  - maximize glyph & calligrapher coverage; reuse eval100 seeds (they already
    use correct per-script encoding).
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

NUM_CHARACTERS = 7026  # per-script char count (glyph_id = script_id*7026 + char_id)
SCRIPT_ID_NAME = {0: "楷", 3: "行", 4: "隶"}  # 3top30 subset

print("Loading manifest...")
manifest = json.load(open(MANIFEST, encoding='utf-8'))
id2entry = {e['img_id']: e for e in manifest}
print(f"  manifest: {len(id2entry)} entries")

print("Loading train CSV (authoritative per-script character_id)...")
train_ids = set()
train_char_map = {}  # (script_name, character) -> per-script character_id
with open(TRAIN_CSV, encoding='utf-8') as f:
    for r in csv.DictReader(f):
        iid = int(os.path.basename(r['image_path']).replace('.png', ''))
        train_ids.add(iid)
        key = (r['script'], r['character'])
        if key not in train_char_map:
            train_char_map[key] = int(r['character_id'])
print(f"  train imgs: {len(train_ids)}, (script,char) pairs: {len(train_char_map)}")

# ── Candidate eval images: in manifest, script in {0,3,4}, not in train,
#    and (script, character) present in train (guarantees valid glyph_id) ──
cand = []
for iid, e in id2entry.items():
    sid = int(e['script_id'])
    if sid not in SCRIPT_ID_NAME or iid in train_ids:
        continue
    key = (SCRIPT_ID_NAME[sid], e['orig_char'])
    if key in train_char_map:
        cand.append((iid, e, key, train_char_map[key]))
print(f"  candidates (script in 3top30, disjoint img, char in train): {len(cand)}")

sd = Counter(e['script_id'] for _, e, _, _ in cand)
print(f"  candidate scripts: {dict(sd)}")

# ── eval100 seeds (correct per-script encoding already) ──
seed_rows = []
with open(EVAL100, encoding='utf-8') as f:
    for r in csv.DictReader(f):
        seed_rows.append(r)
seed_ids = {int(os.path.basename(r['image_path']).replace('.png', '')) for r in seed_rows}
print(f"  eval100 seeds: {len(seed_ids)}")

random.seed(42)
cand = [c for c in cand if c[0] not in seed_ids]

# Target counts: 楷 180, 行 180, 隶 140 (500 total)
targets = {0: 180, 3: 180, 4: 140}

selected = []
for sid, tgt in targets.items():
    script_cand = [c for c in cand if int(c[1]['script_id']) == sid]
    seed_in_script = sum(1 for r in seed_rows if int(r['script_id']) == sid)
    need = tgt - seed_in_script
    if need <= 0:
        continue
    random.shuffle(script_cand)
    # prefer unique glyphs first
    by_glyph = defaultdict(list)
    for i, e, key, cid in script_cand:
        by_glyph[(key[0], key[1])].append((i, e, cid))
    chosen = []
    glyphs = list(by_glyph.keys())
    random.shuffle(glyphs)
    for g in glyphs:
        if len(chosen) >= need:
            break
        chosen.append(random.choice(by_glyph[g]))
    if len(chosen) < need:
        remaining = [c for c in script_cand if (c[0], c[1], c[3]) not in
                     [(i, e, cid) for i, e, cid in chosen]]
        random.shuffle(remaining)
        chosen.extend(remaining[:need - len(chosen)])
    selected.extend((i, e, cid) for i, e, cid in chosen)
    print(f"  script {SCRIPT_ID_NAME[sid]}: target {tgt}, seed {seed_in_script}, "
          f"need {need}, got {len(chosen)}")

print(f"\nselected new: {len(selected)}, seeds: {len(seed_rows)}")

# ── Build output rows with per-script character_id + glyph_id ──
out_rows = []
for iid, e, cid in selected:
    sid = int(e['script_id'])
    out_rows.append({
        'image_path': f"final_imgs_256/{iid}.png",
        'calligrapher': e.get('orig_calli', ''),
        'script': SCRIPT_ID_NAME[sid],
        'character': e.get('orig_char', ''),
        'calligrapher_id': int(e['calli_id']),
        'script_id': sid,
        'character_id': cid,
        'glyph_id': sid * NUM_CHARACTERS + cid,
    })
seen = {int(os.path.basename(r['image_path']).replace('.png', '')) for r in out_rows}
for r in seed_rows:
    iid = int(os.path.basename(r['image_path']).replace('.png', ''))
    if iid not in seen:
        out_rows.append(r)
        seen.add(iid)

# ── Validation ──
out_ids = {int(os.path.basename(r['image_path']).replace('.png', '')) for r in out_rows}
assert not (out_ids & train_ids), "eval500 overlaps train!"
assert len(out_ids) == len(out_rows), "duplicate img_ids in eval500"
for r in out_rows:
    cid = int(r['character_id'])
    gid = int(r['glyph_id'])
    assert 0 <= cid < NUM_CHARACTERS, f"char_id out of range: {r['character']} cid={cid}"
    assert gid < NUM_CHARACTERS * 5, f"glyph_id out of range: {r['character']} gid={gid}"

with open(OUT, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader()
    w.writerows(out_rows)

print(f"\nDone: {len(out_rows)} rows -> {OUT}")
print(f"scripts: {dict(Counter(r['script'] for r in out_rows))}")
print(f"unique glyphs: {len(set(r['glyph_id'] for r in out_rows))}")
print(f"unique calligs: {len(set(r['calligrapher_id'] for r in out_rows))}")
print(f"char_id range: {min(int(r['character_id']) for r in out_rows)}.."
      f"{max(int(r['character_id']) for r in out_rows)}")
print(f"glyph_id range: {min(int(r['glyph_id']) for r in out_rows)}.."
      f"{max(int(r['glyph_id']) for r in out_rows)}")
