"""Build eval500_3top30.csv: 500 eval images disjoint from train (img_id).

CRITICAL: manifest 与 train CSV 用两套不兼容的 id 编码
  (manifest: 楷=3/行=7/隶=6 + calli_id 全局 0..1871;
   train/eval100: 楷=0/行=3/隶=4 + calligrapher_id 全局 MCCD 49..994).
因此绝不能用 manifest 的 script_id/calli_id，必须用 train CSV 权威映射:
  - calligrapher_id: train CSV 的 (script, calligrapher) -> 全局 MCCD id
  - character_id:    train CSV 的 (script, character) -> per-script id
  - glyph_id:        = script_id * 7026 + character_id

Selection criteria (user-defined):
  - 书家出现: (script, calligrapher) by NAME in train CSV
  - 字出现:   (script, character) by NAME in train CSV (glyph 见过)
  - 图未见:   img_id NOT in train CSV (这个组合/这张图没训练过)
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
SCRIPT_NAME_TO_ID = {"楷": 0, "行": 3, "隶": 4}

print("Loading manifest...")
manifest = json.load(open(MANIFEST, encoding='utf-8'))
id2entry = {e['img_id']: e for e in manifest}
print(f"  manifest: {len(id2entry)} entries")

print("Loading train CSV (authoritative name->id maps)...")
train_ids = set()
train_char_map = {}   # (script, character) -> per-script character_id
train_calli_map = {}  # (script, calligrapher) -> global calligrapher_id
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

# ── Candidates: script in {楷,行,隶}, img NOT in train, (script,char) in
#    train, (script,calligrapher) in train ──
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
print(f"  candidates (calli+glyph in train, img disjoint): {len(cand)}")
print(f"  by script: {dict(Counter(e['orig_script'] for _, e, _, _ in cand))}")

# ── eval100 seeds (already correct encoding, reuse as-is) ──
seed_rows = []
with open(EVAL100, encoding='utf-8') as f:
    for r in csv.DictReader(f):
        seed_rows.append(r)
seed_ids = {int(os.path.basename(r['image_path']).replace('.png', '')) for r in seed_rows}
print(f"  eval100 seeds: {len(seed_ids)}")

random.seed(42)
cand = [c for c in cand if c[0] not in seed_ids]

# Targets: 楷 180, 行 180, 隶 140 (500 total)
targets = {0: 180, 3: 180, 4: 140}

selected = []
for sid, tgt in targets.items():
    sname = {0: "楷", 3: "行", 4: "隶"}[sid]
    script_cand = [c for c in cand if c[1].get('orig_script') == sname]
    seed_in_script = sum(1 for r in seed_rows if int(r['script_id']) == sid)
    need = tgt - seed_in_script
    if need <= 0:
        continue
    random.shuffle(script_cand)
    # prefer unique glyphs first
    by_glyph = defaultdict(list)
    for i, e, skey, ckey in script_cand:
        by_glyph[skey].append((i, e, skey, ckey))
    chosen = []
    glyphs = list(by_glyph.keys())
    random.shuffle(glyphs)
    for g in glyphs:
        if len(chosen) >= need:
            break
        chosen.append(random.choice(by_glyph[g]))
    if len(chosen) < need:
        remaining = [c for c in script_cand if c not in chosen]
        random.shuffle(remaining)
        chosen.extend(remaining[:need - len(chosen)])
    selected.extend(chosen)
    print(f"  script {sname}: target {tgt}, seed {seed_in_script}, "
          f"need {need}, got {len(chosen)}")

print(f"\nselected new: {len(selected)}, seeds: {len(seed_rows)}")

# ── Build output rows from authoritative train ids ──
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
    caid = int(r['calligrapher_id'])
    sid = int(r['script_id'])
    assert sid in (0, 3, 4), f"script_id bad: {r['script']} sid={sid}"
    assert 0 <= cid < NUM_CHARACTERS, f"char_id out of range: {r['character']} cid={cid}"
    assert 0 <= gid < NUM_CHARACTERS * 5, f"glyph_id out of range: {r['character']} gid={gid}"
    assert 0 <= caid < 1011, f"calligrapher_id out of range: {r['calligrapher']} caid={caid}"
    assert gid == sid * NUM_CHARACTERS + cid, f"glyph_id mismatch: {r['character']}"

with open(OUT, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader()
    w.writerows(out_rows)

print(f"\nDone: {len(out_rows)} rows -> {OUT}")
print(f"scripts: {dict(Counter(r['script'] for r in out_rows))}")
print(f"unique glyphs: {len(set(r['glyph_id'] for r in out_rows))}")
print(f"unique callis: {len(set(r['calligrapher_id'] for r in out_rows))}")
print(f"char_id range: {min(int(r['character_id']) for r in out_rows)}.."
      f"{max(int(r['character_id']) for r in out_rows)}")
print(f"calli_id range: {min(int(r['calligrapher_id']) for r in out_rows)}.."
      f"{max(int(r['calligrapher_id']) for r in out_rows)}")
print(f"glyph_id range: {min(int(r['glyph_id']) for r in out_rows)}.."
      f"{max(int(r['glyph_id']) for r in out_rows)}")
