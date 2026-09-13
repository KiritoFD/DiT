import csv, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from collections import Counter
with open('5script/train_top30_clean.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
print(f"rows: {len(rows)}")
cids = set(int(r['calligrapher_id']) for r in rows)
chars = set(int(r['character_id']) for r in rows)
glyphs = set(int(r['glyph_id']) for r in rows)
sids = set(int(r['script_id']) for r in rows)
print(f"calligrapher_id: {len(cids)} unique, range {min(cids)}-{max(cids)}")
print(f"character_id: {len(chars)} unique, range {min(chars)}-{max(chars)}")
print(f"script_id: {sorted(sids)}")
print(f"glyph_id: {len(glyphs)} unique, range {min(glyphs)}-{max(glyphs)}")
# glyph_id = script_id * 7026 + character_id ?
for r in rows[:3]:
    sid = int(r['script_id']); cid = int(r['character_id']); gid = int(r['glyph_id'])
    print(f"  sid={sid} cid={cid} gid={gid}  sid*7026+cid={sid*7026+cid}  match={gid==sid*7026+cid}")
# num_characters should be max(char_id)+1? or 7026?
print(f"max char_id={max(chars)}, so num_characters should be >= {max(chars)+1}")
print(f"max glyph_id={max(glyphs)}, so num_characters (as glyph) should be >= {max(glyphs)+1}")
# Also check eval100_clean
with open('5script/eval100_top30_clean.csv', encoding='utf-8') as f:
    eval_rows = list(csv.DictReader(f))
print(f"\neval100_clean: {len(eval_rows)} rows")
ecids = set(int(r['calligrapher_id']) for r in eval_rows)
echars = set(int(r['character_id']) for r in eval_rows)
eglyphs = set(int(r['glyph_id']) for r in eval_rows)
print(f"  calligrapher_id: {len(ecids)} unique")
print(f"  character_id: {len(echars)} unique")
print(f"  glyph_id: {len(eglyphs)} unique")
# Are all eval callig_ids in train?
missing_c = ecids - cids
print(f"  eval callig_ids not in train: {len(missing_c)}")
# Are all eval glyph_ids in train?
missing_g = eglyphs - glyphs
print(f"  eval glyph_ids not in train: {len(missing_g)}")
