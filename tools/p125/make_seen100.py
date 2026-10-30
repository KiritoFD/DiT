"""Sample 100 training rows for the CPU eval: seen-set fitness reference.

All rows come from the trusted 37-calligrapher tail of stage_p0.csv so every
row has a std skeleton and lands in the generic writer slots 10-46. The eval
rows must reuse the val100 column schema; calligrapher_id is rewritten into
the writer-slot space (that is what the evaluator remaps through).
"""
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path('/root/Workspace/xy/DiT')
src_csv = ROOT / 'assets/p125_20261030/p0_train.csv'
val_csv = ROOT / 'assets/u100_pretrain_20261024/validation_val100.csv'
out_csv = ROOT / 'assets/p125_20261030/eval_seen100.csv'

rows = list(csv.DictReader(open(src_csv, encoding='utf-8-sig')))
trusted = [r for r in rows
           if r.get('writer_trusted') == '1' and r.get('std_path') and r.get('calligrapher_id') != '-1']
print('trusted rows with std:', len(trusted))

val_cols = next(csv.reader(open(val_csv, encoding='utf-8-sig')))
by_writer = defaultdict(list)
for r in trusted:
    by_writer[r['writer_slot']].append(r)
slots = sorted(by_writer)
print('writer slots covered:', slots[0], '..', slots[-1], f'({len(slots)} slots)')

rng = random.Random(125125)
picked = []
i = 0
while len(picked) < 100:
    for s in slots:
        pool = by_writer[s]
        picked.append(pool[rng.randrange(len(pool))])
        if len(picked) >= 100:
            break

def seed_of(img_id):
    return str(int(hashlib.sha1(str(img_id).encode()).hexdigest(), 16) % 2**63)

seen_ids = {r['img_id'] for r in rows}
with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=val_cols)
    w.writeheader()
    for r in picked:
        assert r['img_id'] in seen_ids
        w.writerow({
            'image_path': r['image_path'], 'calligrapher': r['calligrapher'],
            'script': r['script'], 'character': r['character'],
            'calligrapher_id': r['writer_slot'],
            'script_id': r['script_id'], 'character_id': r['character_id'],
            'glyph_id': r['glyph_id'], 'aug': '', 'std_path': r['std_path'],
            'source': r['source'], 'src_image_path': '', 'slot_name': r['slot_name'],
            'pair_id': r['pair_id'], 'img_id': r['img_id'], 'fix': 'False',
            'char_used': 'True', 'split_group': r.get('split_group', ''),
            'duplicate_group': r.get('duplicate_group', ''),
            'pixel_sha256': r['pixel_sha256'],
            'eval_noise_seed': seed_of(r['img_id']),
            'original_character_id': '', 'original_glyph_id': '',
            'original_image_path': '', 'generic': r.get('generic', ''),
        })
json.dump(dict(out=str(out_csv), n=len(picked), slots=[int(s) for s in slots],
               per_slot={s: sum(1 for p in picked if p['writer_slot'] == s) for s in slots}),
          open(ROOT / 'scratch/seen100_meta.json', 'w'), indent=1)
print('wrote', out_csv, 'n=100 over', len(slots), 'writer slots')
