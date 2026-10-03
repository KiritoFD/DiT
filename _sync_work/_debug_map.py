# -*- coding: utf-8 -*-
import csv, json, os, glob
charid2char = {}
for r in csv.DictReader(open('assets/train_fame.csv', encoding='utf-8')):
    charid2char[int(r['character_id'])] = r['character']
idx = json.load(open('data/pretrained/dino_embeddings/glyph_dino_index.json', encoding='utf-8'))
glyphs = [tuple(x) for x in idx['glyphs']]
cids = [c for _, c in glyphs]
print('dino cid range:', min(cids), max(cids), 'n distinct', len(set(cids)))
print('csv char_id range:', min(charid2char), max(charid2char), 'n', len(charid2char))
in_csv = sum(1 for c in set(cids) if c in charid2char)
print('dino cids present in csv char_id:', in_csv, '/', len(set(cids)))
kai = set(int(os.path.basename(p)[2:].replace('.png', ''), 16) for p in glob.glob('data/skel/std_skeleton_d3/kai/U+*.png'))
print('kai n=', len(kai))
chars_in_kai = 0
for c in set(cids):
    ch = charid2char.get(c)
    if ch and len(ch) == 1 and ord(ch) in kai:
        chars_in_kai += 1
print('dino distinct chars with skel in kai:', chars_in_kai, '/', len(set(cids)))
for c in list(set(cids))[:8]:
    ch = charid2char.get(c)
    ok = (ch and len(ch) == 1 and ord(ch) in kai) if ch else False
    print('  cid', c, 'char', repr(ch), 'in_kai', ok)
