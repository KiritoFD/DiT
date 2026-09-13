import csv
mx = 0
with open("/root/Workspace/xy/DiT/5script/train_top30.csv") as f:
    for r in csv.DictReader(f):
        g = int(r["glyph_id"])
        if g > mx:
            mx = g
        sid = int(r["script_id"])
        cid = int(r["character_id"])
        # Check: glyph_id = script_id * NUM_CHARACTERS + char_id
        # From first row: script_id=1, char_id=0, glyph_id=7026 => NUM_CHARACTERS=7026
        if sid == 1 and cid == 0:
            print(f"Check: sid=1 cid=0 glyph={g} => NUM_CHARACTERS={g}")
print(f"max glyph_id: {mx}")
# Also count unique calligrapher_id
cids = set()
with open("/root/Workspace/xy/DiT/5script/train_top30.csv") as f:
    for r in csv.DictReader(f):
        cids.add(int(r["calligrapher_id"]))
print(f"num calligrapher_id: {len(cids)}, max: {max(cids)}")
