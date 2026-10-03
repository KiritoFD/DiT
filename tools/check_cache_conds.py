import torch as th
cache = th.load("data/top10_style23/eval_real200_cache.pt", weights_only=False)
conds = cache["conds"]
print(f"Total conds: {len(conds)}")
callig_ids = [c[0] for c in conds]
glyph_ids = [c[1] for c in conds]
print(f"callig_ids min={min(callig_ids)}, max={max(callig_ids)}, unique={len(set(callig_ids))}")
print(f"glyph_ids min={min(glyph_ids)}, max={max(glyph_ids)}, unique={len(set(glyph_ids))}")
print("First 10 conds:", conds[:10])
