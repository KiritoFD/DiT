import torch
import json

cache_path = "/home/ds/Workspace/moyi/data/top10_style23/eval_real200_cache.pt"
cache = torch.load(cache_path, map_location="cpu")
rows = cache["rows"]

print(f"Total rows in eval_real200_cache.pt: {len(rows)}")
for i in range(min(25, len(rows))):
    r = rows[i]
    print(f"i={i:3d}: {r.get('calligrapher')} | {r.get('script')} | {r.get('character')} | img_id={r.get('img_id')}")

# Group by calligrapher
calligs = {}
for i, r in enumerate(rows):
    c = r.get("calligrapher")
    if c not in calligs:
        calligs[c] = []
    calligs[c].append(i)

print("\nSamples per calligrapher in cache:")
for c, idxs in calligs.items():
    print(f"  {c}: {len(idxs)} samples, first idx={idxs[0]} ({rows[idxs[0]].get('script')} {rows[idxs[0]].get('character')})")
