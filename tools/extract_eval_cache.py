import torch

cache_path = "/home/ds/Workspace/moyi/data/top10_style23/eval_real200_cache.pt"
print(f"Loading {cache_path}...")
cache = torch.load(cache_path, map_location="cpu")
print("Cache keys:", list(cache.keys()))
for k, v in cache.items():
    if hasattr(v, "shape"):
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")
    elif isinstance(v, (list, tuple)):
        print(f"  {k}: len={len(v)}")
    elif isinstance(v, dict):
        print(f"  {k}: dict keys={list(v.keys())[:5]}")
