"""查 eval cache 里 noise / conds / skels_latent 的长度是否一致。

崩溃现象：forward_with_cfg 里 x=32 而 y_callig=30（进来时 16 vs 15）。
"""
import os
import sys

import torch

os.chdir("/root/Workspace/xy/DiT")
sys.path.insert(0, "/root/Workspace/xy/DiT")
from src.eval.inference import make_eval_cache  # noqa: E402
from src.utils.callig_map import load_callig_id_map  # noqa: E402

CMAP, N = load_callig_id_map("assets/callig_id_map_50k.json")

for name, csvp, n in (("seen", "assets/eval_v13_seen.csv", 20),
                      ("strict", "assets/eval_v13_strict.csv", 50)):
    import csv as _csv
    rows = list(_csv.DictReader(open(csvp, encoding="utf-8")))
    print(f"\n  === {name}: csv {len(rows)} 行, 传入 n={n} ===")
    cache = make_eval_cache(csvp, None, None, 256, n, 8, 4, 0.18215,
                            skel_latent_shards_dir="data/50k/shards_std",
                            callig_id_map=CMAP)
    for k in ("noise", "skels_latent", "gts", "skels"):
        v = cache.get(k)
        print(f"    {k:<15} {tuple(v.shape) if hasattr(v, 'shape') else type(v)}")
    c = cache.get("conds")
    print(f"    {'conds':<15} len={len(c) if c is not None else None}")
    if c:
        print(f"      conds[0] = {c[0]}")
