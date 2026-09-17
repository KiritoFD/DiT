"""单独测 make_eval_cache —— v13 第一版就是在这里崩的（书家 id 越界）。

不跑训练，只构造 eval 缓存，几秒钟就能验证 id 是否全部可映射。
"""
import os
import sys

import torch

os.chdir("/root/Workspace/xy/DiT")
sys.path.insert(0, "/root/Workspace/xy/DiT")
from src.eval.inference import make_eval_cache  # noqa: E402
from src.utils.callig_map import load_callig_id_map  # noqa: E402

CMAP, N = load_callig_id_map("assets/callig_id_map_50k.json")
print(f"  callig_id_map_50k: {N} 个书家")

for name, csvp in (("seen", "assets/eval_v13_seen.csv"),
                   ("strict", "assets/eval_v13_strict.csv")):
    print(f"\n  === {name}: {csvp} ===")
    try:
        cache = make_eval_cache(
            csvp, None, None, 256, 10, 8, 4, 0.18215,
            skel_latent_shards_dir="data/50k/shards_std",
            callig_id_map=CMAP)
        print(f"    OK  cache 构造成功")
        import torch as _t
        sl = cache["skels_latent"]
        gt = cache["gts"]
        print(f"      skels_latent 非零? {bool(sl.abs().sum() > 0)}  "
              f"abs.mean={float(sl.abs().mean()):.4f} abs.max={float(sl.abs().max()):.4f}")
        print(f"      gts 范围 [{float(gt.min()):.2f}, {float(gt.max()):.2f}]")
        if "conds" in cache:
            cs = cache["conds"]
            cid = [int(c[0]) for c in cs] if isinstance(cs, (list, tuple)) else []
            if cid:
                print(f"      conds 书家索引: min={min(cid)} max={max(cid)} (表 {N}，越界 {sum(1 for c in cid if c >= N)})")
        # 看 conds 里的书家索引范围
        import numpy as np
        for k, v in (cache.items() if isinstance(cache, dict) else []):
            if hasattr(v, "shape"):
                print(f"      {k}: {tuple(v.shape)}")
    except Exception as e:
        print(f"    ✗ 失败: {type(e).__name__}: {str(e)[:300]}")
