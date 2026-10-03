# -*- coding: utf-8 -*-
"""_chk_cfg.py — 确认 sym config 的关键字段 (lr / 数据 / 资产路径)."""
import json
import os
import sys

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

p = "src/train/configs/v11_pretrain_Sp2_base_sym.json"
c = json.load(open(p, encoding="utf-8"))
keys = ("lr", "warmup_steps", "global_batch_size", "max_steps", "lr_schedule",
        "min_lr_ratio", "data_csv", "latent_shards_dir", "skel_latent_shards_dir",
        "eval_skel_latent_shards_dir", "repa_cache_dir", "glyph_inject_mode",
        "glyph_inject_layers", "rope", "w_repa")
for k in keys:
    print(f"  {k:30s} = {c.get(k)!r}")

print("\n[资产存在性]")
for k in ("data_csv", "latent_shards_dir", "skel_latent_shards_dir",
          "eval_skel_latent_shards_dir", "repa_cache_dir"):
    v = c.get(k)
    if not v:
        print(f"  {k:30s} (空)")
        continue
    ok = os.path.exists(v)
    n = len(os.listdir(v)) if (ok and os.path.isdir(v)) else "-"
    print(f"  {k:30s} {v}  exists={ok} files={n}")
