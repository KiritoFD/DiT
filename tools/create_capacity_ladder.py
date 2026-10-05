import json
import os

BASE_CONFIG = {
    "glyph_inject_layers": 0,
    "glyph_inject_mode": "adaln",
    "condition_fusion": "factorized_cat",
    "cond_fusion_norm": "split",
    "triple_table_prefix": "/home/ds/Workspace/DiT/assets/triple_tables_best_minimal/",
    "char_remap_json": "/home/ds/Workspace/DiT/assets/triple_tables_best_minimal/char_remap.json",
    "callig_remap_json": "/home/ds/Workspace/DiT/assets/triple_tables_best_minimal/callig_remap.json",
    "font_remap_json": "/home/ds/Workspace/DiT/assets/triple_tables_best_minimal/font_remap.json",
    "num_calligraphers": 10,
    "num_script_classes": 3,
    "num_characters": 4690,
    "use_script_cond": True,
    "callig_embed_dim": 32,
    "script_embed_dim": 16,
    "char_embed_dim": 256,
    "freeze_triple_tables": True,
    "freeze_callig_table": False,
    "freeze_char_table": False,
    "no_char_cond": False,
    "cond_drop_all_prob": 0.05,
    "cond_drop_one_prob": 0.0,
    "cond_drop_callig_prob": 0.16,
    "cond_drop_script_prob": 0.08,
    "cond_drop_char_prob": 0.08,
    "latent_shards_dir": "/home/ds/Workspace/moyi/data/top10_style23/shards_img",
    "img_root": None,
    "diffusion_type": "flow",
    "t_sampler": "logit_normal",
    "t_mean": 0.0,
    "t_std": 1.0,
    "flow_sampler": "heun",
    "heun_batch": False,
    "shift": 1.0,
    "attn_impl": "sdpa",
    "norm_type": "layer",
    "mlp_type": "gelu",
    "qk_norm": 0,
    "rope": 0,
    "compile": False,
    "epochs": 100000,
    "max_steps": 20000,
    "warmup_steps": 1000,
    "min_lr_ratio": 0.1,
    "lr_schedule": "cosine",
    "weight_decay": 0.0,
    "num_workers": 4,
    "preload": True,
    "preload_workers": 8,
    "ckpt_every": 5000,
    "epoch_steps": 5000,
    "use_ema": True,
    "ema_decay": 0.9999,
    "ema_interval": 4,
    "seed": 0,
    "w_repa": 0.0,
    "early_stop": False,
    "data_csv": "/home/ds/Workspace/moyi/exp-std-csv/train.csv",
    "use_checkpoint": False,
}

TIERS = [
    {
        "id": "tier2_sp",
        "name": "DiT-2Cond-Sp/2",
        "params": "59.2M (1.75x S)",
        "hidden": 512,
        "batch": 580,
        "lr": 0.00015,
        "comment": "Tier 2: Sp/2 (h=512, d=12, 8 heads, ~59M params, Batch 580 -> ~43.6G VRAM)",
    },
    {
        "id": "tier3_b",
        "name": "DiT-2Cond-B/2",
        "params": "131.0M (3.88x S)",
        "hidden": 768,
        "batch": 384,
        "lr": 0.00015,
        "comment": "Tier 3: B/2 (h=768, d=12, 12 heads, ~131M params, Batch 384 -> ~43.5G VRAM)",
    },
    {
        "id": "tier4_b1024",
        "name": "DiT-2Cond-B1024/2",
        "params": "233.0M (6.90x S)",
        "hidden": 1024,
        "batch": 288,
        "lr": 0.00010,
        "comment": "Tier 4: B1024/2 (h=1024, d=12, 8 heads, ~233M params, Batch 288 -> ~44.0G VRAM)",
    },
]

CFG_DIR = "/home/ds/Workspace/DiT/experiments/capacity_ladder/configs"
RES_DIR = "/home/ds/Workspace/DiT/experiments/capacity_ladder/results"
os.makedirs(CFG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

for t in TIERS:
    cfg = dict(BASE_CONFIG)
    cfg["_comment"] = t["comment"]
    cfg["model"] = t["name"]
    cfg["experiment_name"] = f"cap_{t['id']}"
    cfg["results_dir"] = os.path.join(RES_DIR, t["id"])
    cfg["global_batch_size"] = t["batch"]
    cfg["lr"] = t["lr"]

    p = os.path.join(CFG_DIR, f"{t['id']}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"Generated config: {p} (Model: {t['name']}, Batch: {t['batch']})")
