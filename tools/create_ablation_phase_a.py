import json
import os

BASE_CONFIG = {
    "model": "DiT-2Cond-S/2",
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
    "compile": False,
    "epochs": 100000,
    "max_steps": 40000,
    "lr": 0.00015,
    "warmup_steps": 2000,
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
    "global_batch_size": 780,
    "use_checkpoint": False,
}

EXPERIMENTS = [
    {
        "name": "exp1_rmsnorm",
        "exp_name": "exp1_v61_s_rmsnorm",
        "comment": "Exp 1: v61-S + RMSNorm (Ablation: LayerNorm -> RMSNorm)",
        "norm_type": "rms",
        "mlp_type": "gelu",
        "qk_norm": 0,
        "rope": 0,
    },
    {
        "name": "exp2_qknorm",
        "exp_name": "exp2_v61_s_qknorm",
        "comment": "Exp 2: v61-S + QK-Norm (Ablation: Add QK-RMSNorm to Attention)",
        "norm_type": "layer",
        "mlp_type": "gelu",
        "qk_norm": 1,
        "rope": 0,
    },
    {
        "name": "exp3_rope",
        "exp_name": "exp3_v61_s_rope",
        "comment": "Exp 3: v61-S + 2D RoPE (Ablation: Absolute sincos -> 2D Axial RoPE)",
        "norm_type": "layer",
        "mlp_type": "gelu",
        "qk_norm": 0,
        "rope": 1,
    },
    {
        "name": "exp4_swiglu",
        "exp_name": "exp4_v61_s_swiglu",
        "comment": "Exp 4: v61-S + SwiGLU (Ablation: Standard GELU MLP -> SwiGLU Gated MLP)",
        "norm_type": "layer",
        "mlp_type": "swiglu",
        "qk_norm": 0,
        "rope": 0,
    },
]

cfg_dir = "/home/ds/Workspace/DiT/experiments/ablation_phase_a/configs"
res_dir = "/home/ds/Workspace/DiT/experiments/ablation_phase_a/results"

os.makedirs(cfg_dir, exist_ok=True)
os.makedirs(res_dir, exist_ok=True)

for exp in EXPERIMENTS:
    cfg = dict(BASE_CONFIG)
    cfg["_comment"] = exp["comment"]
    cfg["experiment_name"] = exp["exp_name"]
    cfg["results_dir"] = os.path.join(res_dir, exp["name"])
    cfg["norm_type"] = exp["norm_type"]
    cfg["mlp_type"] = exp["mlp_type"]
    cfg["qk_norm"] = exp["qk_norm"]
    cfg["rope"] = exp["rope"]

    out_path = os.path.join(cfg_dir, f"{exp['name']}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"Generated: {out_path}")
