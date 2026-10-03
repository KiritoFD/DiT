#!/bin/bash
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
F=assets/results/v25_stdskel/20260928-214726-v25-stdskel/resolved_config.json
$PY - <<'EOF'
import json
c = json.load(open("assets/results/v25_stdskel/20260928-214726-v25-stdskel/resolved_config.json"))
keys = ["model","depth","hidden_size","num_heads","patch_size",
        "skel_as_glyph_cond","skel_latent_shards_dir","eval_skel_latent_shards_dir",
        "glyph_inject_layers","glyph_inject_mode","glyph_embedder_depth",
        "glyph_vec_cond","glyph_vec_dim","glyph_vec_pool","glyph_scale_init",
        "glyph_drop_prob","glyph_noise_prob","glyph_noise_scale","glyph_patch_drop",
        "glyph_gate_t","glyph_gate_floor","cond_drop_which_glyph_prob",
        "conda_drop","cond_drop_all_prob",
        "lr","global_batch_size","weight_decay","warmup_steps","lr_schedule",
        "ema_decay","ema_interval","optimizer","min_lr_ratio",
        "norm_type","mlp_type","rope","rope_theta","qk_norm","attn_impl",
        "diffusion_type","flow_sampler","t_sampler","t_mean","t_std","shift",
        "max_steps","num_calligraphers","num_characters","no_char_cond",
        "local_ca_layers","use_checkpoint","compile","std_mid_carrier",
        "std_mid_ahi","std_mid_alo","std_mid_lp","std_mid_k_slope",
        "skel_latent_shards_dir_pred","latent_shards_dir","data_csv"]
for k in keys:
    print(f"{k} = {c.get(k)}")
EOF
