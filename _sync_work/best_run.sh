#!/bin/bash
cd /root/Workspace/xy/DiT
D=assets/results/v10b_stdskel_fame3_c41x_cos_e
PY=/opt/conda/envs/cu121/bin/python
echo "=== 目录 ==="
ls -1 "$D" | head -8
F=$(ls -1 "$D"/*/resolved_config.json 2>/dev/null | head -1)
echo "=== resolved_config: $F ==="
$PY - "$F" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
for k in ["experiment_name","model","data_csv","latent_shards_dir","skel_latent_shards_dir",
          "skel_as_glyph_cond","glyph_inject_mode","glyph_inject_layers","glyph_embedder_depth",
          "glyph_vec_cond","glyph_scale_init","glyph_drop_prob","condition_fusion",
          "cond_fusion_norm","xattn_q_pos","lr","lr_schedule","global_batch_size","warmup_steps",
          "weight_decay","ema_decay","max_steps","w_repa","repa_cache_dir","diffusion_type",
          "flow_sampler","t_sampler","shift","norm_type","mlp_type","rope","qk_norm",
          "num_calligraphers","num_characters","no_char_cond","freeze_callig_table",
          "eval_cfg","eval_steps","in_mem_eval","eval_skel_latent_shards_dir","use_checkpoint",
          "cond_drop_all_prob","cond_drop_one_prob","cond_drop_which_glyph_prob"]:
    print(f"  {k} = {c.get(k)}")
EOF
echo
echo "=== 评估曲线 (strict 最好 5 行 / seen 最好 5 行) ==="
head -1 "$D/eval_stdskel_summary.csv"
grep ',strict,' "$D/eval_stdskel_summary.csv" | sort -t, -k5 -rn | head -5
grep ',seen,' "$D/eval_stdskel_summary.csv" | sort -t, -k5 -rn | head -5
