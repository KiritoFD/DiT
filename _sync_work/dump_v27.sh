#!/bin/bash
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
for D in v27_skelnet_front v28_reg v29_skelnet_v2 v24_frozenskel v24_top10_style23; do
  F=$(ls -1 assets/results/$D/*/resolved_config.json 2>/dev/null | head -1)
  [ -z "$F" ] && { echo "--- $D: 无 resolved_config"; continue; }
  echo "--- $D"
  $PY - "$F" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
for k in ["experiment_name","model","data_csv","latent_shards_dir","skel_latent_shards_dir",
          "skel_as_glyph_cond","glyph_inject_layers","glyph_embedder_depth","glyph_vec_cond",
          "glyph_scale_init","condition_fusion","cond_fusion_norm","norm_type","mlp_type",
          "rope","qk_norm","deform_skel","deform_trainable","w_deform_skel","w_latent_skel",
          "lr","global_batch_size","warmup_steps","weight_decay","ema_decay","max_steps",
          "num_calligraphers","callig_emb_pretrained","callig_script_map",
          "in_mem_eval","eval_csv","eval_n","eval_steps","std_mid_carrier","t_sampler"]:
    print(f"    {k} = {c.get(k)}")
EOF
done
echo
echo "=== 是否有 config 让 latent_shards_dir 指向骨架 ==="
grep -rn '"latent_shards_dir"' src/train/configs/*.json 2>/dev/null | grep -i skel | head -8
