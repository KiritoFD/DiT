#!/bin/bash
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
echo "=== configs 目录里 v13/v21/v22/v23/v27/v29 ==="
ls -1 src/train/configs/ | grep -E '^v(13|19|20|21|22|23|24|27|28|29)' | head -20
echo
for D in v19_gq_e0_100k v20_deform_skel_100k v21_skelnet_200k v22_aug_skelnet_200k \
         v23_splitnorm v24_frozenskel v27_skelnet_front v28_reg v29_skelnet_v2; do
  F=$(ls -1 assets/results/$D/*/resolved_config.json 2>/dev/null | head -1)
  [ -z "$F" ] && { echo "--- $D: (无 resolved_config)"; continue; }
  echo "--- $D"
  $PY - "$F" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
for k in ["model","skel_as_glyph_cond","skel_latent_shards_dir","latent_shards_dir",
          "glyph_inject_layers","glyph_inject_mode","glyph_embedder_depth",
          "glyph_vec_cond","glyph_scale_init","glyph_noise_prob","glyph_noise_scale",
          "glyph_patch_drop","cond_fusion_norm","condition_fusion",
          "deform_skel","deform_trainable","deform_max_off","deform_grid","deform_width",
          "lr","global_batch_size","warmup_steps","weight_decay","ema_decay",
          "w_deform_skel","w_latent_skel","max_steps","experiment_name",
          "norm_type","mlp_type","rope","qk_norm","std_mid_carrier"]:
    print(f"    {k} = {c.get(k)}")
EOF
done
