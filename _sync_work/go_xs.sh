#!/bin/bash
# 1) 删掉 β-blend 的东西  2) 校验新配置  3) tmux 拉起 stage1-XS 重训
set -u
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python

echo "=== 删 β-blend 产物 ==="
rm -rf data/top10_style23/predskel_s1_seen20_b* \
       data/top10_style23/predskel_s1_strict84_b* \
       assets/results/_calib_s1_b* \
       assets/results/_calib_stdas_pred_v32 \
       _sync_work/blend_sweep.py _sync_work/blend_eval.sh
ls -d data/top10_style23/predskel_s1_* 2>/dev/null

echo "=== 校验 v33 ==="
$PY - <<'EOF' || exit 1
import json
c = json.load(open("src/train/configs/v33_stage1_xs.json", encoding="utf-8"))
for k in ["model","latent_shards_dir","skel_latent_shards_dir","global_batch_size",
          "max_steps","glyph_deform_prob","glyph_noise_prob","glyph_mask_prob",
          "weight_decay","early_stop"]:
    print(f"  {k} = {c.get(k)}")
EOF

echo "=== tmux 拉起 ==="
tmux kill-session -t s1xs 2>/dev/null
pkill -f 'src\.train\.train' 2>/dev/null
sleep 5
tmux new-session -d -s s1xs "cd /root/Workspace/xy/DiT && PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/cu121/bin/python -m src.train.train --config src/train/configs/v33_stage1_xs.json 2>&1 | tee logs/v33_stage1_xs.log"
sleep 3
tmux ls
echo GO_XS_DONE
