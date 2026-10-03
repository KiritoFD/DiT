#!/bin/bash
cd /root/Workspace/xy/DiT
export PYTHONPATH=/root/Workspace/xy/DiT
CK=$(ls -t assets/results/v15c_fixed/*/checkpoints/0210000.pt | head -1)
echo "ckpt=$CK"
echo "=== baseline (正常 g) ==="
rm -rf /tmp/_g_base; mkdir -p /tmp/_g_base
CUDA_VISIBLE_DEVICES="" nice -n 10 /opt/conda/envs/cu121/bin/python -u tools/eval/eval_stdskel_batch.py \
  --results-dir /tmp/_g_base --ckpt-override "$CK" --device cpu \
  --sets "seen:assets/eval_v13_seen_fixed.csv:20" --dit-batch 4 --vae-batch 4 2>&1 | grep -E "ssim=|Error|Traceback" | tail -3
echo "=== g=0 (零 shards) ==="
rm -rf /tmp/_g_zero; mkdir -p /tmp/_g_zero
CUDA_VISIBLE_DEVICES="" nice -n 10 /opt/conda/envs/cu121/bin/python -u tools/eval/eval_stdskel_batch.py \
  --results-dir /tmp/_g_zero --ckpt-override "$CK" --device cpu \
  --skel-shards data/50k/shards_std_zero \
  --sets "seen:assets/eval_v13_seen_fixed.csv:20" --dit-batch 4 --vae-batch 4 2>&1 | grep -E "ssim=|Error|Traceback" | tail -3
echo "=== DONE ==="
