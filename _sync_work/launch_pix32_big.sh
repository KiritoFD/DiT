#!/bin/bash
# pixel-32, batch 拉满显存 (~20G): 1024 -> 768 -> 512 自动降级
set -u
cd /root/Workspace/xy/DiT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/conda/envs/cu121/bin/python
DS=data/top10_style23/skel32
CKPT=assets/skelnet_fm_pix32.pt
MODEL=DiT-2Cond-S/2

pkill -f 'train_skelnet_[f]m' 2>/dev/null
pkill -f 'chain_[p]ix32' 2>/dev/null
sleep 6

try_batch() {
  B=$1
  L=$2
  rm -f /tmp/p32.log
  nohup $PY -u tools/train_skelnet_fm_dit.py --res 32 --dataset $DS \
      --model $MODEL --glyph-inject-layers 4 --glyph-inject-mode adaln \
      --steps 4000 --batch $B --lr $L --wd 0.02 \
      --w-struct 0.5 --struct-min-t 0.4 --w-ink-ratio 0.3 \
      --eval-every 500 --val-n 128 --es-patience 6 \
      --log logs/skelnet_fm_pix32b.log --out $CKPT > /tmp/p32.log 2>&1 &
  local P=$!
  sleep 70
  if kill -0 $P 2>/dev/null && ! grep -qi 'OutOfMemory' /tmp/p32.log; then
    echo "BATCH_OK $B (lr $L)"
    return 0
  fi
  echo "BATCH_FAIL $B"; tail -2 /tmp/p32.log
  kill $P 2>/dev/null; sleep 5
  return 1
}

try_batch 1024 3e-4 || try_batch 768 2.5e-4 || try_batch 512 2e-4
echo "--- GPU ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "--- 进度 ---"
grep -E 'step |eval\]' logs/skelnet_fm_pix32b.log 2>/dev/null | tail -3
