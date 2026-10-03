#!/bin/bash
# 64² 像素域 Flow Matching (主干口径: logit-normal t / velocity / MSE / Heun)
# batch 拉满显存, OOM 自动降级
cd /root/Workspace/xy/DiT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pkill -f 'train_skelnet_[pd]ix' 2>/dev/null
pkill -f 'sweep_[w]idth' 2>/dev/null
sleep 6

try_batch() {
  B=$1
  L=$2
  rm -f /tmp/fm64.log
  nohup /opt/conda/envs/cu121/bin/python -u tools/train_skelnet_fm64.py \
      --steps 10000 --batch $B --base 32 --lr $L \
      --t-sampler logit_normal --sampler heun --sample-steps 50 \
      --eval-every 1000 --val-n 128 --es-patience 5 \
      --log logs/skelnet_fm64_v2.log --out assets/skelnet_fm64.pt \
      > /tmp/fm64.log 2>&1 &
  local P=$!
  sleep 60
  if kill -0 $P 2>/dev/null && ! grep -qi 'OutOfMemory' /tmp/fm64.log; then
    echo "BATCH_OK $B (pid $P, lr $L)"
    return 0
  fi
  echo "BATCH_FAIL $B"
  tail -3 /tmp/fm64.log
  kill $P 2>/dev/null
  sleep 5
  return 1
}

try_batch 2048 4e-4 || try_batch 1024 3e-4 || try_batch 512 2e-4

echo "--- 启动日志 ---"
grep -E "落盘数据集|CondUNet|训练|step |eval\]|Traceback|FATAL" /tmp/fm64.log | tail -8
echo "--- GPU ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
