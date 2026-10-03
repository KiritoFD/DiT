#!/bin/bash
# 拉满显存: 默认 batch 2048 (~20GB), OOM 自动降级 1536 -> 1024
cd /root/Workspace/xy/DiT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pkill -f 'train_skelnet_[p]ix' 2>/dev/null
pkill -f 'sweep_[w]idth' 2>/dev/null
sleep 6

try_batch() {
  B=$1
  L=$2
  rm -f /tmp/pix64_v3.log
  nohup /opt/conda/envs/cu121/bin/python -u tools/train_skelnet_pix.py \
      --res 64 --steps 2500 --batch $B --base 32 --lr $L \
      --eval-every 250 --val-n 128 --es-patience 8 \
      --log logs/skelnet_pix64_v3.log --out assets/skelnet_pix64_v3.pt \
      > /tmp/pix64_v3.log 2>&1 &
  local P=$!
  sleep 50
  if kill -0 $P 2>/dev/null && ! grep -qi 'OutOfMemory' /tmp/pix64_v3.log; then
    echo "BATCH_OK $B (pid $P, lr $L)"
    return 0
  fi
  echo "BATCH_FAIL $B"
  kill $P 2>/dev/null
  sleep 5
  return 1
}

try_batch 2048 1.5e-3 || try_batch 1536 1.2e-3 || try_batch 1024 9e-4

echo "--- 启动日志 ---"
grep -E "落盘数据集|SkelUNet|step |eval|Traceback|FATAL" /tmp/pix64_v3.log | tail -6
echo "--- GPU ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

# 排宽度扫描 (训练结束自动跑)
sed -i 's#CKPT=.*#CKPT=assets/skelnet_pix64_v3.pt#' _sync_work/sweep_width.sh
nohup bash _sync_work/sweep_width.sh > /tmp/sweep_width.log 2>&1 &
echo "SWEEP_QUEUED"
