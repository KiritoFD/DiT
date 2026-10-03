#!/bin/bash
# 清干净所有残留 -> batch 2048 重开
cd /root/Workspace/xy/DiT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pkill -f 'train_skelnet_[fp]m' 2>/dev/null
pkill -f 'train_skelnet_[pd]ix' 2>/dev/null
sleep 8
echo "--- 清后 GPU ---"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
ps -eo pid,cmd | grep '[t]rain_skelnet' | head -3

rm -f /tmp/fm64.log
nohup /opt/conda/envs/cu121/bin/python -u tools/train_skelnet_fm64.py \
    --steps 10000 --batch 2048 --base 32 --lr 4e-4 \
    --t-sampler logit_normal --sampler heun --sample-steps 50 \
    --eval-every 1000 --val-n 128 --es-patience 5 \
    --log logs/skelnet_fm64_v3.log --out assets/skelnet_fm64.pt \
    > /tmp/fm64.log 2>&1 &
echo "TRAIN_PID $!"
sleep 90
grep -E "退化样本|落盘数据集|CondUNet|训练|step |eval\]|Traceback|OutOfMemory" /tmp/fm64.log | tail -8
echo "--- GPU ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
