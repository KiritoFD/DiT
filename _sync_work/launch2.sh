#!/bin/bash
# 清掉旧口径训练, 用**落盘 64² 数据集** + 大 batch 重开
cd /root/Workspace/xy/DiT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pkill -f 'train_skelnet_[p]ix' 2>/dev/null
sleep 6
nvidia-smi --query-gpu=memory.used --format=csv,noheader

nohup /opt/conda/envs/cu121/bin/python -u tools/train_skelnet_pix.py \
    --res 64 --steps 8000 --batch 512 --base 32 --lr 6e-4 \
    --eval-every 500 --val-n 128 --es-patience 10 \
    --log logs/skelnet_pix64_v2.log --out assets/skelnet_pix64_v2.pt \
    > /tmp/pix64_v2.log 2>&1 &
echo "TRAIN_PID $!"
sleep 35
echo "--- 启动日志 ---"
grep -E "落盘数据集|SkelUNet|step |Traceback|Error|FATAL" /tmp/pix64_v2.log | tail -6
echo "--- GPU ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
