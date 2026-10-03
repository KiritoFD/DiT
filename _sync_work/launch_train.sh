#!/bin/bash
# 启动 64² 1px 骨架训练 + 排队宽度扫描
cd /root/Workspace/xy/DiT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 括号技巧: 不会匹配到本脚本自己的命令行
pkill -f 'sweep_[w]idth' 2>/dev/null
sleep 2

nohup /opt/conda/envs/cu121/bin/python -u tools/train_skelnet_pix.py \
    --prep skel --res 64 --steps 20000 --batch 64 --base 32 \
    --eval-every 1000 --val-n 64 > /tmp/pix64_train.log 2>&1 &
echo "TRAIN_PID $!"
sleep 4
nohup bash _sync_work/sweep_width.sh > /tmp/sweep_width.log 2>&1 &
echo "SWEEP_PID $!"
sleep 20
echo "--- 训练日志 ---"
grep -E "配对|decode|像素就绪|Traceback|Error" /tmp/pix64_train.log | tail -4
