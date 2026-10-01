#!/usr/bin/env bash
# run_v34_stage2_mix25.sh — Stage 2 鲁棒重训 (75% GT骨架 + 25% Stage 1 v31 预测骨架动态混合 + GPU 几何形变增强)
cd /root/Workspace/xy/DiT
mkdir -p logs exp/v34_stage2_mix25

# 检查 predskel_v31 分片是否存在
if [ ! -f "data/top10_style23/shards_predskel_v31/shard_00007.npz" ]; then
    echo "⚠ 警告: shards_predskel_v31 尚未完全生成完毕，请等待生成任务结束！"
fi

tmux kill-session -t v34_mix25 2>/dev/null || true

tmux new-session -d -s v34_mix25 "cd /root/Workspace/xy/DiT && PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/cu121/bin/python -m src.train.train --config src/train/configs/v34_stage2_mix25.json 2>&1 | tee /root/Workspace/xy/DiT/logs/v34_stage2_mix25.log"
echo "tmux session 'v34_mix25' successfully launched!"
