#!/bin/bash
# launch_repa.sh — 启动 s32 REPA 训练 (tmux 内, 脱离 ssh)
# 基础: S30 base 0132500 + s31 ctrl 042500(best); diff+REPA 联合微调
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT${PYTHONPATH:+:$PYTHONPATH}

tmux kill-session -t s32repa 2>/dev/null
sleep 1
tmux new-session -d -s s32repa 'cd /root/Workspace/xy/DiT && /opt/conda/envs/cu121/bin/python src/train/train_repa.py \
    --config src/train/configs/s32_repa_finetune.json \
    --main-ckpt assets/results/s30_dino_char_strong_pretrain/20260901-052520-s30-dino-char-strong-pretrain/checkpoints/0132500.pt \
    --ctrl-ckpt assets/results/s31_ctrl_gt_skel_1px/20260901-135832-s31-ctrl-gt-skel-1px/checkpoints/0042500.pt \
    --batch-size 128 \
    --w-repa 0.1 \
    --repa-layer 8 \
    --compile true --compile-mode default \
    > /tmp/s32_repa_pipe.log 2>&1'
sleep 2
tmux ls
echo "s32 REPA 已启动 (tmux s32repa), 日志 /tmp/s32_repa_pipe.log"