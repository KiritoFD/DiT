#!/bin/bash
# launch_s32b.sh — 启动 s32b 强 REPA (w=0.3, L2 8&11, eval cfg=0.7), tmux
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT${PYTHONPATH:+:$PYTHONPATH}

tmux kill-session -t s32b 2>/dev/null
sleep 1
tmux new-session -d -s s32b 'cd /root/Workspace/xy/DiT && /opt/conda/envs/cu121/bin/python src/train/train_repa.py \
    --config src/train/configs/s32b_repa_strong.json \
    --main-ckpt assets/results/s30_dino_char_strong_pretrain/20260901-052520-s30-dino-char-strong-pretrain/checkpoints/0132500.pt \
    --ctrl-ckpt assets/results/s31_ctrl_gt_skel_1px/20260901-135832-s31-ctrl-gt-skel-1px/checkpoints/0042500.pt \
    --w-repa 0.3 \
    --repa-layers "8,11" \
    --compile true --compile-mode default \
    > /tmp/s32b_repa_pipe.log 2>&1'
sleep 2
tmux ls
echo "s32b 强 REPA 已启动 (tmux s32b), 日志 /tmp/s32b_repa_pipe.log"