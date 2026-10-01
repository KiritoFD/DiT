#!/bin/bash
# run_v31_v32_two_stage.sh —— 两阶段**分别**训练, 单脚本串行, 跑在 tmux 里。
#
#   stage 1 (v31_stage1_skel): std 骨架 + 风格 --> GT 骨架 latent   ← 替代 SkelNet
#   stage 2 (v32_stage2_img ): GT 骨架 + 风格 --> 真迹图像          ← v26 续训 (+整块抹白)
#
# 用仓库自己的训练管线 (src/train/train.py): flow / t_sampler / heun / EMA / CFG /
# 条件扰动 / 显存策略 全是已验证的那套, 才算与主干同口径。
#
# batch / 步数 (2026-10-01 实测, 目标显存 ~22G, 每阶段约 5h):
#   batch 128 -> 7.2G   13.2 steps/s
#   batch 256 -> 13.7G   6.6 steps/s
#   batch 384 -> 20.1G   4.4 steps/s   <- 采用
#   batch 448 -> OOM
#   80k 步 / 4.4 = 5.0h (stage1); stage2 由 30k 续到 80k = 50k/4.4 = 3.2h
#
# ⚠ 不要在 tmux 内层脚本里写 pkill -f "src.train.train" —— 内层脚本的命令行本身
#   就含这串字, pkill 会把自己杀掉, 表现为"会话瞬间消失、GPU 空" (2026-10-01 踩过)。
#
# 用法: bash run_v31_v32_two_stage.sh
#       tmux attach -t two_stage      (看进度)
set -u
cd /root/Workspace/xy/DiT
SESSION=two_stage

echo "=== 清场 (在外层做) ==="
pkill -f 'src\.train\.train' 2>/dev/null
sleep 6
nvidia-smi --query-gpu=memory.used --format=csv,noheader

tmux kill-session -t "$SESSION" 2>/dev/null
tmux new-session -d -s "$SESSION" "bash /root/Workspace/xy/DiT/_sync_work/two_stage_inner.sh"
sleep 3
tmux ls
echo "已启动. 看进度: tmux attach -t $SESSION"
