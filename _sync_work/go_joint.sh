#!/bin/bash
# 停旧的联合训练 -> 用修好 I/O 的版本重起 (5000 步, 每 1000 步 eval)
set -u
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0

tmux kill-session -t joint 2>/dev/null
pkill -f 'train_joint_stage1_stage2' 2>/dev/null
sleep 6
nvidia-smi --query-gpu=memory.used --format=csv,noheader

S1=$(ls -1 assets/results/v33_stage1_xs/*/checkpoints/*.pt | grep -v done | sort | tail -1)
B2=$(ls -1 assets/results/v32_stage2_img/*/checkpoints/0080000.pt | head -1)
echo "stage1 = $S1"; echo "stage2 = $B2"

tmux new-session -d -s joint "cd /root/Workspace/xy/DiT && PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY -u tools/train_joint_stage1_stage2.py --gen-ckpt '$S1' --bak-ckpt '$B2' --train-bak 0 --lam-skel 0.05 --batch 24 --gen-steps 8 --lr 1e-5 --max-steps 5000 --eval-every 1000 --ckpt-every 500 --out-dir assets/results/v34_joint 2>&1 | tee logs/v34_joint.log"
sleep 3
tmux ls
echo GO_JOINT2_DONE
