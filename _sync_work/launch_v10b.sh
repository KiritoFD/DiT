#!/bin/bash
# launch_v10b.sh — v10b (no-char) 干净拉起 (幂等)
cd /root/Workspace/xy/DiT || exit 1
tmux kill-session -t v10b 2>/dev/null
pkill -f 'train.py --config src/train/configs/v10a' 2>/dev/null
sleep 3
rm -rf assets/results/v10b_skel_only_pretrain/2026090*
tmux new-session -d -s v10b 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export PYTHONPATH=/root/Workspace/xy/DiT; /opt/conda/envs/cu121/bin/python -u src/train/train.py --config src/train/configs/v10b_skel_only_pretrain.json 2>&1 | tee /tmp/v10b_pretrain.log'
sleep 3
# daemon 切到 v10b (v10a 已停, 它的曲线已回填)
tmux kill-session -t cpu_eval 2>/dev/null
tmux new-session -d -s cpu_eval 'export PYTHONPATH=/root/Workspace/xy/DiT; /opt/conda/envs/cu121/bin/python -u src/eval/cpu_eval_daemon.py --mode pretrain_g --watch-root /root/Workspace/xy/DiT/assets/results/v10b_skel_only_pretrain --threads 32 > /tmp/cpu_eval_daemon.log 2>&1'
sleep 3
if tmux has-session -t v10b 2>/dev/null; then echo V10B_TMUX_OK; else echo V10B_TMUX_FAIL; exit 1; fi
grep -m1 -E "Building 2-Cond" /tmp/v10b_pretrain.log || echo "log pending"
