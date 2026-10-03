#!/bin/bash
# launch_v10a.sh — v10a 干净拉起 (幂等: 杀旧→清目录→tmux 起训练→自检)
cd /root/Workspace/xy/DiT || exit 1
tmux kill-session -t v10a 2>/dev/null
pkill -f 'train.py --config src/train/configs/v10a' 2>/dev/null
sleep 3
rm -rf assets/results/v10a_skel_cond_pretrain/2026090*
tmux new-session -d -s v10a 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export PYTHONPATH=/root/Workspace/xy/DiT; /opt/conda/envs/cu121/bin/python -u src/train/train.py --config src/train/configs/v10a_skel_cond_pretrain.json 2>&1 | tee /tmp/v10a_pretrain.log'
sleep 3
if tmux has-session -t v10a 2>/dev/null; then echo V10A_TMUX_OK; else echo V10A_TMUX_FAIL; exit 1; fi
grep -m1 -E "Building 2-Cond" /tmp/v10a_pretrain.log || echo "log pending"
