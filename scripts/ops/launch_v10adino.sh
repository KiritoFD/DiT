#!/bin/bash
# launch_v10adino.sh — v10a-dino (冻结 DINO char 表) 干净拉起 (幂等)
# v10a 配方唯一变量: char 因子 LabelEmbedder随机表 -> StdDinoCharEmbedder冻结表 + ln_only 直通
cd /root/Workspace/xy/DiT || exit 1
tmux kill-session -t v10adino 2>/dev/null
pkill -f 'train.py --config src/train/configs/v10adino' 2>/dev/null
sleep 3
rm -rf assets/results/v10adino_skel_cond_pretrain/20260*
tmux new-session -d -s v10adino 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export PYTHONPATH=/root/Workspace/xy/DiT; /opt/conda/envs/cu121/bin/python -u src/train/train.py --config src/train/configs/v10adino_skel_cond_pretrain.json 2>&1 | tee /tmp/v10adino_pretrain.log'
sleep 3
# daemon 切到 v10adino (v10b 已停, 曲线已入册)
tmux kill-session -t cpu_eval 2>/dev/null
tmux new-session -d -s cpu_eval 'export PYTHONPATH=/root/Workspace/xy/DiT; /opt/conda/envs/cu121/bin/python -u src/eval/legacy/cpu_eval_daemon.py --mode pretrain_g --watch-root /root/Workspace/xy/DiT/assets/results/v10adino_skel_cond_pretrain --threads 32 > /tmp/cpu_eval_daemon.log 2>&1'
sleep 3
if tmux has-session -t v10adino 2>/dev/null; then echo V10ADINO_TMUX_OK; else echo V10ADINO_TMUX_FAIL; exit 1; fi
grep -m1 -E "Building 2-Cond|StdDino|std-dino" /tmp/v10adino_pretrain.log || echo "log pending"