#!/usr/bin/env bash
# 用新配置 (eval/存盘周期 5k) 重启 A/B 序列
set -u
cd /root/Workspace/xy/DiT || exit 1

echo "[1] 停掉在跑的 train.py 与 launcher"
pkill -f 'src/train/train.py' && echo "  train.py 已停" || echo "  (无 train.py 在跑)"
pkill -f 'launch_AB_100k.sh' && echo "  launcher 已停" || echo "  (无 launcher 在跑)"
sleep 8

echo "[2] GPU 现状 (应为空闲)"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

echo "[3] 旧 run 目录 (保留, 便于对照)"
ls -dt exp-std/runs_AB/*/ 2>/dev/null | head -3

echo "[4] 校验新配置的刻度"
grep -aE 'ckpt_every|epoch_steps|gpu_eval_every|max_steps' \
  src/train/configs/v50_A_space_xattn_style_adaLN_top10.json \
  src/train/configs/v51_B_joint_kv_k4_top10.json

echo "[5] 重新启动"
mkdir -p exp-std/logs_AB
nohup setsid bash _sync_work/launch_AB_100k.sh > exp-std/logs_AB/launch_r2.log 2>&1 < /dev/null &
echo "  launched (setsid, 断开 ssh 不影响)"
sleep 25
echo "[6] launcher 日志前 12 行"
head -12 exp-std/logs_AB/launch_r2.log
