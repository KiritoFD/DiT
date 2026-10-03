#!/bin/bash
# run_v10a_skel_cond.sh — v10a (skel latent 即字条件, 从头预训练) 拉起守护.
# 等 v9c 结束 (GPU 空闲) 后自动点火. 用法:
#   tmux new-session -d -s v10a_waiter 'bash /root/Workspace/xy/DiT/_sync_work/run_v10a_skel_cond.sh'
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT${PYTHONPATH:+:$PYTHONPATH}
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
PY=/opt/conda/envs/cu121/bin/python
LOG=/tmp/v10a_pretrain.log

echo "=== [v10a-waiter] $(date '+%F %T') 启动, 等待 GPU 空闲 (v9c 收敛) ===" >> $LOG
FREE=0
while [ $FREE -lt 2 ]; do
  sleep 120
  MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "${MEM:-99999}" -lt 2000 ]; then FREE=$((FREE+1)); else FREE=0; fi
done
sleep 60
echo "=== [v10a] $(date '+%F %T') GPU 空闲, 点火 (从头预训练, 预计 ~14h/131k 步) ===" >> $LOG

$PY -u src/train/train.py \
    --config src/train/configs/v10a_skel_cond_pretrain.json \
    >> $LOG 2>&1
RC=$?
echo "=== [v10a] rc=$RC $(date '+%F %T') ===" >> $LOG
[ $RC -ne 0 ] && tail -25 $LOG
