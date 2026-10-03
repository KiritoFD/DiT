#!/bin/bash
# run_v10b_after_a.sh — v10a 正确早停后自动接 v10b (含 daemon 切换)
cd /root/Workspace/xy/DiT || exit 1
LOG=/tmp/v10b_waiter.log
echo "=== [v10b-waiter] $(date '+%F %T') 等待 v10a 早停 ===" >> $LOG
FREE=0
while [ $FREE -lt 2 ]; do
  sleep 120
  MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "${MEM:-99999}" -lt 2000 ]; then FREE=$((FREE+1)); else FREE=0; fi
done
sleep 60
echo "=== [v10b-waiter] $(date '+%F %T') v10a 已停, 拉起 v10b ===" >> $LOG
bash /root/Workspace/xy/DiT/_sync_work/launch_v10b.sh >> $LOG 2>&1
