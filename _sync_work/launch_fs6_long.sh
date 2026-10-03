#!/bin/bash
# fs6 充分训练（用户协议 v3）：batch 384，**3 路并行**（1536 单跑会 OOM），
# lr 3e-4 constant，+50k 步，每 2500 步 eval。判读以 Diff 走平为准。
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
R=_sync_work/run_fs6.sh

wave() {
  for t in "$@"; do
    bash $R train "$t" row_pt 0.0003 50000 long &
    sleep 15                      # 错峰编译
  done
  wait
}

wave 沈周-行 伊秉绶-行 傅山-行
wave 伊秉绶-隶 徐渭-行

/opt/conda/envs/cu121/bin/python _sync_work/fs6_report.py > logs/_fs6_long_report.txt 2>&1
echo "[long] ALL DONE $(date '+%m-%d %H:%M')"
