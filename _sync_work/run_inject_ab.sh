#!/usr/bin/env bash
# 四臂注入方式 A/B: 16ch Flux VAE / 从零 / 各 30k 步 / 同判分( eval200fix 187 + seen 20 )
cd /root/Workspace/xy/DiT || exit 1
mkdir -p exp-std/logs_inject
LOG="exp-std/logs_inject/ab_$(date +%Y%m%d-%H%M%S).log"
exec > "$LOG" 2>&1
ln -sf "$(basename "$LOG")" exp-std/logs_inject/ab_latest.log
echo "logfile=$LOG start=$(date '+%F %T')"
export PYTHONPATH=/root/Workspace/xy/DiT
export HF_HUB_OFFLINE=1
PY=/opt/conda/envs/cu121/bin/python
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

for A in A_adaln B_xattn C_every_layer D_k4_styleca; do
  echo
  echo "##################################################################"
  echo "########## ARM ${A}   起跑 $(date '+%F %T') ##########"
  echo "##################################################################"
  $PY -u src/train/train.py --config "src/train/configs/v48_inject_${A}.json"
  RC=$?
  echo "########## ARM ${A} 结束 rc=$RC  $(date '+%F %T') ##########"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
  if [ "$RC" -ne 0 ]; then
    echo "[FATAL] ${A} 失败 rc=$RC -> 中止 (前面的臂结果仍在各自 results 目录)"
    break
  fi
done
echo
echo "########## 汇总: 各臂 eval200fix 曲线 ##########"
for A in A_adaln B_xattn C_every_layer D_k4_styleca; do
  echo "--- ${A}"
  find "exp-std/runs_inject" -maxdepth 3 -name 'eval_stdskel_summary.csv' -path "*${A}*" 2>/dev/null \
    | while read -r f; do echo "   $f"; cat "$f"; done
done
echo "[ab] 全部结束 $(date '+%F %T')"
