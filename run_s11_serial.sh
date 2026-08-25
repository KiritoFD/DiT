#!/bin/bash
# 串行跑 s11 top6 三个 patch 实验: p4 -> p2 -> p8
# 每个实验: 启动训练(前台等待退出) + 重启 eval 指标 daemon 指向对应 results_dir。
# 训练结束(early_stop / max_steps)自然退出后接下一个。
set -u
cd "$(dirname "$0")"
PY=/opt/conda/bin/python

CONFIGS=(
  configs_s11/s11_top6_p4.json
  configs_s11/s11_top6_p2.json
  configs_s11/s11_top6_p8.json
)

for CFG in "${CONFIGS[@]}"; do
  EXP=$("$PY" -c "import json,sys; print(json.load(open('$CFG'))['experiment_name'])")
  RES=$("$PY" -c "import json,sys; print(json.load(open('$CFG'))['results_dir'])")
  LOG="logs_${EXP}.txt"
  EVAL_LOG="logs_evalmetrics_${EXP}.txt"

  echo "[serial] ==========================================================="
  echo "[serial] Launching $EXP  (config=$CFG)"
  echo "[serial] results_dir = $RES   log = $LOG"
  date

  # 重启指标 daemon 指向本实验的 results dir
  tmux kill-session -t evalmetrics 2>/dev/null
  sleep 1
  tmux new-session -d -s evalmetrics \
    "$PY eval_metrics_daemon.py $RES > $EVAL_LOG 2>&1"
  echo "[serial] evalmetrics daemon -> $RES"

  # 前台训练, 等待退出; 保留真实退出码
  "$PY" train.py --config "$CFG" 2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  echo "[serial] $EXP finished rc=$RC"
  date

  if [ "$RC" -ne 0 ]; then
    echo "[serial] WARN: $EXP exit code $RC (非0, 继续下一个实验)"
  fi
done

echo "[serial] ==========================================================="
echo "[serial] ALL THREE DONE: p4 p2 p8"