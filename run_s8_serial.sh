#!/bin/bash
# S8: 修复版结构损失 (use_struct_loss_v2 + struct_max_t=500 t门控)。
# 从 diffonly@195000 出发, struct 权重 20000 步线性爬升。
# 串行两个实验: A=b8 全体struct, B=b32 subset8。各自 early-stop 收敛后自动切换。
cd ~/Workspace/xy/DiT || exit 1
PY=/opt/conda/bin/python

run_exp () {
  local CFG=$1 RES=$2 TAG=$3
  echo "[$(date '+%F %T')] ===== START $TAG ====="
  $PY auto_eval_cpu.py --results-dir "$RES" --seen5-csv 5script/seen2_top6.csv \
      --workers 4 --worker-threads 8 --interval 20 > "cpu_eval_${TAG}.log" 2>&1 &
  local EPID=$!
  $PY src/train.py --config "$CFG" > "train_${TAG}.log" 2>&1
  local CODE=$?
  echo "[$(date '+%F %T')] ===== TRAIN $TAG EXITED code=$CODE ====="
  sleep 15
  kill $EPID 2>/dev/null
  wait $EPID 2>/dev/null
}

run_exp exp_s8_structv2_b8all.json   5script/results/s8_structv2_b8all   s8_structv2_b8all
run_exp exp_s8_structv2_b32sub8.json 5script/results/s8_structv2_b32sub8 s8_structv2_b32sub8
echo "[$(date '+%F %T')] ===== S8 ALL DONE ====="
