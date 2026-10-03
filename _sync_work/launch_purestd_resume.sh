#!/usr/bin/env bash
# 续训 v47_purestd (纯 std 条件 / Sp2 + xattn×12) : 从 0025000.pt 接着跑到 80000。
# 与原跑**同一 config、同一条 cosine**:
#   - STAGE_STEPS=55000 -> 25000+55000 = 80000 (原定目标, 不是新开一轮)
#   - FIRST_LR=4.0e-5   -> = 原 cosine 在 step 25000 处的值(实测日志 LR 4.00e-05), 不跳变
#   - 不传 --fresh-scheduler (runner 也已明令禁用: 它会把步数计数器归零)
cd /root/Workspace/xy/DiT || exit 1

export CFG=src/train/configs/v47_purestd_xattn12_top10.json
export STAGES="1.0"                       # 纯 std (部署条件)
export STAGE_STEPS=55000
export FIRST_LR=4.0e-5
export BATCH=128
export RUNS=exp-std/runs_purestd
export LOGD=exp-std/logs_purestd
export INIT=exp-std/runs_purestd/20261003-163412-v46-std-adaln4-top10-p1.0/checkpoints/0025000.pt

mkdir -p "$LOGD"
TS=$(date +%Y%m%d-%H%M%S)
LOG="$LOGD/resume_${TS}.log"

echo "[launch] INIT=$INIT"
echo "[launch] 目标: 25000 -> 80000 | lr 从 $FIRST_LR 接 | log=$LOG"
nohup setsid bash _sync_work/run_stdmix_cascade.sh > "$LOG" 2>&1 &
PID=$!
echo "[launch] pid=$PID  已后台启动 (setsid, 断开 ssh 不影响)"
sleep 90
echo "---------------------------------------------"
echo "[check] 进程:"; ps -o pid,etime,cmd -p "$PID" 2>/dev/null || echo "  进程不在了!"
echo "[check] GPU:"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "[check] 日志尾:"; tail -8 "$LOG"
