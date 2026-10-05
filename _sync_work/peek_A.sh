#!/usr/bin/env bash
# evalonly_v50a -> unique name
echo MARKER_evalonly_running
set -u
cd /root/Workspace/xy/DiT || exit 1
PY=/opt/conda/envs/cu121/bin/python

CKPT=exp-std/runs_AB/20261004-000014-v50-A-space-xattn-style-adaln-top10/checkpoints/0055000.pt
LOG=exp-std/logs_smoke/v50A_evalonly.log

PYTHONPATH=. HF_HUB_OFFLINE=1 $PY -u src/train/train.py \
  --config src/train/configs/v50_A_space_xattn_style_adaLN_top10.json \
  --resume-full "$CKPT" --eval-only \
  > "$LOG" 2>&1
RC=$?
echo "rc=$RC"
grep -aE 'eval-only|in-mem-eval|Traceback|Error' "$LOG" | sed 's/\x1b\[[0-9;]*m//g' | tail -10
echo
ls exp-std/runs_AB/eval_samples_ctrl/step0055000/eval200fix/ 2>/dev/null | head -4
ls exp-std/runs_AB/eval_samples_ctrl/step0055000/eval200fix/ 2>/dev/null | wc -l
nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used,clocks.current.sm --format=csv,noheader
echo
echo "=== A 路日志文件 ==="
LOG=$(ls -t exp-std/logs_AB/A_*.log 2>/dev/null | head -1)
echo "log=$LOG   大小=$(stat -c%s "$LOG" 2>/dev/null) 字节"
echo
echo "=== 关键里程碑 (带时间) ==="
grep -aE 'preload|Begin epoch|Reached|step=0000|alloc|Traceback|Error|Done' "$LOG" 2>/dev/null | tail -12
echo
echo "=== Steps/Sec (若有) ==="
grep -aE 'Steps/Sec|Step/Sec' "$LOG" 2>/dev/null | tail -5 || echo "(还没有步速行 -> 仍在编译/warmup)"
echo
echo "=== 最后 8 行原文 ==="
tail -8 "$LOG" 2>/dev/null
echo
echo "=== 该 run 目录 ==="
ls -t exp-std/runs_AB/ 2>/dev/null | head -3
echo
echo "=== 进程 & 线程状态 (llvm=编译中) ==="
PID=$(pgrep -f 'config src/train/configs/v50_A' | head -1)
echo "pid=$PID"
if [ -n "${PID:-}" ]; then
  ps -eLo pid,tid,pcpu,comm --sort=-pcpu | grep -E '^\s*'"$PID" | head -8
fi
