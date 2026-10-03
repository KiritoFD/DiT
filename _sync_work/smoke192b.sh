#!/usr/bin/env bash
# batch=192 + 梯度检查点 (use_checkpoint) 能否放下? 速度代价多大?
set -u
cd /root/Workspace/xy/DiT || exit 1
PY=/opt/conda/envs/cu121/bin/python
export PYTHONPATH=/root/Workspace/xy/DiT
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

echo "################ [0] 确认 CLI 有 --use-checkpoint ################"
grep -n 'use.checkpoint\|use_checkpoint' src/train/cli.py | head -5

echo
echo "################ [1] 确认 GPU 空闲 ################"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
pgrep -af 'train.py' | head -2 || echo "  (无 train.py)"

echo
echo "################ [2] 192 + checkpoint 冒烟 (60 步) ################"
LOG=exp-std/logs_smoke/smoke_A192ckpt.log
mkdir -p exp-std/logs_smoke
( $PY -u src/train/train.py \
    --config src/train/configs/v50_A_space_xattn_style_adaLN_top10.json \
    --experiment-name smoke_A192ckpt --results-dir exp-std/runs_smoke \
    --global-batch-size 192 --skel-latent-shards-weights 1.0,0.0 \
    --max-steps 60 --lr 7.5e-5 --use-checkpoint true > "$LOG" 2>&1 ) &
SMOKE=$!
MAX=0
while kill -0 $SMOKE 2>/dev/null; do
  V=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  V=${V:-0}
  if [ "$V" -gt "$MAX" ] 2>/dev/null; then MAX=$V; fi
  sleep 2
done
wait $SMOKE; RC=$?
echo "  rc=$RC   ★ nvidia-smi 峰值 = ${MAX} MiB / 24564 MiB"

echo
echo "  --- alloc / Mem / 步速 ---"
grep -aE 'alloc|Mem:|Steps/Sec|out of memory|Traceback' "$LOG" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | tail -12

echo
echo "################ [3] 对照: 之前 128 无 checkpoint 的步速 ################"
grep -a 'Steps/Sec' exp-std/logs_AB/A_20261003-212648.log 2>/dev/null | tail -2 | sed 's/\x1b\[[0-9;]*m//g'
