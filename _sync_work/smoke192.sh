#!/usr/bin/env bash
# 决定性问题: batch=192 到底放不放下?
#   [1] 从代码确认 "Mem: a/b" 的 a/b 到底是什么 (allocated? reserved? peak?)
#   [2] 停掉现训练, 用 batch=192 跑 60 步 (compile 开, 复现真实显存), 看:
#       - 首步 warmup 峰值 (最大风险点, OOM 就死在这)
#       - 稳态峰值
#   背景: 现在 batch=128 时 nvidia-smi 只报 13.7G, 而日志 high-water 报 19.58G。
#         若 19.58G 只是首步/REPA-lazy 的一次性峰值, 192 就有余量。
set -u
cd /root/Workspace/xy/DiT || exit 1
PY=/opt/conda/envs/cu121/bin/python
export PYTHONPATH=/root/Workspace/xy/DiT
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

echo "################ [1] 日志里 Mem: a/b 的真实语义 ################"
grep -n 'Mem: {.*}/{.*}\|Mem: \|memory_reserved\|max_memory\|memory_allocated' src/train/train.py | head -20

echo
echo "################ [2] 停掉现有训练 ################"
pkill -f 'src/train/train.py' && echo "  train.py 已停" || echo "  (无)"
pkill -f 'launch_AB_100k.sh' && echo "  launcher 已停" || echo "  (无)"
sleep 8
nvidia-smi --query-gpu=memory.used --format=csv,noheader

echo
echo "################ [3] batch=192 冒烟 (60 步, compile 开) ################"
LOG=exp-std/logs_smoke/smoke_A192.log
mkdir -p exp-std/logs_smoke
( $PY -u src/train/train.py \
    --config src/train/configs/v50_A_space_xattn_style_adaLN_top10.json \
    --experiment-name smoke_A192 --results-dir exp-std/runs_smoke \
    --global-batch-size 192 --skel-latent-shards-weights 1.0,0.0 \
    --max-steps 60 --lr 7.5e-5 > "$LOG" 2>&1 ) &
SMOKE=$!
# 边跑边采 GPU 显存, 记录最大值
MAX=0
while kill -0 $SMOKE 2>/dev/null; do
  V=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  V=${V:-0}
  if [ "$V" -gt "$MAX" ] 2>/dev/null; then MAX=$V; fi
  sleep 2
done
wait $SMOKE; RC=$?
echo "  冒烟退出码 rc=$RC"
echo "  ★ 实测 nvidia-smi 峰值 = ${MAX} MiB (卡 24564 MiB)"

echo
echo "  --- 日志 alloc / Mem 行 ---"
grep -aE 'alloc|Mem:|Steps/Sec|OutOfMemory|CUDA out of memory|Traceback' "$LOG" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | tail -14
echo
echo "  --- 若 OOM, 尾部 ---"
grep -a 'OutOfMemory\|out of memory' "$LOG" >/dev/null 2>&1 && tail -20 "$LOG"
