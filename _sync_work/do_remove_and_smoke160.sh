#!/usr/bin/env bash
# 1) 应用"彻底移除梯度检查点"补丁  2) 验证  3) batch=160 冒烟(带峰值采样)
set -u
cd /root/Workspace/xy/DiT || exit 1
PY=/opt/conda/envs/cu121/bin/python
export PYTHONPATH=/root/Workspace/xy/DiT
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

echo "################ [1] 应用移除补丁 ################"
$PY -u _sync_work/remove_ckpt_patch.py 2>&1 | tail -30

echo
echo "################ [2] 验证: 全仓不该再有活引用 ################"
echo "--- dit.py 里 use_checkpoint / checkpoint( ---"
grep -n 'use_checkpoint\|grad_ckpt' src/model/dit.py || echo "  (干净 ✓)"
echo "--- train.py ---"
grep -n 'use_checkpoint\|grad_ckpt' src/train/train.py || echo "  (干净 ✓)"
echo "--- eval/model_io.py / cli.py ---"
grep -n 'use_checkpoint' src/eval/model_io.py src/train/cli.py || echo "  (干净 ✓)"
echo "--- 非 legacy 残留 ---"
grep -rn 'use_checkpoint' src/ --include=*.py | grep -v '/legacy/' || echo "  (干净 ✓)"
echo "--- dit.py 里 block 循环应只剩一个 (else 分支的循环被提上来) ---"
grep -n 'for i, block in enumerate(self.blocks)' src/model/dit.py

echo
echo "################ [3] batch=160 冒烟 (60 步, compile 开) ################"
LOG=exp-std/logs_smoke/smoke_A160.log
mkdir -p exp-std/logs_smoke
nvidia-smi --query-gpu=memory.used --format=csv,noheader
( $PY -u src/train/train.py \
    --config src/train/configs/v50_A_space_xattn_style_adaLN_top10.json \
    --experiment-name smoke_A160 --results-dir exp-std/runs_smoke \
    --global-batch-size 160 --skel-latent-shards-weights 1.0,0.0 \
    --max-steps 60 --lr 5e-5 > "$LOG" 2>&1 ) &
SMOKE=$!
MAX=0
while kill -0 $SMOKE 2>/dev/null; do
  V=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  V=${V:-0}
  if [ "$V" -gt "$MAX" ] 2>/dev/null; then MAX=$V; fi
  sleep 1
done
wait $SMOKE
echo "  ★ nvidia-smi 峰值 = ${MAX} MiB / 24564 MiB"
echo "  --- 日志 ---"
grep -aE 'alloc|Mem:|Steps/Sec|out of memory|Traceback|Done' "$LOG" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | tail -12
if grep -aq 'OutOfMemoryError' "$LOG"; then
  echo "  ✗✗ 仍然 OOM"
else
  echo "  ✓ 无 OOM"
fi
