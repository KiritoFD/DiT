#!/bin/bash
# v13 流水线**冒烟测试**：不跑长训练，只验证
#   ① base 4ch 能训 + eval 能跑通（旧 eval 集曾在这里崩）
#   ② 12ch 后训练能起（--expand-from-4ch 接线）+ eval 能跑通
#
# 手法：--max-steps 6 --ckpt-every 3 --epoch-steps 3
#   -> 在 step 3 就触发一次 ckpt + eval（ckpt_every 与 epoch_steps 必须相等，
#      见 train.py 里的合并刻度说明）
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
PY=/opt/conda/envs/cu121/bin/python
LOGD=/root/Workspace/xy/DiT/logs/_smoke_v13
mkdir -p "$LOGD"

BASE_CKPT=$(ls -t /root/Workspace/xy/DiT/assets/results/v13_base_50k/*/checkpoints/*.pt 2>/dev/null | head -1)
echo "[smoke] base ckpt = $BASE_CKPT"

echo
echo "=========== 测试 1/2: base 4ch + eval ==========="
$PY -u src/train/train.py --config src/train/configs/v13_base_50k.json \
    --max-steps 6 --ckpt-every 3 --epoch-steps 3 \
    --results-dir /tmp/_smoke_base \
    2>&1 | tee "$LOGD/base.log" | grep -E "in-mem-eval|\[alloc\]|Error|Traceback|\(step=|assert" | tail -25
echo "[smoke] 测试 1 rc=${PIPESTATUS[0]}"

echo
echo "=========== 测试 2/2: 12ch 后训练 + eval ==========="
$PY -u src/train/train.py --config src/train/configs/v13_12ch_post.json \
    --expand-from-4ch "$BASE_CKPT" \
    --max-steps 6 --ckpt-every 3 --epoch-steps 3 \
    --results-dir /tmp/_smoke_12ch \
    2>&1 | tee "$LOGD/12ch.log" | grep -E "expand-4ch|in-mem-eval|channels|Error|Traceback|\(step=|assert" | tail -25
echo "[smoke] 测试 2 rc=${PIPESTATUS[0]}"

echo
echo "=========== 结果摘要 ==========="
for f in base 12ch; do
    echo "--- $f ---"
    grep -cE "in-mem-eval\] step .* done" "$LOGD/$f.log" 2>/dev/null | xargs echo "  eval 成功次数:"
    grep -E "in-mem-eval\] step .* (done|FAILED)" "$LOGD/$f.log" 2>/dev/null | tail -3 | cut -c1-170
    grep -E "Traceback|Error" "$LOGD/$f.log" 2>/dev/null | head -3
done
