#!/bin/bash
# 探针: 验证"消灭显存空洞"后 batch 360 能否装下。
#
# 背景: xattn @ batch240 的活跃需求 14.42G，但 torch.compile 把高水位顶到 20.51G
#   （空洞 6.09G）。按每样本 61.5MB 算，batch360 的活跃需求只有 22.2G（< 24.5G 容量），
#   装不下的只是高水位。
#
# 本探针跑 200 步（约 35 秒 + 编译 ~3 分钟），只看两件事:
#   ① 有没有 OOM
#   ② [alloc] 那行打出的高水位是多少
#
# 通过标准: 无 OOM 且高水位 < 23G。
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
# train.py 里已 setdefault expandable_segments；这里显式写一遍以示可覆盖
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/conda/envs/cu121/bin/python

echo "[probe360] START $(date)"
echo "[probe360] ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

$PY -u src/train/train.py \
    --config src/train/configs/v12_xattn_pretrain.json \
    --global-batch-size 360 \
    --max-steps 200 \
    --ckpt-every 0 \
    --in-mem-eval false \
    --results-dir /tmp/_probe360 \
    2>&1 | tee /root/Workspace/xy/DiT/logs/_probe360.log

echo "[probe360] rc=$? END $(date)"
echo "=== 关键行 ==="
grep -E "\[alloc\]|Mem:|OutOfMemory|Error" /root/Workspace/xy/DiT/logs/_probe360.log | tail -12
