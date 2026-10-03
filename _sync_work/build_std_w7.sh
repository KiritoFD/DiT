#!/bin/bash
# 先探针 (量宽度), 再全量造 shards_std_w7
cd /root/Workspace/xy/DiT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/conda/envs/cu121/bin/python
FILT='Warning|warn|pytree|pkg_resources|preload|skel-guard'

echo "======== 探针: 宽度对照 ========"
$PY -u tools/build_std_w7.py --probe --limit 1 2>&1 | grep -vE "$FILT" | tail -12

echo "======== 全量生成 shards_std_w7 (iters=3 -> 7px) ========"
$PY -u tools/build_std_w7.py --iters 3 --batch 16 2>&1 | grep -vE "$FILT" | tail -16
echo BUILD_STD_W7_DONE
