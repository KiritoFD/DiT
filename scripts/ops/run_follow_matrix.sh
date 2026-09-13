#!/bin/bash
# run_follow_matrix.sh — GPU 系统性 skel 遵循矩阵 (n=30, 同字同seed)
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
RES=5script/results/skel_follow_gpu
mkdir -p $RES

run() { # model ckpt tag
  local m=$1 c=$2 t=$3
  echo "===== $m $t ($(date '+%H:%M:%S')) ====="
  $PY tools/eval/skel_follow_gpu.py --model $m --ckpt "$c" --tag $t --n 30 \
    > $RES/${t}.log 2>&1
  local s=$(grep -m1 "summary" $RES/${t}.log 2>/dev/null)
  echo "[$m $t] $s"
}

echo "===== v10b 遵循度曲线 (无char, 多训练点) ====="
for st in 67500 75000 82500 85000; do
  run v10b $st v10b_$st
done

echo "===== 对照模型 ====="
run v10a 127500 v10a_127500
run v8e "" v8e_22500

echo "===== ALL DONE ====="
echo "结果: $RES/*/metrics.json (n=30)"