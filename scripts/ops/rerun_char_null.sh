#!/bin/bash
# rerun_char_null.sh — 修正 char 协议重跑: v10a/v8e 用 null-char (y_char=num_classes),
# v10b 对照验证不变。输出到 char_null 子目录, 不覆盖旧数据。
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
RES=5script/results/skel_follow_gpu

run() { # model ckpt tag
  local m=$1 c=$2 t=$3
  echo "===== $m $t (char-null 协议, $(date '+%H:%M:%S')) ====="
  $PY tools/eval/skel_follow_gpu.py --model $m --ckpt "$c" --tag $t --n 30 \
    > $RES/${t}.log 2>&1
  local s=$(grep -m1 "summary" $RES/${t}.log 2>/dev/null)
  echo "[$m $t] $s"
}

echo "##### char-null 协议矩阵 (v10a/v8e 修正, v10b 对照) #####"
run v10a 127500 v10a_127500_charnull
run v8e "" v8e_22500_charnull
run v10b 85000 v10b_85000_charnull
echo "===== ALL DONE ====="
echo "结果: $RES/*_charnull/metrics.json (n=30)"