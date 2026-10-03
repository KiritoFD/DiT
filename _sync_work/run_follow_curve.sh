#!/bin/bash
# run_follow_curve.sh — skel 遵循度随训练曲线: v10b 多 ckpt + v8e 对照
# 有说服力要点: 同 10 新字同 seed; v10b 采样 5 个训练点; v8e 对照; 输出汇总表
set -u
cd /root/Workspace/xy/DiT || exit 1
PY=/opt/conda/bin/python
OUT=assets/results/v10b_handwrite_test

echo "===== 遵循度随训练曲线 (同字同seed, 3px IoU) ====="
echo "model  step   iou3_mean  iou3_median"
run() {
  local model=$1 ckpt=$2 tag=$3
  local log=/tmp/follow_${model}_${tag}.log
  nice -n 10 $PY tools/eval/handwrite_follow_test.py --model $model --ckpt $ckpt --tag $tag --n 10 > $log 2>&1
  local line=$(grep "3px 容差 IoU" $log | head -1)
  echo "  $model  $tag  $line"
}

# v10b: 采样多个训练点 (12.5k / 30k / 50k / 72.5k / 最新 82.5k)
# 12.5k 已有 (文档 36, median 0.561); 其他各跑
for st in 50000 72500 82500; do
  run v10b $st $st
done

# v8e 对照 (两阶段, 文档 36 已测 median 0.194; 重跑确认同协议)
run v8e 0022500 v8e_22500

echo "===== 汇总完成 ====="
echo "日志: /tmp/follow_v10b_*.log, 图: $OUT/{step*/,v8e_22500/}"