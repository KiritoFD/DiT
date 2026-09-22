#!/bin/bash
# 对所有「数据集匹配」的实验，跑 eval（含新墨迹指标）+ 把图收集到统一目录
#
# 统一目录结构:
#   assets/ink_eval/{run}__{step}__{set}/
#       g{i}.png   gt{i}.png
#   assets/ink_eval_summary.csv   逐样本（含 ink_ssim / ink_iou / skel_iou）
#
cd /root/Workspace/xy/DiT
export PYTHONPATH=/root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
COLLECT=/root/Workspace/xy/DiT/assets/ink_eval
SUM=/root/Workspace/xy/DiT/assets/ink_eval_summary.csv
RAW=/root/Workspace/xy/DiT/assets/ink_eval_raw
mkdir -p $COLLECT $RAW
rm -f $SUM
echo "run,step,set,idx,img_id,char,script,mse,ssim,lpips,ink_ssim,ink_iou,skel_iou" > $SUM

# 只跑「数据集匹配」的（v11/v12 用旧数据集 28,569，与 50k eval 不兼容）
for d in assets/results/*/; do
  run=$(basename "$d")
  case "$run" in v11_*|v10b_*|v12_*|_archive*|v9*|v8*) continue;; esac
  ck=$(ls -v $d*/checkpoints/[0-9]*.pt 2>/dev/null | tail -1)
  [ -n "$ck" ] || continue
  step=$(basename "$ck" .pt)
  for spec in "seen:assets/eval_v13_seen_fixed.csv:20" "strict:assets/eval_v13_strict_fixed.csv:249"; do
    set="${spec%%:*}"; rest="${spec#*:}"; csv="${rest%:*}"; n="${rest##*:}"
    D=/tmp/_ink_${run}_${set}; rm -rf $D; mkdir -p $D
    o=$(timeout 500 $PY -u tools/eval/eval_stdskel_batch.py --results-dir $D \
        --ckpt-override "$ck" --device cuda --sets "$set:$csv:$n" \
        --dit-batch 64 --vae-batch 32 --save-samples 2>&1)
    s=$(echo "$o" | grep -oE "ssim=[0-9.]+" | tail -1 | cut -d= -f2)
    if [ -n "$s" ]; then
      # 收集逐样本 CSV（含新指标）
      cp $D/eval_stdskel_batch.csv $RAW/${run}__${set}.csv 2>/dev/null
      # 收集图片
      src=$(find assets/results/$run -path "*eval_samples_ctrl*" -name "g0.png" -newermt "-5 minutes" 2>/dev/null | head -1)
      if [ -n "$src" ]; then
        dst=$COLLECT/${run}__${step}__${set}; mkdir -p $dst
        cp "$(dirname $src)"/g*.png "$(dirname $src)"/gt*.png $dst/ 2>/dev/null
      fi
      echo "  $run [$set] ssim=$s  图=$(ls $COLLECT/${run}__${step}__${set} 2>/dev/null | wc -l)"
    else
      echo "  $run [$set] FAIL"
    fi
  done
done
echo "=== DONE ==="
ls $COLLECT | wc -l | sed 's/^/  收集目录数: /'
