#!/bin/bash
# 只评 v12+ 的实验（v11 用旧数据集 28,569/36 书家，与 50k eval 集不兼容）
cd /root/Workspace/xy/DiT
export PYTHONPATH=/root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
RAW=/root/Workspace/xy/DiT/assets/full_eval_raw
mkdir -p $RAW
OUT=/root/Workspace/xy/DiT/assets/full_eval.csv
echo "run,step,set,ssim,med,n" > $OUT
for d in assets/results/*/; do
  run=$(basename "$d")
  case "$run" in v11_*|v10b_*|v9*|v8*) continue;; esac
  ck=$(ls -v $d*/checkpoints/[0-9]*.pt 2>/dev/null | tail -1)
  [ -n "$ck" ] || continue
  step=$(basename "$ck" .pt)
  for spec in "seen:assets/eval_v13_seen_fixed.csv:20" "strict:assets/eval_v13_strict_fixed.csv:249"; do
    set="${spec%%:*}"; rest="${spec#*:}"; csv="${rest%:*}"; n="${rest##*:}"
    D=/tmp/_fe3_${run}_${set}; rm -rf $D; mkdir -p $D
    o=$(timeout 400 $PY -u tools/eval/eval_stdskel_batch.py --results-dir $D \
        --ckpt-override "$ck" --device cuda --sets "$set:$csv:$n" \
        --dit-batch 64 --vae-batch 32 2>&1)
    s=$(echo "$o" | grep -oE "ssim=[0-9.]+" | tail -1 | cut -d= -f2)
    m=$(echo "$o" | grep -oE "med=[0-9.]+" | tail -1 | cut -d= -f2)
    if [ -n "$s" ]; then
      echo "$run,$step,$set,$s,$m,$n" >> $OUT
      cp $D/eval_stdskel_batch.csv $RAW/${run}__${set}.csv 2>/dev/null
      echo "  $run [$set] ssim=$s"
    else
      echo "$run,$step,$set,FAIL,," >> $OUT
      echo "  $run [$set] FAIL"
    fi
  done
done
echo "=== DONE ==="
