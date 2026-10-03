#!/bin/bash
# 每个实验取最新 ckpt，GPU 评 seen(20) + strict(249)
cd /root/Workspace/xy/DiT
export PYTHONPATH=/root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
OUT=/root/Workspace/xy/DiT/assets/full_eval.csv
echo "run,step,set,ssim,med,n" > $OUT
for d in assets/results/*/; do
  run=$(basename "$d")
  # 最新 ckpt（按数字排序取最大）
  ck=$(ls -v $d*/checkpoints/[0-9]*.pt 2>/dev/null | tail -1)
  [ -n "$ck" ] || continue
  step=$(basename "$ck" .pt)
  for spec in "seen:assets/eval_v13_seen_fixed.csv:20" "strict:assets/eval_v13_strict_fixed.csv:249"; do
    set="${spec%%:*}"; rest="${spec#*:}"; csv="${rest%:*}"; n="${rest##*:}"
    D=/tmp/_full_${run}_${set}; rm -rf $D; mkdir -p $D
    o=$(timeout 300 $PY -u tools/eval/eval_stdskel_batch.py --results-dir $D \
        --ckpt-override "$ck" --device cuda --sets "$set:$csv:$n" \
        --dit-batch 64 --vae-batch 32 2>&1)
    s=$(echo "$o" | grep -oE "ssim=[0-9.]+" | tail -1 | cut -d= -f2)
    m=$(echo "$o" | grep -oE "med=[0-9.]+" | tail -1 | cut -d= -f2)
    if [ -n "$s" ]; then
      echo "$run,$step,$set,$s,$m,$n" >> $OUT
      echo "  $run [$set] ssim=$s"
    else
      echo "$run,$step,$set,LOAD_FAIL,," >> $OUT
      echo "  $run [$set] LOAD_FAIL"
    fi
  done
done
echo "=== FULL EVAL DONE ==="
