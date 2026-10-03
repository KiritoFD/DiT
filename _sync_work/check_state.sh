#!/bin/bash
cd /root/Workspace/xy/DiT
echo "=== GPU / 进程 ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
pgrep -af 'train_skelnet' | head -3
echo
echo "=== 我这一轮的 w7/w3 下游成绩 (我的口径) ==="
for d in _calib_dit_w7raw _calib_dit_w7ren _calib_dit _calib_dit_renorm; do
  f=assets/results/$d/eval_stdskel_batch.csv
  [ -f "$f" ] && /opt/conda/envs/cu121/bin/python _sync_work/agg_batch.py $f
done
echo
echo "=== 关键产物是否在 ==="
ls -la assets/skelnet_dit_H_bridge_nog_w7.pt 2>/dev/null
ls -la assets/skelnet_dit_H_bridge_nog_w7_clean84.pt.best 2>/dev/null
ls -la assets/eval_v13_strict84_aligned.csv assets/train_top10_style23_minusval_clean84.csv 2>/dev/null
echo
echo "=== v30 相关目录 ==="
ls -d assets/results/v30_* 2>/dev/null
ls -d data/top10_style23/predskel_eval_strict84_* 2>/dev/null | head -5
