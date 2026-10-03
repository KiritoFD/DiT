#!/bin/bash
# 出可读海报 + seen 集的"什么都不做"基线
cd /root/Workspace/xy/DiT || exit 1
D=exp-std/reeval/20261003-163412-v46-std-adaln4-top10-p1.0__0025000

echo "########## 0. eval 产物目录 ##########"
find "$D/eval_samples_ctrl" -maxdepth 2 -type d | head -8

echo
echo "########## 1. 可读海报 (eval200 / seen) ##########"
for S in eval200 seen; do
  PYTHONPATH=. /opt/conda/envs/cu121/bin/python -u tools/make_poster.py \
    --dir "$D" --set $S --cols 16 2>&1 | tail -3
done

echo
echo "########## 2. seen 集的 std 直出基线 ##########"
PYTHONPATH=. HF_HUB_OFFLINE=1 /opt/conda/envs/cu121/bin/python -u tools/std_baseline_eval.py \
  --set seen --csv exp-std/csv/seen20.csv --n 20 --shards exp-std/data/shards_std_w7 2>&1 \
  | grep -vE 'Warning|warn|pkg_resources|FutureWarning|_torch_pytree' | tail -12

echo
echo "########## 3. 汇总对照 ##########"
echo "--- 基线 ---"; cat exp-std/reeval/std_baseline_eval200.csv exp-std/reeval/std_baseline_seen.csv 2>/dev/null
echo "--- 模型 @25k (summary) ---"
cat "$D/eval_stdskel_summary.csv" 2>/dev/null
