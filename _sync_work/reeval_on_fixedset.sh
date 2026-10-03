#!/bin/bash
# 用**修复版评测集**回溯评估历史 ckpt —— 解决"新 run 在修复集、E0/E1 在旧集"的口径断裂。
#
# 做法: 造一个只含软链的 results 目录, 让 batch_eval 的 glob 能找到目标 ckpt。
#   batch_eval 的 ckpt 来源 = <results_dir>/*/checkpoints/[0-9]*.pt
#
# ⚠ 必须显式 --skel-shards: E0/E1 的 ckpt args 里存的是未修的 data/50k/shards_std,
#   不覆盖的话骨架条件仍是旧骨架 -> 换评测集等于白换。
#
# 用法: bash _sync_work/reeval_on_fixedset.sh [E0|E1|INJ3_25K]
set -u
cd /root/Workspace/xy/DiT || exit 1
export CUDA_VISIBLE_DEVICES=0
PY=/opt/conda/envs/cu121/bin/python

WHICH="${1:-E0}"
SKEL=data/50k/shards_std_fixed
# ⚠ --sets 是 nargs="+", 每项格式为 name:csv:n —— 不能用逗号拼成一个字符串
SET_STRICT="strict:assets/eval_v13_strict_fixed.csv:249"
SET_SEEN="seen:assets/eval_v13_seen_fixed.csv:20"

case "$WHICH" in
  E0)
    SRC=$(ls -d assets/results/v17_s2_s2z_baseline/20260923-011335-*/checkpoints | head -1)
    OUT=assets/results/_e0_on_fixedset ;;
  E1)
    SRC=$(ls -d assets/results/v17_s2z_ada1_ln/20260923-083730-*/checkpoints | head -1)
    OUT=assets/results/_e1_on_fixedset ;;
  INJ3_25K)
    SRC=assets/results/v17_inj3_100k/20260924-101345-v17-inj3-100k/checkpoints
    OUT=assets/results/_inj3_on_fixedset ;;
  *) echo "未知目标: $WHICH"; exit 1 ;;
esac

[ -d "$SRC" ] || { echo "✗ 找不到 ckpt 目录: $SRC"; exit 1; }
mkdir -p "$OUT/_link/checkpoints"
# 只链 5000 的整数倍, 跳过 .done 与其它杂项
for f in "$SRC"/[0-9]*.pt; do
  [ -e "$f" ] || continue
  ln -sf "$(readlink -f "$f")" "$OUT/_link/checkpoints/$(basename "$f")"
done
echo "=== $WHICH: $(ls "$OUT/_link/checkpoints" | wc -l) 个 ckpt ==="
ls "$OUT/_link/checkpoints" | head -4

echo "=== $(date '+%F %T') batch_eval on 修复评测集 ==="
exec $PY -u -m src.eval.batch_eval \
    --results-dir "$OUT" \
    --sets "$SET_STRICT" "$SET_SEEN" \
    --skel-shards "$SKEL" \
    --cfg 0.7 --steps 50 \
    --force
