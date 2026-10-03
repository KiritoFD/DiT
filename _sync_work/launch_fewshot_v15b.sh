#!/bin/bash
# FEW-SHOT（v15 多风格版）—— 等 v15b+SupCon 训完后一键跑
#
# 用法:  bash _sync_work/launch_fewshot_v15b.sh [base_ckpt]
#   不传参: 自动取 v15b_supcon 目录里最新的 ckpt
#
# 设计要点（都是踩过的坑）:
#   - 基模: v15b + SupCon 表（87 pair x 4 子风格 x 384 维，行间余弦 0.0019）
#   - 新书家 = 新的「书家x书体」pair（v15 的条件粒度就是 pair）
#   - 书体只用 行(3)/楷(0)/隶(4) —— 训练数据里**没有草书**，用草书会分布外（实测踩过）
#   - 只训新增行 [87:88)（1536 参数），主干全冻
#   - use_ema=False: 否则评测用 EMA，短跑只反映 ~4% 的更新（实测踩过）
#   - 基线(--eval-only)用 ckpt 的 ema、训练跑用 raw —— **两者不可直接比**，
#     只有"训练内部的多次 eval"是可信的对照
#   - max_steps = ckpt步数 + 200，否则 resume 后立刻"Reached max_steps"（实测踩过）
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
PY=/opt/conda/envs/cu121/bin/python
LOGD=/root/Workspace/xy/DiT/logs/v15_series/v15_fewshot
mkdir -p "$LOGD"
TS=$(date +%Y%m%d-%H%M%S)

CK="${1:-}"
if [ -z "$CK" ]; then
    CK=$(ls -t /root/Workspace/xy/DiT/assets/results/v15b_supcon/*/checkpoints/*.pt 2>/dev/null | head -1)
fi
if [ -z "$CK" ] || [ ! -f "$CK" ]; then
    echo "[fs] 找不到基模 ckpt（v15b_supcon 还没训出 ckpt？）"
    exit 1
fi
echo "[fs] base ckpt = $CK"

# 基模的步数（决定 max_steps 起点）
BASE_STEP=$(basename "$CK" .pt | sed 's/^0*//')
echo "[fs] base step = $BASE_STEP"

port=29920
for c in 伊秉绶 沈周 徐渭 宋高宗; do
  CFG="src/train/configs/v15_fs_${c}.json"
  [ -f "$CFG" ] || { echo "[fs] 缺 $CFG"; continue; }

  echo "[fs] ===== $c BASELINE (不训练) $(date) ====="
  port=$((port+1))
  LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=$port \
  $PY -u src/train/train.py --config "$CFG" \
      --resume-full "$CK" --eval-only \
      --results-dir /tmp/_fs_${c}_base 2>&1 | tee "$LOGD/${c}_base_$TS.log" \
      | grep -E "eval-only|callig-emb|set=fewshot|Traceback|Error" | tail -6

  echo "[fs] ===== $c TRAIN K=50 $(date) ====="
  port=$((port+1))
  LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=$port \
  $PY -u src/train/train.py --config "$CFG" \
      --resume-full "$CK" --train-only-new-callig --init-new-callig mean_scaled \
      --results-dir assets/results/v15_fs_${c} 2>&1 | tee "$LOGD/${c}_train_$TS.log" \
      | grep -E "train-only-new-callig|callig-emb|set=fewshot|Reached|Traceback|Error" | tail -8
done

echo "[fs] ===== ALL DONE $(date) ====="
echo "  --- 基线 ---"
grep -h "set=fewshot" "$LOGD"/*_base_$TS.log 2>/dev/null | sed 's/^/  /'
echo "  --- 训练后 ---"
grep -h "set=fewshot" "$LOGD"/*_train_$TS.log 2>/dev/null | sed 's/^/  /'
