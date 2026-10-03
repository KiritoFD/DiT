#!/bin/bash
# v15 版 FEW-SHOT：在 **v15a @150k**（87 个「书家x书体」pair，每对 4 子风格 x 384 维）上，
# 新增一个 pair 并只训该行，看能否 few-shot 捕获新书家。
#
# ⚠ 与之前 v13 版的关键差别:
#   - 基模: v13 base 155k(45书家 x 128维**单向量**)  ->  v15a 150k(87 pair x 4子风格 x 384维)
#   - 容量: 128 维           ->  1536 维 (12 倍)
#   - 书体: 之前用了草书(训练数据里根本没有草书 -> 分布外，实验被混淆)
#           现在只用 行(3)/楷(0)/隶(4) 三种在分布内的书体
#
# 书家: 伊秉绶-隶 / 沈周-行 / 徐渭-行 / 宋高宗-楷
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
PY=/opt/conda/envs/cu121/bin/python
LOGD=/root/Workspace/xy/DiT/logs/v15_series/v15_fewshot
mkdir -p "$LOGD"
TS=$(date +%Y%m%d-%H%M%S)

CK=$(ls -t /root/Workspace/xy/DiT/assets/results/v15a_multistyle_k4/*/checkpoints/0150000.pt | head -1)
if [ -z "$CK" ]; then echo "[v15fs] 找不到 v15a 150k ckpt"; exit 1; fi
echo "[v15fs] base ckpt = $CK"

port=29800
for c in 伊秉绶 沈周 徐渭 宋高宗; do
  CFG="src/train/configs/v15_fs_${c}.json"
  [ -f "$CFG" ] || { echo "[v15fs] 缺 $CFG"; continue; }

  echo "[v15fs] ===== $c BASELINE (不训练) $(date) ====="
  LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=$port \
  $PY -u src/train/train.py --config "$CFG" \
      --resume-full "$CK" --eval-only \
      --results-dir /tmp/_v15fs_${c}_base 2>&1 | tee "$LOGD/${c}_base_$TS.log" \
      | grep -E "eval-only|callig-emb|set=fewshot|Traceback|Error" | tail -6

  echo "[v15fs] ===== $c TRAIN K=50 $(date) ====="
  port=$((port+1))
  LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=$port \
  $PY -u src/train/train.py --config "$CFG" \
      --resume-full "$CK" --train-only-new-callig --init-new-callig mean_scaled \
      --results-dir assets/results/v15_fs_${c} 2>&1 | tee "$LOGD/${c}_train_$TS.log" \
      | grep -E "train-only-new-callig|callig-emb|set=fewshot|Traceback|Error" | tail -8
  port=$((port+1))
done

echo "[v15fs] ===== ALL DONE $(date) ====="
echo "  --- 基线 ---"
grep -h "set=fewshot" "$LOGD"/*_base_$TS.log 2>/dev/null | sed 's/^/  /'
echo "  --- 训练后 ---"
grep -h "set=fewshot" "$LOGD"/*_train_$TS.log 2>/dev/null | sed 's/^/  /'
