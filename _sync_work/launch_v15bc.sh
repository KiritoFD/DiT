#!/bin/bash
# v15 注入方式矩阵剩下的两格，单卡串行：
#   b) v15b（书家化骨架 CrossAttn）从最新 ckpt 续到 150k —— 约 2.5h（4 步/秒）
#   c) v15c（K 个 style token 进每层 xattn context）从零跑 150k —— 约 20h
# 为什么 v15b 用 --resume-full 而不是 fresh：train.py 默认保留 ckpt 的 train_steps
# 与 optimizer/EMA 状态，cosine 调度按**绝对步数**继续（115k 处 LR≈2e-5 接着往下走），
# 与 v15a/v13_base 的 150k 刻度才可比。加 --fresh-scheduler 反而会退回 warmup 起点。
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
PY=/opt/conda/envs/cu121/bin/python
LOGD=logs/v15_series
mkdir -p "$LOGD"

run() {   # run <日志前缀> <args...>
  local name=$1; shift
  local log="$LOGD/${name}_$(date +%Y%m%d-%H%M%S).log"
  echo "[v15bc] >>> $name -> $log  ($(date '+%m-%d %H:%M'))"
  LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=$PORT \
    $PY -u src/train/train.py "$@" > "$log" 2>&1
  echo "[v15bc] <<< $name rc=$?  ($(date '+%m-%d %H:%M'))  见 $log"
}

CKB=$(ls -t assets/results/v15b_multistyle_k4/*/checkpoints/*.pt 2>/dev/null | head -1)
if [ -z "$CKB" ]; then echo "[v15bc] ✗ 找不到 v15b ckpt"; exit 1; fi
echo "[v15bc] v15b 续训自 $CKB"
PORT=29950 run v15b_resume --config src/train/configs/v15b_multistyle_k4_ca.json \
     --resume-full "$CKB"

PORT=29951 run v15c_train --config src/train/configs/v15c_multistyle_k4_ctx.json
echo "[v15bc] ALL DONE $(date '+%m-%d %H:%M')"
