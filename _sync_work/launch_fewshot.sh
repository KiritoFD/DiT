#!/bin/bash
# FEW-SHOT 新书家实验（怀素·草书）—— 等 v15 跑完后再执行
#
# 实验目的:
#   检验「书家条件 = 单向量能否捕获一个新书家的风格」。
#   背景: 实测 r(训练样本数, strict) = -0.62，样本多的书家风格跨度大，
#         而书家条件是**一个预训练且冻结的 128 维向量**。
#
# 流程:
#   0) 基线: 不训练，新行=已有 45 个的均值，直接评 eval -> 得到 ssim_base
#   1) K=10: 冻结主干，只训书家表新增行 [45:46)，评 eval -> ssim_k10
#   2) 对比 ssim_k10 vs ssim_base
#
# 判据:
#   ssim_k10 明显高于 ssim_base  -> 单向量能捕获新书家 -> 该上多 style token
#   两者差不多                    -> 单向量装不下风格，多 style token 才对症
#
# ⚠ 必须等 GPU 空闲（v15a/v15b 跑完）
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
PY=/opt/conda/envs/cu121/bin/python
LOGD=/root/Workspace/xy/DiT/logs/v13_series/v13_fewshot
mkdir -p "$LOGD"
TS=$(date +%Y%m%d-%H%M%S)

CK=$(ls -t /root/Workspace/xy/DiT/assets/results/v13_base_50k/*/checkpoints/0155000.pt | head -1)
if [ -z "$CK" ]; then echo "[fewshot] 找不到 base 155k ckpt"; exit 1; fi
echo "[fewshot] base ckpt = $CK"

# ── 0) 基线：不训练（新行=均值），直接评 eval ─────────────────────────
echo "[fewshot] ===== 0) BASELINE (eval-only, 不训练) $(date) ====="
LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=29650 \
$PY -u src/train/train.py --config src/train/configs/v13_fewshot_k10.json \
    --resume-full "$CK" --eval-only \
    --results-dir /tmp/_fewshot_base 2>&1 | tee "$LOGD/baseline_$TS.log" \
    | grep -E "eval-only|set=fewshot|Traceback|Error" | tail -6

# ── 1) K=10 训练：冻结主干，只训新增行 ───────────────────────────────
echo "[fewshot] ===== 1) TRAIN K=10 $(date) ====="
LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=29651 \
$PY -u src/train/train.py --config src/train/configs/v13_fewshot_k10.json \
    --resume-full "$CK" --train-only-new-callig --init-new-callig mean \
    --results-dir assets/results/v13_fewshot_k10 2>&1 | tee "$LOGD/train_$TS.log"

echo "[fewshot] ===== DONE rc=$? $(date) ====="
echo "[fewshot] 对比: baseline_$TS.log 的 eval ssim  vs  train_$TS.log 的 eval ssim"
