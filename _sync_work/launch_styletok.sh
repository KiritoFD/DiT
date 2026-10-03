#!/bin/bash
# 受控实验：冻结主干，只训风格模块（callig_style_ca + style_role）
#
# 起点: v13 base 155k ckpt（已训好）
# 变体: v13_styletok.json = base + style_token_n=32 + glyph_inject_mode=xattn
# 冻结: --train-only-style -> 12,288 / 39,336,209 参数可训 (0.03%)
# 关键: out_proj 是 zero-init -> step0 输出**恒等于** base 155k
#       -> 之后任何 strict 变化都可直接归因于新增的风格容量
#
# 目的：验证「书家条件是单向量 -> 装不下多模态风格」这个假说
#       （实测 r(训练样本数, strict) = -0.62，样本越多反而越差）
#
# ⚠ 必须先停掉其它训练：本实验要走满 batch，与 wd01 抢显存会 OOM
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
PY=/opt/conda/envs/cu121/bin/python
LOGD=/root/Workspace/xy/DiT/logs/v13_series/v13_styletok
mkdir -p "$LOGD"
TS=$(date +%Y%m%d-%H%M%S)

CK=$(ls -t /root/Workspace/xy/DiT/assets/results/v13_base_50k/*/checkpoints/0155000.pt | head -1)
if [ -z "$CK" ]; then echo "[styletok] 找不到 base 155k ckpt"; exit 1; fi

echo "[styletok] ===== START $(date) ====="
echo "[styletok] ckpt  = $CK"
echo "[styletok] 只训风格模块（冻结主干）"

LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=29640 \
$PY -u src/train/train.py --config src/train/configs/v13_styletok.json \
    --resume-full "$CK" \
    --train-only-style \
    --results-dir assets/results/v13_styletok \
    2>&1 | tee "$LOGD/train_$TS.log"

echo "[styletok] ===== END rc=$? $(date) ====="
