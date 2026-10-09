#!/usr/bin/env bash
# run_vae_fix_arms.sh — VAE 修正训练（三臂，从现冻结版续训）
#
#   臂 A (collapse) : 只推后验坍缩（σ→0）      —— 去掉 mode/sample 歧义 = 去掉 15% 目标噪声
#   臂 B (hf)       : 只加高频保真损失(dog)     —— 补 L1/DINO 都看不见的纹理
#   臂 C (both)     : A + B（推荐组合）
#
# 用法:  bash tools/vae/run_vae_fix_arms.sh check     # 预检（不训练）
#        bash tools/vae/run_vae_fix_arms.sh run       # 串行跑三臂
#        STEPS=1500 ARMS="C" bash tools/vae/run_vae_fix_arms.sh run
set -eu
cd "$(dirname "$0")/../.." || exit 1
ROOT=$(pwd)
PY=/opt/conda/envs/cu121/bin/python
BASE=${BASE:-$ROOT/experiments/vae_frozen/calli_vae_kl1e6_s12500_stdconv}
DINO=$ROOT/data/pretrained/pretrained_models/dinov2_vits14_pretrain.safetensors
CSV=/root/Workspace/xy/UNIFIED_RAW/meta/train_clean.csv
DATA_ROOT=/root/Workspace/xy/UNIFIED_RAW
STEPS=${STEPS:-1500}
ARMS=${ARMS:-A B C}

check() {
  echo "== 预检 =="
  [ -d "$BASE" ] && echo "  base VAE   ✓ $BASE" || { echo "  ✗ base VAE 缺失"; return 1; }
  [ -f "$DINO" ] && echo "  DINO ckpt  ✓" || { echo "  ✗ DINO 缺失"; return 1; }
  [ -f "$CSV" ] && echo "  CSV        ✓ ($(wc -l < "$CSV") 行)" || { echo "  ✗ CSV 缺失"; return 1; }
  local img; img=$(head -2 "$CSV" | tail -1 | cut -d, -f2)
  [ -f "$DATA_ROOT/$img" ] && echo "  图片样本   ✓ $DATA_ROOT/$img" || { echo "  ✗ 图片路径拼接失败: $DATA_ROOT/$img"; return 1; }
  $PY -c "import ast;ast.parse(open('tools/vae/train_dino_calli_vae.py',encoding='utf-8').read());print('  脚本语法   ✓')"
  $PY -c "import sys;sys.path.insert(0,'tools/vae');import hf_losses;print('  hf_losses  ✓ 可导入')"
  local mem; mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr -dc '0-9')
  echo "  GPU used   = ${mem} MiB  (VAE 训练需 ~20-33G，**必须等 v74 结束**)"
  [ "${mem:-9999}" -lt 2000 ] && echo "  → GPU 空闲，可以起" || echo "  → ⚠ GPU 忙，先别起"
}

run_arm() {
  local tag=$1; shift
  local out=$ROOT/experiments/vae_fix_${tag}
  mkdir -p "$out"
  echo "=================================================================="
  echo ">>> 臂 $tag -> $out   args: $*"
  echo "=================================================================="
  PYTHONPATH="$ROOT" $PY -u "$ROOT/tools/vae/train_dino_calli_vae.py" \
     --vae-base "$BASE" --dino-ckpt "$DINO" --csv "$CSV" --data-root "$DATA_ROOT" \
     --output "$out" --max-steps "$STEPS" --batch-size 8 --grad-accum 4 \
     --lr 5e-5 --w-l1 1.0 --w-kl 1e-6 --w-struct 0.1 --w-style 0.5 \
     --save-every 500 --vis-every 250 "$@" 2>&1 | tee "$out/train.log"
}

case "${1:-check}" in
  check) check ;;
  run)
    check
    for a in $ARMS; do
      case "$a" in
        A) run_arm collapse --w-std 0.02 ;;
        B) run_arm hf       --w-hf 0.3 --hf-mode dog ;;
        C) run_arm both     --w-std 0.02 --w-hf 0.3 --hf-mode dog ;;
        *) echo "  未知臂: $a" ;;
      esac
    done
    echo "=== 全部完成。判定: python tools/vae/judge_vae.py --spec ..." ;;
  *) echo "用法: $0 [check|run]"; exit 2 ;;
esac
