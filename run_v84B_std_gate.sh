#!/bin/bash
# ============================================================================
# v84B 系列：**空间注入时间门**（单变量实验，2026-10-13 挂上）
#
# 背景（本次读码实测，不是猜测）:
#   · 代码里 `_gate` 同时乘到两路：`x += glyph_scale * (gate * g_tok)`
#     （src/model/dit_conditioning.py:778）与池化进 adaLN 的 `e_glyph_vec`
#     （:786-789）⇒ 打开它就同时管住两条通路 ✓
#   · **但 v84A 根本没打开**：glyph_gate_t 默认 0.0 ⇒ _gate=None
#     ⇒ 空间骨架注入在**整个去噪过程恒为全强度** ✓（这正是待检验的怀疑点）
#   · 时间约定 x_t=(1-t)x0+t*noise ⇒ t=1 是噪声端 ⇒ 门恒为「高 t 全强度、低 t 收 floor」✓
#
# 用法:
#   bash run_v84B_std_gate.sh --dry-run     # 只打印将要执行的命令 ✓
#   tmux new -s v84b "bash run_v84B_std_gate.sh"    # 等 v84A 结束后自动跑三个变体
#
# 行为:
#   ① 等当前 src.train.train 进程退出（**不 pkill** ✓ 绝不打断正在跑的 v84A）
#   ② 以 v84A 最新 checkpoint 为起点，**fine-tune** 三个门控变体（各 30k 步）
#   ③ 每个变体独立日志；结果落在 assets/results/v84B_std_gate/
# ============================================================================
set -u
cd /root/Workspace/xy/DiT

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
PY=/opt/conda/envs/cu121/bin/python

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

FT_STEPS=${FT_STEPS:-30000}          # 每个变体 fine-tune 步数（分析建议 20–40k ✓）
V84A_RUN=$(ls -d assets/results/v84A_std_cond/*/ 2>/dev/null | head -1)
RESULTS=assets/results/v84B_std_gate
mkdir -p "$RESULTS" logs

echo "=== v84B 空间注入时间门实验 @ $(date) ==="
echo "  v84A run: ${V84A_RUN:-'(未找到)'}"
echo "  fine-tune 步数: $FT_STEPS | dry-run=$DRY"

# ① 等 GPU 空出来（**不杀任何进程** ✓）
if [ "$DRY" = "0" ]; then
  while pgrep -f 'src\.train\.train' >/dev/null 2>&1; do
    echo "  [$(date '+%H:%M:%S')] 等待现有训练结束（不打断 ✓）..."
    sleep 300
  done
  echo "  ✓ GPU 已空"
fi

# ② 起点 checkpoint = v84A 里最新的一个
CKPT=""
if [ -n "${V84A_RUN:-}" ]; then
  CKPT=$(ls -t "${V84A_RUN}"checkpoints/*.pt 2>/dev/null | head -1)
fi
echo "  起点 ckpt: ${CKPT:-'(未找到 ⇒ 将从头训练 ✗ 请检查)'}"

# ③ 三个单变量变体（配置里与 v84A 的差异**只有** glyph_gate_t / glyph_gate_floor ✓）
run_variant () {
  local CFG=$1 NAME=$2
  local LOG="logs/std_${NAME}_$(date +%Y%m%d-%H%M%S).log"
  local ARGS=(--config "src/train/configs/${CFG}" --results-dir "$RESULTS")
  if [ -n "$CKPT" ]; then
    local CUR
    CUR=$($PY - "$CKPT" <<'PYEOF'
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(sd.get("train_steps", 0)) if isinstance(sd, dict) else 0)
PYEOF
)
    ARGS+=(--resume-full "$CKPT" --max-steps "$((CUR + FT_STEPS))")
    echo "    起点步数=$CUR ⇒ 目标 max_steps=$((CUR + FT_STEPS))"
  fi
  echo "  --- [$NAME]  $PY -u -m src.train.train ${ARGS[*]}  | tee $LOG"
  if [ "$DRY" = "0" ]; then
    PYTHONPATH=. $PY -u -m src.train.train "${ARGS[@]}" 2>&1 | tee "$LOG"
    echo "  ✓ [$NAME] 结束 rc=${PIPESTATUS[0]} @ $(date)"
  fi
}

run_variant v84B_std_gate05_floor0.json     v84B-gate05-floor0
run_variant v84C_std_gate05_floor35.json    v84C-gate05-floor35
run_variant v84D_std_gate90_floor0.json     v84D-gate90-floor0

echo "=== 全部变体结束 @ $(date) ==="
