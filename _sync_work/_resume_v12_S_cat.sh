#!/bin/bash
# 续训 v12 (DiT-2Cond-S/2 + factorized_cat + glyph_vec), 从 step 55000 跑到 max_steps=400000。
#
# 为什么单独写一个脚本 (而不是复用 _launch_v12_S_cat.sh):
#   1) 那个脚本不收 resume 参数。
#   2) **不要 --fresh-scheduler**: train.py:856-858 在 resume-full 时会
#      从 ckpt 恢复 scheduler 状态 -> LR 精确接续 (warmup 已过, 处于 cosine 中段)。
#      加了 --fresh-scheduler 反而会按"绝对 step 0"重算, 语义不同。
#
# ⚠⚠ 关键坑 (已实测): --resume-full 的 start_step 是从**文件名里的数字**推断的
#     (train.py:763-766, 取 re.findall(r"\d+") 的**最后一个**数字串)。
#     实测:
#       .../checkpoints/0055000.pt             -> digits ['0055000']        -> 55000  ✓
#       assets/results/v12_S2_cat_last.pt      -> digits ['12', '2']       -> 2      ✗ 灾难
#         (后果: LR 从 step 2 重算 -> warmup=3000 -> LR~6.7e-8 看着像卡死;
#          ckpt 从 0001000.pt 重新命名 -> **覆盖已有 ckpt**; eval CSV 追加低 step 重复行)
#       v11_M432_best_strict_152500.pt         -> digits [...,'152500']    -> 152500 ✓ (凑巧)
#     ckpt 里虽然有顶层 'train_steps', 但 resume 侧只读 ck['args'].train_steps
#     (train.py:769-771), 而注释 train.py:759 自己写着 args 没有该字段 -> 死键。
#     => **必须用 checkpoints/ 下的数字命名 ckpt**。
set -u

NAME=${1:-v12cont}
CFG=${2:-v12_pretrain_S_cat_fame_kxl_tj_px60}
RES=${3:-v12_pretrain_S_cat_fame_kxl_tj_px60}
CKPT=${4:?resume ckpt 绝对路径}
RESUME_LR=${5:-}

cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
PY=/opt/conda/envs/cu121/bin/python
LOGD=/root/Workspace/xy/DiT/logs/$RES
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOGD"

if [ ! -f "$CKPT" ]; then
    echo "[resume] FATAL: ckpt not found: $CKPT"; exit 1
fi
case "$(basename "$CKPT")" in
    [0-9]*.pt) : ;;
    *) echo "[resume] WARN: ckpt 文件名不以数字开头 -> start_step 推断可能错误: $(basename "$CKPT")" ;;
esac

EXTRA=""
if [ -n "$RESUME_LR" ]; then
    EXTRA="--resume-lr $RESUME_LR"
    echo "[resume] override LR -> $RESUME_LR"
fi

# ---- 停旧训练 ----
tmux kill-session -t "$NAME" 2>/dev/null
pkill -f "train.py --config src/train/configs/" 2>/dev/null
sleep 5
echo "[resume] after pkill:"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

# ---- 起续训 ----
tmux new-session -d -s "$NAME" \
  "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export PYTHONPATH=/root/Workspace/xy/DiT; export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor; $PY -u src/train/train.py --config src/train/configs/$CFG.json --resume-full $CKPT $EXTRA 2>&1 | tee $LOGD/train_resume_$TS.log"

sleep 30
echo "[resume] tmux:"; tmux ls 2>/dev/null
echo "[resume] --- resume 关键行 ---"
grep -aE "resume-full|Inferred start step|Restored optimizer|cosine schedule|early-stop|Trainable Parameters" "$LOGD/train_resume_$TS.log" 2>/dev/null | head -20
echo "[resume] --- log tail ---"; tail -6 "$LOGD/train_resume_$TS.log" 2>/dev/null
echo "[resume] gpu:"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "[resume] LOGFILE=$LOGD/train_resume_$TS.log"
