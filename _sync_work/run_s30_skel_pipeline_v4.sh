#!/bin/bash
# =============================================================================
# run_s30_skel_pipeline.sh — 串行训练 v4 (幂等版, 已有 ckpt 的阶段自动跳过)
#   阶段A: base s30 (DINO真迹字表+char强化, 默认早停)  compile+batch384  [resume-full]
#   阶段B: skel-ctrl s31 (1px GT skel, 默认早停)        compile+batch192
#   阶段C: REPA 联合微调 s32 (diff+REPA, 防灾难遗忘)    compile, 保守 batch
#
# v4 变更: 若 S30/S31 检出已有非 ema ckpt 则跳过对应阶段的训练, 直接进入下一阶段,
#          支持崩溃后重启续跑。注意: 运行期间不要覆盖本文件 (bash 增量读取会错乱)。
# 用法: tmux new-session -d -s s30pipe "bash _sync_work/run_s30_skel_pipeline.sh"
# =============================================================================
set -u
cd /root/Workspace/xy/DiT
export PYTHONPATH=/root/Workspace/xy/DiT${PYTHONPATH:+:$PYTHONPATH}

PY_BASE=/opt/conda/bin/python            # base env (metrics daemon 用, CPU 轻量)
PY_CU=/opt/conda/envs/cu121/bin/python   # cu121 env (训练 + compile)
S30_RES=assets/results/s30_dino_char_strong_pretrain
CTRL_RES=assets/results/s31_ctrl_gt_skel_1px
REPA_RES=assets/results/s32_repa_finetune
RESUME_CKPT="${RESUME_CKPT:-}"

# ------------------------------------------------------------------ 阶段A
BASE_CKPT=$(ls -t "$S30_RES"/*/checkpoints/*.pt 2>/dev/null | grep -v ema_ | head -1)
if [ -z "$BASE_CKPT" ]; then
  echo "[pipeline] $(date '+%F %T') 阶段A: 无 s30 ckpt, 开始 base 预训练"
  $PY_BASE src/eval/eval_metrics_daemon.py "$S30_RES" > /tmp/s30_metrics_pipe.log 2>&1 &
  METRICS_PID=$!
  sleep 3
  mkdir -p "$S30_RES"
  S30_ARGS=(--config src/train/configs/s30_dino_char_strong_pretrain.json
            --compile true --compile-mode default --global-batch-size 384)
  if [ -n "$RESUME_CKPT" ]; then
    S30_ARGS+=(--resume-full "$RESUME_CKPT")
    echo "[pipeline] resume-full from $RESUME_CKPT"
  fi
  $PY_CU src/train/train.py "${S30_ARGS[@]}" > /tmp/s30_train_pipe.log 2>&1
  echo "[pipeline] $(date '+%F %T') 阶段A train 退出 exit=$?"
  kill $METRICS_PID 2>/dev/null; wait $METRICS_PID 2>/dev/null
  BASE_CKPT=$(ls -t "$S30_RES"/*/checkpoints/*.pt 2>/dev/null | grep -v ema_ | head -1)
fi
if [ -z "$BASE_CKPT" ]; then
  echo "[pipeline] 错误: 无 base ckpt, 终止。"; exit 1
fi
echo "[pipeline] $(date '+%F %T') 阶段A 完成, base ckpt = $BASE_CKPT"

# ------------------------------------------------------------------ 阶段B
CTRL_CKPT=$(ls -t "$CTRL_RES"/*/checkpoints/*.pt 2>/dev/null | grep -v ema_ | head -1)
if [ -z "$CTRL_CKPT" ]; then
  echo "[pipeline] $(date '+%F %T') 阶段B: 无 s31 ckpt, 开始 skel-ctrl 训练"
  $PY_BASE src/eval/eval_ctrl_metrics_daemon.py "$CTRL_RES" > /tmp/s31_metrics_pipe.log 2>&1 &
  CTRL_METRICS_PID=$!
  sleep 3
  mkdir -p "$CTRL_RES"
  $PY_CU src/train/train_controlnet.py \
      --config src/train/configs/s31_ctrl_gt_skel_1px.json \
      --main-ckpt "$BASE_CKPT" \
      --compile true --compile-mode default \
      --early-stop true --early-stop-metric ssim \
      --early-stop-patience 5 --early-stop-min-delta 0.002 \
      --early-stop-min-steps 10000 > /tmp/s31_ctrl_pipe.log 2>&1
  echo "[pipeline] $(date '+%F %T') 阶段B train 退出 exit=$?"
  kill $CTRL_METRICS_PID 2>/dev/null; wait $CTRL_METRICS_PID 2>/dev/null
  CTRL_CKPT=$(ls -t "$CTRL_RES"/*/checkpoints/*.pt 2>/dev/null | grep -v ema_ | head -1)
fi
if [ -z "$CTRL_CKPT" ]; then
  echo "[pipeline] 警告: 无 s31 ctrl ckpt, 跳过阶段C。"; exit 0
fi
echo "[pipeline] $(date '+%F %T') 阶段B 完成, ctrl ckpt = $CTRL_CKPT"

# ------------------------------------------------------------------ 阶段C
echo "[pipeline] $(date '+%F %T') 阶段C: REPA 联合微调 s32 (diff+REPA)"
mkdir -p "$REPA_RES"
$PY_CU src/train/train_repa.py \
    --config src/train/configs/s32_repa_finetune.json \
    --main-ckpt "$BASE_CKPT" \
    --ctrl-ckpt "$CTRL_CKPT" \
    --compile true --compile-mode default > /tmp/s32_repa_pipe.log 2>&1
echo "[pipeline] $(date '+%F %T') 阶段C train 退出 exit=$?"
echo "[pipeline] $(date '+%F %T') 全部阶段结束"
echo "[pipeline] 日志: /tmp/s30_train_pipe.log (base), /tmp/s31_ctrl_pipe.log (ctrl), /tmp/s32_repa_pipe.log (repa)"
exit 0