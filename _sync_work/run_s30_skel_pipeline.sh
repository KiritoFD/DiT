#!/bin/bash
# =============================================================================
# run_s30_skel_pipeline.sh — 串行训练 v3 (含 REPA 阶段C):
#   阶段A: base s30 (DINO真迹字表+char强化, 默认早停)  compile+batch384  [resume-full]
#   阶段B: skel-ctrl s31 (1px GT skel, 默认早停)        compile+batch192
#   阶段C: REPA 联合微调 s32 (diff+REPA, 防灾难遗忘)    compile, 保守 batch
#
# 用法: bash run_s30_skel_pipeline.sh (建议放入 tmux 脱离 ssh)
# 说明: 阶段C 需要 s31 ckpt 有 ctrl encoder; 若 ctrl 意外结束早停且 ckpt 齐全则直接接续.
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

echo "[pipeline] $(date '+%F %T') ============ 阶段A: base s30 预训练 (DINO字表+char强化, compile+batch384, 默认早停) ============"

# ---- 阶段A: base ----
$PY_BASE src/eval/eval_metrics_daemon.py "$S30_RES" > /tmp/s30_metrics_pipe.log 2>&1 &
METRICS_PID=$!
echo "[pipeline] metrics daemon pid=$METRICS_PID (base)"
sleep 3

mkdir -p "$S30_RES"
S30_ARGS=(--config src/train/configs/s30_dino_char_strong_pretrain.json
          --compile true --compile-mode default --global-batch-size 384)
if [ -n "$RESUME_CKPT" ]; then
  S30_ARGS+=(--resume-full "$RESUME_CKPT")
  echo "[pipeline] resume-full from $RESUME_CKPT"
fi
$PY_CU src/train/train.py "${S30_ARGS[@]}" > /tmp/s30_train_pipe.log 2>&1
S30_EXIT=$?
echo "[pipeline] $(date '+%F %T') base train 退出, exit=$S30_EXIT"
kill $METRICS_PID 2>/dev/null
wait $METRICS_PID 2>/dev/null
echo "[pipeline] metrics daemon 已停"

# ---- 定位 base 最新 ckpt 作为 ctrl main_ckpt ----
BASE_CKPT=$(ls -t $S30_RES/*/checkpoints/*.pt 2>/dev/null | grep -v ema_ | head -1)
if [ -z "$BASE_CKPT" ]; then
  echo "[pipeline] 错误: 未找到 base ckpt, 终止。"
  exit 1
fi
echo "[pipeline] base ckpt -> $BASE_CKPT"

echo "[pipeline] $(date '+%F %T') ============ 阶段B: skel-ctrl s31 (compile+batch192, 早停) ============"

# ---- 阶段B: skel-ctrl ----
$PY_BASE src/eval/eval_ctrl_metrics_daemon.py "$CTRL_RES" > /tmp/s31_metrics_pipe.log 2>&1 &
CTRL_METRICS_PID=$!
echo "[pipeline] ctrl metrics daemon pid=$CTRL_METRICS_PID"
sleep 3

mkdir -p "$CTRL_RES"
$PY_CU src/train/train_controlnet.py \
    --config src/train/configs/s31_ctrl_gt_skel_1px.json \
    --main-ckpt "$BASE_CKPT" \
    --compile true --compile-mode default \
    --early-stop true --early-stop-metric ssim \
    --early-stop-patience 5 --early-stop-min-delta 0.002 \
    --early-stop-min-steps 10000 > /tmp/s31_ctrl_pipe.log 2>&1
S31_EXIT=$?
echo "[pipeline] $(date '+%F %T') ctrl train 退出, exit=$S31_EXIT"
kill $CTRL_METRICS_PID 2>/dev/null
wait $CTRL_METRICS_PID 2>/dev/null
echo "[pipeline] ctrl metrics daemon 已停"

echo "[pipeline] $(date '+%F %T') ============ 阶段C: REPA 联合微调 s32 (diff+REPA) ============"
CTRL_CKPT=$(ls -t $CTRL_RES/*/checkpoints/*.pt 2>/dev/null | grep -v ema_ | head -1)
if [ -z "$CTRL_CKPT" ]; then
  echo "[pipeline] 错误: 未找到 s31 ctrl ckpt, 跳过阶段C。"
else
  echo "[pipeline] ctrl ckpt -> $CTRL_CKPT"
  mkdir -p "$REPA_RES"
  $PY_CU src/train/train_repa.py \
      --config src/train/configs/s32_repa_finetune.json \
      --main-ckpt "$BASE_CKPT" \
      --ctrl-ckpt "$CTRL_CKPT" \
      --compile true --compile-mode default \
      > /tmp/s32_repa_pipe.log 2>&1
  S32_EXIT=$?
  echo "[pipeline] $(date '+%F %T') repa train 退出, exit=$S32_EXIT"
fi

echo "[pipeline] $(date '+%F %T') 全部阶段结束."
echo "[pipeline] 日志: /tmp/s30_train_pipe.log (base), /tmp/s31_ctrl_pipe.log (ctrl), /tmp/s32_repa_pipe.log (repa)"
exit 0