#!/bin/bash
# =============================================================================
# run_s28_s29_pipeline.sh — 串行训练: base (s28, 早停) → skel-ctrl (s29, 早停)
#
# 阶段A base:   train.py + s28 配置 + cu121 + torch.compile + batch 384
#               auto_eval(true) + 默认早停(ssim_lpips, patience 5, min 20000)
# 阶段B ctrl:   train_controlnet.py + s29 配置 + cu121 + torch.compile + batch 192
#               --early-stop true (ctrl.ssim, patience 5)
#
# 配套: base 用 eval_metrics_daemon.py (CPU, 算指标写 eval_auto_*.json 供早停),
#       ctrl 用 eval_ctrl_metrics_daemon.py (CPU, 写 eval_auto_ctrl_*.json 供早停)。
# 两个 daemon 都是 while-True 轮询, 由本脚本在阶段结束时 kill。
#
# 用法: bash run_s28_s29_pipeline.sh   (建议放入 tmux 脱离 ssh)
# =============================================================================
set -u
cd /root/Workspace/xy/DiT
export PYTHONPATH=/root/Workspace/xy/DiT${PYTHONPATH:+:$PYTHONPATH}

PY_BASE=/opt/conda/bin/python            # base env (metrics daemon 用, CPU 轻量)
PY_CU=/opt/conda/envs/cu121/bin/python   # cu121 env (训练 + compile)
S28_RES=assets/results/s28_std_dino_pretrain
S29_RES=assets/results/s29_ctrl_gt_skel_1px

echo "[pipeline] $(date '+%F %T') ============ 阶段A: base s28 预训练 (compile + batch384 + 默认早停) ============"

# ---- 阶段A: base ----
# 1) 起 metrics daemon (CPU)
$PY_BASE src/eval/eval_metrics_daemon.py "$S28_RES" > /tmp/s28_metrics_pipe.log 2>&1 &
METRICS_PID=$!
echo "[pipeline] metrics daemon pid=$METRICS_PID (base)"
sleep 3

# 2) base 训练: compile + batch 384 (CLI 覆盖 config 的 global_batch_size=192)
mkdir -p "$S28_RES"
$PY_CU src/train/train.py \
    --config src/train/configs/s28_std_dino_pretrain.json \
    --compile true --compile-mode default \
    --global-batch-size 384 > /tmp/s28_train_pipe.log 2>&1
S28_EXIT=$?
echo "[pipeline] $(date '+%F %T') base train 退出, exit=$S28_EXIT"
kill $METRICS_PID 2>/dev/null
wait $METRICS_PID 2>/dev/null
echo "[pipeline] metrics daemon 已停"

# ---- 定位 base 最新 ckpt 作为 s29 main_ckpt ----
BASE_CKPT=$(ls -t $S28_RES/*/checkpoints/*.pt 2>/dev/null | grep -v ema_ | head -1)
if [ -z "$BASE_CKPT" ]; then
  echo "[pipeline] 错误: 未找到 base ckpt, 终止。"
  exit 1
fi
echo "[pipeline] base ckpt -> $BASE_CKPT"

echo "[pipeline] $(date '+%F %T') ============ 阶段B: skel-ctrl s29 (compile + batch192 + 早停) ============"

# ---- 阶段B: ctrl ----
# 1) ctrl metrics daemon (CPU)
$PY_BASE src/eval/eval_ctrl_metrics_daemon.py "$S29_RES" > /tmp/s29_metrics_pipe.log 2>&1 &
CTRL_METRICS_PID=$!
echo "[pipeline] ctrl metrics daemon pid=$CTRL_METRICS_PID"
sleep 3

# 2) ctrl 训练: compile + batch 192 + early-stop true (CLI 覆盖 config 的 batch_size=96)
mkdir -p "$S29_RES"
$PY_CU src/train/train_controlnet.py \
    --config src/train/configs/s29_ctrl_gt_skel_1px.json \
    --main-ckpt "$BASE_CKPT" \
    --compile true --compile-mode default \
    --batch-size 192 \
    --early-stop true --early-stop-metric ssim \
    --early-stop-patience 5 --early-stop-min-delta 0.002 \
    --early-stop-min-steps 10000 > /tmp/s29_ctrl_pipe.log 2>&1
S29_EXIT=$?
echo "[pipeline] $(date '+%F %T') ctrl train 退出, exit=$S29_EXIT"
kill $CTRL_METRICS_PID 2>/dev/null
wait $CTRL_METRICS_PID 2>/dev/null
echo "[pipeline] ctrl metrics daemon 已停"

echo "[pipeline] $(date '+%F %T') 全流程结束."
echo "[pipeline] 日志: /tmp/s28_train_pipe.log (base), /tmp/s29_ctrl_pipe.log (ctrl)"
exit 0