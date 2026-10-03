#!/bin/bash
# run_posttrain_v8d-h.sh — 5 个后训练配置串行跑 (v8d/v8e/v8f/v8g/v8h + v8d-retry 归一)
# 起点固定路径: A_main_final.pt (A段best) / B_ctrl_best.pt (B段best ctrl)
# 顺序: v8d(解冻,需重跑) → v8h(repa-early) → v8e(repa-strong) → v8f(repa-deep) → v8g(repa-short)
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT${PYTHONPATH:+:$PYTHONPATH}
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
PY=/opt/conda/envs/cu121/bin/python
LOG=/tmp/v8degh.log
FIX=assets/results/v8_3stage
A="$FIX/A_main_final.pt"
B="$FIX/B_ctrl_best.pt"

echo "=== [posttrain-v2] $(date '+%F %T') 启动 5 实验串行 (v8d重跑) ===" >> $LOG

# ---------- v8d: B 段解冻主模型 (param group 已修复) ----------
echo "--- [v8d] unfreeze-main (main_lr 3e-5) $(date '+%F %T') ---" >> $LOG
$PY src/train/train_controlnet.py \
    --config src/train/configs/v8d_unfreeze_main.json \
    --main-ckpt "$A" > /tmp/v8d.log 2>&1
echo "[v8d] rc=$? $(date '+%F %T')" >> $LOG
[ $? -ne 0 ] && tail -20 /tmp/v8d.log >> $LOG

# ---------- v8h: B 段早期 REPA 挂载 ----------
echo "--- [v8h] repa-early w=0.2 $(date '+%F %T') ---" >> $LOG
$PY src/train/train_controlnet.py \
    --config src/train/configs/v8h_repa_early.json \
    --main-ckpt "$A" > /tmp/v8h.log 2>&1
echo "[v8h] rc=$? $(date '+%F %T')" >> $LOG

# ---------- v8e: REPA 强 w=0.5 @30k ----------
echo "--- [v8e] repa-strong w=0.5 @30k $(date '+%F %T') ---" >> $LOG
$PY src/train/train_repa.py \
    --config src/train/configs/v8e_repa_strong.json \
    --main-ckpt "$A" --ctrl-ckpt "$B" > /tmp/v8e.log 2>&1
echo "[v8e] rc=$? $(date '+%F %T')" >> $LOG

# ---------- v8f: REPA 深 layers 8,11,15 ----------
echo "--- [v8f] repa-deep layers 8,11,15 $(date '+%F %T') ---" >> $LOG
$PY src/train/train_repa.py \
    --config src/train/configs/v8f_repa_deep.json \
    --main-ckpt "$A" --ctrl-ckpt "$B" > /tmp/v8f.log 2>&1
echo "[v8f] rc=$? $(date '+%F %T')" >> $LOG

# ---------- v8g: REPA 低 LR (3e-5, vs v8c 1e-4) ----------
echo "--- [v8g] repa-lrlow lr=3e-5 $(date '+%F %T') ---" >> $LOG
$PY src/train/train_repa.py \
    --config src/train/configs/v8g_repa_lrlow.json \
    --main-ckpt "$A" --ctrl-ckpt "$B" > /tmp/v8g.log 2>&1
echo "[v8g] rc=$? $(date '+%F %T')" >> $LOG

echo "=== [posttrain-v2] 全部完成 $(date '+%F %T') ===" >> $LOG
echo "日志: /tmp/v8d.log /tmp/v8h.log /tmp/v8e.log /tmp/v8f.log /tmp/v8g.log (汇总: $LOG)"