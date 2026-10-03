#!/bin/bash
# run_posttrain_vi.sh — 后训练链 v3: v8i(解冻+REPA早挂) → v8e(repa-strong) → v8f(repa-deep) → v8g(repa-lrlow)
# v8d/v8h 已完成保留结果。起点固定路径: A_main_final.pt / B_ctrl_best.pt
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT${PYTHONPATH:+:$PYTHONPATH}
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
PY=/opt/conda/envs/cu121/bin/python
LOG=/tmp/v8efgi.log
FIX=assets/results/v8_3stage
A="$FIX/A_main_final.pt"
B="$FIX/B_ctrl_best.pt"

echo "=== [posttrain-v3] $(date '+%F %T') 启动 v8i->v8e->v8f->v8g ===" >> $LOG

# ---------- v8i: B 段解冻 + REPA 早挂 (组合) ----------
echo "--- [v8i] unfreeze-main + repa-early w0.2 $(date '+%F %T') ---" >> $LOG
$PY src/train/train_controlnet.py \
    --config src/train/configs/v8i_unfreeze_repa_early.json \
    --main-ckpt "$A" > /tmp/v8i.log 2>&1
echo "[v8i] rc=$? $(date '+%F %T')" >> $LOG

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

# ---------- v8g: REPA 低 LR (3e-5) ----------
echo "--- [v8g] repa-lrlow lr=3e-5 $(date '+%F %T') ---" >> $LOG
$PY src/train/train_repa.py \
    --config src/train/configs/v8g_repa_lrlow.json \
    --main-ckpt "$A" --ctrl-ckpt "$B" > /tmp/v8g.log 2>&1
echo "[v8g] rc=$? $(date '+%F %T')" >> $LOG

echo "=== [posttrain-v3] 全部完成 $(date '+%F %T') ===" >> $LOG
echo "日志: /tmp/v8i.log /tmp/v8e.log /tmp/v8f.log /tmp/v8g.log (汇总: $LOG)"