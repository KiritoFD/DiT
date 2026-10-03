#!/bin/bash
# run_v8i_grid.sh — v8i 超参网格 (a/c/d) + 后续 REPA 扫描 (e/f/g)
# v8i-b(0.2/3e-5/layer8) 已跑完 0.7657/0.5531 作为网格基准, 不重跑。
# 起点固定: A_main_final.pt / B_ctrl_best.pt
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT${PYTHONPATH:+:$PYTHONPATH}
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
PY=/opt/conda/envs/cu121/bin/python
LOG=/tmp/v8igrid.log
FIX=assets/results/v8_3stage
A="$FIX/A_main_final.pt"
B="$FIX/B_ctrl_best.pt"

echo "=== [v8i-grid] $(date '+%F %T') 启动 ===" >> $LOG

# ---- v8i-a: 最小干预 (w0.1 / main_lr 1e-5 / layer8) ----
echo "--- [v8ia] minimal w0.1 main_lr1e-5 $(date '+%F %T') ---" >> $LOG
$PY src/train/train_controlnet.py \
    --config src/train/configs/v8ia_minimal.json \
    --main-ckpt "$A" > /tmp/v8ia.log 2>&1
echo "[v8ia] rc=$? $(date '+%F %T')" >> $LOG

# ---- v8i-c: 层前移 (w0.2 / main_lr 3e-5 / layer6) ----
echo "--- [v8ic] layer6 $(date '+%F %T') ---" >> $LOG
$PY src/train/train_controlnet.py \
    --config src/train/configs/v8ic_layer6.json \
    --main-ckpt "$A" > /tmp/v8ic.log 2>&1
echo "[v8ic] rc=$? $(date '+%F %T')" >> $LOG

# ---- v8i-d: 强度顶格 (w0.3 / main_lr 5e-5 / layer8) ----
echo "--- [v8id] strong w0.3 main_lr5e-5 $(date '+%F %T') ---" >> $LOG
$PY src/train/train_controlnet.py \
    --config src/train/configs/v8id_strong.json \
    --main-ckpt "$A" > /tmp/v8id.log 2>&1
echo "[v8id] rc=$? $(date '+%F %T')" >> $LOG

# ---- v8e: REPA 强 w0.5 @80k ----
echo "--- [v8e] repa-strong w0.5 @80k $(date '+%F %T') ---" >> $LOG
$PY src/train/train_repa.py \
    --config src/train/configs/v8e_repa_strong.json \
    --main-ckpt "$A" --ctrl-ckpt "$B" > /tmp/v8e.log 2>&1
echo "[v8e] rc=$? $(date '+%F %T')" >> $LOG

# ---- v8f: REPA 深 layers 8,11,15 @60k ----
echo "--- [v8f] repa-deep @60k $(date '+%F %T') ---" >> $LOG
$PY src/train/train_repa.py \
    --config src/train/configs/v8f_repa_deep.json \
    --main-ckpt "$A" --ctrl-ckpt "$B" > /tmp/v8f.log 2>&1
echo "[v8f] rc=$? $(date '+%F %T')" >> $LOG

# ---- v8g: REPA 低 LR @100k ----
echo "--- [v8g] repa-lrlow @100k $(date '+%F %T') ---" >> $LOG
$PY src/train/train_repa.py \
    --config src/train/configs/v8g_repa_lrlow.json \
    --main-ckpt "$A" --ctrl-ckpt "$B" > /tmp/v8g.log 2>&1
echo "[v8g] rc=$? $(date '+%F %T')" >> $LOG

echo "=== [v8i-grid] 全部完成 $(date '+%F %T') ===" >> $LOG
echo "日志: /tmp/v8ia.log /tmp/v8ic.log /tmp/v8id.log /tmp/v8e.log /tmp/v8f.log /tmp/v8g.log (汇总: $LOG)"