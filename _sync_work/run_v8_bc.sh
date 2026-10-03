#!/bin/bash
# run_v8_bc.sh — A 段(v8a base)已于 09-02 23:05 早停完成 (132.5k, ssim 0.5121 / lpips 0.4132 均 best)。
# 路径策略: 不依赖时间戳 glob。A best 先 copy 到固定路径 A_main_final.pt, B/C 全部从固定路径读写。
# 链路: A_main_final.pt → B(skel-ctrl v8b) → B_ctrl_best.pt → C(REPA v8c)
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT${PYTHONPATH:+:$PYTHONPATH}
# torch.compile 持久化缓存 (inductor 默认 /tmp 进程即丢)
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
PY=/opt/conda/envs/cu121/bin/python
LOG=/tmp/v8_3stage.log
FIX=assets/results/v8_3stage          # 固定路径根 (无时间戳)
mkdir -p "$FIX"

# ---- A best → 固定路径 (0132500 = ssim 0.5121 并列最高 + lpips 0.4132 最低 + EMA 最成熟) ----
A_SRC="$FIX/v8a/20260902-125615-v8a-s30-base/checkpoints/0132500.pt"
A_CKPT="$FIX/A_main_final.pt"
if [ ! -f "$A_CKPT" ]; then
    [ -f "$A_SRC" ] || { echo "[BC] A 源 ckpt 不存在: $A_SRC" >> $LOG; exit 1; }
    cp "$A_SRC" "$A_CKPT"
    echo "[BC] A best 已固化: $A_CKPT" >> $LOG
fi

# ---- B: s31-v8 skel-ctrl (config v8b, early_stop 已固化) ----
echo "========== [B] $(date '+%F %T') s31-v8 skel-ctrl (config v8b) ==========" >> $LOG
$PY src/train/train_controlnet.py \
    --config src/train/configs/v8b_s31_ctrl.json \
    --main-ckpt "$A_CKPT" \
    > /tmp/v8b_s31_ctrl.log 2>&1
RC=$?
echo "========== [B] rc=$RC $(date '+%F %T') ==========" >> $LOG
[ $RC -ne 0 ] && tail -30 /tmp/v8b_s31_ctrl.log >> $LOG

# B best ctrl ckpt → 固定路径 (按 eval_auto_ctrl ssim 最大选, 选完 copy 成固定名)
B_DIR="$FIX/v8b"
B_BEST=$(/opt/conda/bin/python - "$B_DIR" <<'EOF'
import os, sys, json, glob
best, bs = None, -1
for d in glob.glob(os.path.join(sys.argv[1], '*', 'checkpoints')):
    for f in glob.glob(os.path.join(d, 'eval_auto_ctrl_*.json')):
        try:
            dd = json.load(open(f))
            s = dd.get('ctrl', dd).get('ssim', -1)
            if s > bs:
                bs = s; best = f.replace('eval_auto_ctrl_', '').replace('.json', '.pt')
        except Exception:
            pass
print(best or '')
EOF
)
B_CKPT="$FIX/B_ctrl_best.pt"
if [ -n "$B_BEST" ] && [ -f "$B_BEST" ]; then
    cp "$B_BEST" "$B_CKPT"
    echo "[B] ctrl best 已固化: $B_BEST -> $B_CKPT" >> $LOG
else
    echo "[B] 无 eval 选优结果, 终止" >> $LOG
    exit 1
fi

# ---- C: REPA 强化 (config v8c; 全部从固定路径读) ----
echo "========== [C] $(date '+%F %T') s32-v8 REPA (config v8c) ==========" >> $LOG
$PY src/train/train_repa.py \
    --config src/train/configs/v8c_s32_repa.json \
    --main-ckpt "$A_CKPT" \
    --ctrl-ckpt "$B_CKPT" \
    > /tmp/v8c_s32_repa.log 2>&1
RC=$?
echo "========== [C] rc=$RC $(date '+%F %T') ==========" >> $LOG
[ $RC -ne 0 ] && tail -30 /tmp/v8c_s32_repa.log >> $LOG

echo "=== [v8-chain] B+C 全部完成 $(date '+%F %T') ===" >> $LOG
