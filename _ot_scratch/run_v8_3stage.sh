#!/bin/bash
# run_v8_3stage.sh — 新数据(v8资产集) 三段训练链: A(base S30-v8) → B(skel-ctrl s31-v8) → C(REPA s32-v8)
# 设计要点:
#   A: S30-v8 base 预训练 (纯 diff, 数据 = train_fame_clean_v8.csv + final_latents_fame_v8)
#   B: s31-v8 skel-ctrl warm-start (train_ctrl_only, 数据 = v8; skel = final_skel_latents_fame_1px_v8)
#   C: REPA (train_repa.py, 基于 B best ckpt, w_repa=0.3, max_steps 20k 内防 base 崩)
# 串行链, 每段独立 results 子目录; 全程 daemon watch.
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT${PYTHONPATH:+:$PYTHONPATH}
PY=/opt/conda/envs/cu121/bin/python
CHAIN_RES=5script/results/v8_3stage
mkdir -p "$CHAIN_RES"
LOG=/tmp/v8_3stage.log

echo "=== [v8-chain] $(date '+%F %T') 启动 ================" >> $LOG

# ---- A: S30-v8 base 预训练 ----
echo "========== [A] $(date '+%F %T') S30-v8 base 预训练 ==========" >> $LOG
$PY src/train/train.py \
    --config src/train/configs/v8a_s30_base.json \
    > /tmp/v8a_s30_base.log 2>&1
RC=$?
echo "========== [A] rc=$RC $(date '+%F %T') ==========" >> $LOG
[ $RC -ne 0 ] && tail -30 /tmp/v8a_s30_base.log >> $LOG

# 找 A 的最终 ckpt (best 或最后一个)
A_CKPT=$(ls -dt 5script/results/v8_3stage/v8a_*/checkpoints/*.pt 2>/dev/null | head -1)
[ -z "$A_CKPT" ] && A_CKPT=$(ls -dt 5script/results/v8_3stage/*/checkpoints/*.pt 2>/dev/null | head -1)
echo "[A] main ckpt: $A_CKPT" >> $LOG

# ---- B: s31-v8 skel-ctrl ----
echo "========== [B] $(date '+%F %T') s31-v8 skel-ctrl ==========" >> $LOG
$PY src/train/train_controlnet.py \
    --config src/train/configs/v8b_s31_ctrl.json \
    --main-ckpt "$A_CKPT" \
    > /tmp/v8b_s31_ctrl.log 2>&1
RC=$?
echo "========== [B] rc=$RC $(date '+%F %T') ==========" >> $LOG
[ $RC -ne 0 ] && tail -30 /tmp/v8b_s31_ctrl.log >> $LOG

# 找 B 的 best ctrl ckpt (取 eval ctrl.ssim 最大, 找不到就最后)
B_DIR=$(ls -dt 5script/results/v8_3stage/v8b_* 2>/dev/null | head -1)
[ -z "$B_DIR" ] && B_DIR=$(ls -dt 5script/results/v8_3stage/* 2>/dev/null | grep -iv v8a | head -1)
B_CKPT=$(python3 - "$B_DIR" <<'EOF'
import os, sys, json, glob
d = sys.argv[1]
best, bs = None, -1
for f in glob.glob(os.path.join(d, 'checkpoints', 'eval_auto_ctrl_*.json')):
    try:
        dd = json.load(open(f))
        s = dd.get('ctrl', dd).get('ssim', -1)
        if s > bs:
            bs = s; best = f.replace('eval_auto_ctrl_', '').replace('.json', '.pt')
    except Exception:
        pass
print(best or (sorted(glob.glob(os.path.join(d, 'checkpoints', '*.pt')))[-1] if glob.glob(os.path.join(d, 'checkpoints', '*.pt')) else ''))
EOF
)
echo "[B] ctrl ckpt: $B_CKPT (ssim best)" >> $LOG

# ---- C: REPA ----
echo "========== [C] $(date '+%F %T') s32-v8 REPA 强化 ==========" >> $LOG
$PY src/train/train_repa.py \
    --config src/train/configs/v8c_s32_repa.json \
    --main-ckpt "$A_CKPT" \
    --ctrl-ckpt "$B_CKPT" \
    > /tmp/v8c_s32_repa.log 2>&1
RC=$?
echo "========== [C] rc=$RC $(date '+%F %T') ==========" >> $LOG
[ $RC -ne 0 ] && tail -30 /tmp/v8c_s32_repa.log >> $LOG

echo "=== [v8-chain] 全部完成 $(date '+%F %T') ===" >> $LOG
echo "日志: /tmp/v8a_s30_base.log /tmp/v8b_s31_ctrl.log /tmp/v8c_s32_repa.log (汇总: $LOG)"