#!/bin/bash
# run_v8_3stage.sh — 新数据(v8资产集) 三段训练链: A(base S30-v8) → B(skel-ctrl s31-v8) → C(REPA s32-v8)
# 全部关键参数已固化进各 config (v8a/v8b/v8c), CLI 只传 ckpt 路径 (main-ckpt/ctrl-ckpt 必要参数)。
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT${PYTHONPATH:+:$PYTHONPATH}
# torch.compile 持久化缓存: 默认 inductor 用 tempfile(/tmp) 进程结束即丢,
# 固定到磁盘目录避免每个进程/每次切 batch 反复编译。
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
PY=/opt/conda/envs/cu121/bin/python
CHAIN_RES=5script/results/v8_3stage
mkdir -p "$CHAIN_RES"
LOG=/tmp/v8_3stage.log

echo "=== [v8-chain] $(date '+%F %T') 启动 ================" >> $LOG

# ---- A: S30-v8 base 预训练 (batch 384, compile, OT 全在 config) ----
echo "========== [A] $(date '+%F %T') S30-v8 base 预训练 (config v8a) ==========" >> $LOG
$PY src/train/train.py --config src/train/configs/v8a_s30_base.json \
    > /tmp/v8a_s30_base.log 2>&1
RC=$?
echo "========== [A] rc=$RC $(date '+%F %T') ==========" >> $LOG
[ $RC -ne 0 ] && tail -30 /tmp/v8a_s30_base.log >> $LOG

# 找 A 的最终 ckpt (best 或最后一个)
A_CKPT=$(ls -dt 5script/results/v8_3stage/v8a_*/checkpoints/*.pt 2>/dev/null | head -1)
[ -z "$A_CKPT" ] && A_CKPT=$(ls -dt 5script/results/v8_3stage/*/checkpoints/*.pt 2>/dev/null | head -1)
echo "[A] main ckpt: $A_CKPT" >> $LOG
[ -z "$A_CKPT" ] && { echo "[A] 无 ckpt 终止"; exit 1; }

# ---- B: s31-v8 skel-ctrl (config v8b, early_stop 已固化) ----
echo "========== [B] $(date '+%F %T') s31-v8 skel-ctrl (config v8b) ==========" >> $LOG
$PY src/train/train_controlnet.py \
    --config src/train/configs/v8b_s31_ctrl.json \
    --main-ckpt "$A_CKPT" \
    > /tmp/v8b_s31_ctrl.log 2>&1
RC=$?
echo "========== [B] rc=$RC $(date '+%F %T') ==========" >> $LOG
[ $RC -ne 0 ] && tail -30 /tmp/v8b_s31_ctrl.log >> $LOG

# 找 B 的 best ctrl ckpt (取 eval ctrl.ssim 最大, 找不到就最后)
B_DIR=$(ls -dt 5script/results/v8_3stage/v8b_* 2>/dev/null | head -1)
B_CKPT=$(/opt/conda/bin/python - "$B_DIR" <<'EOF'
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
[ -z "$B_CKPT" ] && { echo "[B] 无 ckpt 终止"; exit 1; }

# ---- C: REPA 强化 (config v8c; 只传两个 ckpt 路径) ----
echo "========== [C] $(date '+%F %T') s32-v8 REPA 强化 (config v8c) ==========" >> $LOG
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