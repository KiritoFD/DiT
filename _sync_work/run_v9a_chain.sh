#!/bin/bash
# run_v9a_chain.sh — 两阶段全程 REPA: A(v9a repa-pretrain) → B(v8e 方法后训练)
# A: train.py 全程挂 w_repa=0.1 (主模型 block8 → DINO), 早停 ssim_lpips
# B: train_repa.py (v8e/REPA 后置 w0.5, ctrl 分支从 B_ctrl_best warm start), 已加 early_stop
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT${PYTHONPATH:+:$PYTHONPATH}
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
PY=/opt/conda/envs/cu121/bin/python
LOG=/tmp/v9a_chain.log
FIX=assets/results/v8_3stage
B="$FIX/B_ctrl_best.pt"

echo "=== [v9a-chain] $(date '+%F %T') 启动 (全程REPA) ===" >> $LOG

# ---------- A: v9a repa 预训练 ----------
echo "--- [A] v9a repa-pretrain (w_repa 0.1) $(date '+%F %T') ---" >> $LOG
$PY src/train/train.py --config src/train/configs/v9a_repa_pretrain.json \
    > /tmp/v9a_repa.log 2>&1
RC=$?
echo "[A] rc=$RC $(date '+%F %T')" >> $LOG
[ $RC -ne 0 ] && tail -20 /tmp/v9a_repa.log >> $LOG

# A best ckpt 固化 (eval ssim_lpips best -> 找最新保存, 无删除所以最后 .pt 通常是 best, 但更稳妥找 eval 最高)
A_DIR=$(ls -dt assets/results/v9a_repa_pretrain/*/ 2>/dev/null | head -1)
A_CKPT=$(/opt/conda/bin/python - "$A_DIR" <<'EOF'
import os, sys, json, glob
d = sys.argv[1]
best, bs = None, -1
for f in glob.glob(os.path.join(d, 'checkpoints', 'eval_auto_*.json')):
    try:
        dd = json.load(open(f))
        s = dd.get('ssim', -1)
        if s > bs:
            bs = s
            step = os.path.basename(f).replace('eval_auto_', '').replace('.json', '')
            best = os.path.join(d, 'checkpoints', step + '.pt')
    except Exception:
        pass
print(best or (sorted(glob.glob(os.path.join(d, 'checkpoints', '*.pt')))[-1] if glob.glob(os.path.join(d, 'checkpoints', '*.pt')) else ''))
EOF
)
echo "[A] best ckpt: $A_CKPT" >> $LOG
[ -z "$A_CKPT" ] || [ ! -f "$A_CKPT" ] && { echo "[A] 无 ckpt 终止" >> $LOG; exit 1; }

# ---------- B: v8e 方法后训练 (REPA w0.5, ctrl warm start) ----------
echo "--- [B] v8e-method REPA w0.5 @80k $(date '+%F %T') ---" >> $LOG
$PY src/train/train_repa.py \
    --config src/train/configs/v8e_repa_strong.json \
    --main-ckpt "$A_CKPT" --ctrl-ckpt "$B" > /tmp/v9a_v8e.log 2>&1
echo "[B] rc=$? $(date '+%F %T')" >> $LOG

echo "=== [v9a-chain] 完成 $(date '+%F %T') ===" >> $LOG
echo "日志: /tmp/v9a_repa.log /tmp/v9a_v8e.log (汇总: $LOG)"