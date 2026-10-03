#!/bin/bash
# run_v9_2stage.sh — 两阶段链 (重构后, 无 train_repa): 
#   A: base 预训练 (train.py, 全程挂 REPA w_repa) 
#   B: ctrl 后训练 (train_controlnet.py, 挂 REPA-early w_repa_early)
# 两阶段都可用 REPA (公共 infra src.loss.repa), eval 统一走 eval_facade。
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT${PYTHONPATH:+:$PYTHONPATH}
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
PY=/opt/conda/envs/cu121/bin/python
LOG=/tmp/v9_2stage.log
FIX=assets/results/v8_3stage

echo "=== [v9-2stage] $(date '+%F %T') 启动 (base REPA → ctrl REPA-early) ===" >> $LOG

# ---------- A: base 全程 REPA 预训练 (batch 320, 目标显存 22G) ----------
echo "--- [A] v9a repa-pretrain (w_repa=0.1) $(date '+%F %T') ---" >> $LOG
$PY src/train/train.py --config src/train/configs/v9a_repa_pretrain.json \
    > /tmp/v9a_repa.log 2>&1
RC=$?
echo "[A] rc=$RC $(date '+%F %T')" >> $LOG
[ $RC -ne 0 ] && tail -25 /tmp/v9a_repa.log >> $LOG

# A best ckpt: eval_auto_*.json 最高 ssim
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
            cand = os.path.join(d, 'checkpoints', f'{step}.pt')
            if os.path.exists(cand):
                best = cand
    except Exception:
        pass
print(best or '')
EOF
)
[ -z "$A_CKPT" ] && A_CKPT=$(ls -t $A_DIR/checkpoints/*.pt 2>/dev/null | head -1)
echo "[A] best ckpt: $A_CKPT (best from eval jsons)" >> $LOG
[ -z "$A_CKPT" ] || [ ! -f "$A_CKPT" ] && { echo "[A] 无 ckpt 终止" >> $LOG; exit 1; }

# ---------- B: ctrl 后训练 (v9b = v8b基础 + v8e强度 REPA w0.5, 早停) ----------
echo "--- [B] v9b ctrl (REPA-early w0.5, max 80k, 早停) $(date '+%F %T') ---" >> $LOG
$PY src/train/train_controlnet.py \
    --config src/train/configs/v9b_ctrl_strong.json \
    --main-ckpt "$A_CKPT" \
    > /tmp/v9b_ctrl.log 2>&1
echo "[B] rc=$? $(date '+%F %T')" >> $LOG
[ $? -ne 0 ] && tail -25 /tmp/v9b_ctrl.log >> $LOG

echo "=== [v9-2stage] 完成 $(date '+%F %T') ===" >> $LOG
echo "日志: /tmp/v9a_repa.log /tmp/v9b_ctrl.log (汇总: $LOG)"