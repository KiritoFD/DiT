#!/bin/bash
# run_v9c_skel_joint.sh — v9c skel 联合预训练拉起 (GPU 空闲守护点火).
#
# 用法 (拉起守护, 等 v9b 结束后自动开训):
#   tmux new-session -d -s v9c_waiter 'bash /root/Workspace/xy/DiT/_sync_work/run_v9c_skel_joint.sh'
#
# 逻辑:
#   1. 每 120s 查 GPU 显存, 连续 2 次 < 2000MiB 视为 v9b 已退出 (+60s 稳定期)
#   2. 自动挑 v9a best ckpt (eval_auto ssim 最高) 作为联训初始主干
#   3. train_controlnet.py --config v9c_skel_joint.json --main-ckpt <v9a best>
#      (train_ctrl_only=true + unfreeze_main=true: ctrl lr 1e-4, main lr 3e-5,
#       char 表保持冻结, REPA-early w=0.5 layers 8,11)
#   4. 日志: /tmp/v9c_joint.log
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT${PYTHONPATH:+:$PYTHONPATH}
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
PY=/opt/conda/envs/cu121/bin/python
LOG=/tmp/v9c_joint.log

echo "=== [v9c-waiter] $(date '+%F %T') 启动, 等待 GPU 空闲 (v9b 早停) ===" >> $LOG
FREE=0
while [ $FREE -lt 2 ]; do
  sleep 120
  MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "${MEM:-99999}" -lt 2000 ]; then FREE=$((FREE+1)); else FREE=0; fi
done
sleep 60
echo "=== [v9c] $(date '+%F %T') GPU 空闲, 点火 ===" >> $LOG

# ---- v9a best ckpt (eval_auto 最高 ssim) ----
A_DIR=$(ls -dt assets/results/v9a_repa_pretrain/*/ 2>/dev/null | head -1)
A_CKPT=$($PY - "$A_DIR" <<'EOF'
import os, sys, json, glob
d = sys.argv[1]
best, best_step = -1, ''
for f in glob.glob(os.path.join(d, 'checkpoints', 'eval_auto_*.json')):
    try:
        dd = json.load(open(f))
        s = dd.get('ssim', -1)
        if s > best:
            best = s
            best_step = os.path.basename(f).replace('eval_auto_', '').replace('.json', '')
    except Exception:
        pass
if best_step:
    cand = os.path.join(d, 'checkpoints', f'{best_step}.pt')
    if os.path.exists(cand):
        print(cand)
EOF
)
[ -z "$A_CKPT" ] && A_CKPT=$(ls -t ${A_DIR}checkpoints/*.pt 2>/dev/null | head -1)
echo "=== [v9c] main-ckpt = $A_CKPT (v9a best) ===" >> $LOG
if [ -z "$A_CKPT" ] || [ ! -f "$A_CKPT" ]; then
  echo "[v9c] 无可用 main ckpt, 终止" >> $LOG
  exit 1
fi

$PY -u src/train/train_controlnet.py \
    --config src/train/configs/v9c_skel_joint.json \
    --main-ckpt "$A_CKPT" \
    >> $LOG 2>&1
RC=$?
echo "=== [v9c] rc=$RC $(date '+%F %T') ===" >> $LOG
[ $RC -ne 0 ] && tail -25 $LOG
