#!/usr/bin/env bash
# 便宜版注入 A/B: 共用主干(4ch 40k ckpt), 只换注入头, 每臂 3k 步
cd /root/Workspace/xy/DiT || exit 1
mkdir -p exp-std/logs_branch
LOG="exp-std/logs_branch/ab_$(date +%Y%m%d-%H%M%S).log"
exec > "$LOG" 2>&1
ln -sf "$(basename "$LOG")" exp-std/logs_branch/ab_latest.log
echo "logfile=$LOG start=$(date '+%F %T')"
export PYTHONPATH=/root/Workspace/xy/DiT
export HF_HUB_OFFLINE=1
PY=/opt/conda/envs/cu121/bin/python
ARMS="${ARMS:-A_xattn_cont B_adaln4 C_every_layer D_k4_styleca}"
echo "arms: $ARMS"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

for A in $ARMS; do
  echo
  echo "###############################################################"
  echo "##### ARM ${A}  $(date '+%F %T') #####"
  echo "###############################################################"
  # ★ 必须覆盖 LR: 调度器按 total=43000 恢复后, 37500 步处只剩 7e-6(cosine 尾巴),
  #   四臂都会以极低 LR 微调 -> 区分不出来。分支适配用 2e-5, 且各臂一致。
  $PY -u src/train/train.py --config "src/train/configs/v49_branch_${A}.json" \
       --resume-full "$(grep -oE 'exp-std/runs_purestd/[^"]+\.pt' src/train/configs/v49_branch_${A}.json | head -1)" \
       --resume-lr 2e-5
  RC=$?
  echo "##### ARM ${A} 结束 rc=$RC $(date '+%F %T') #####"
  if [ "$RC" -ne 0 ]; then echo "[FATAL] $A rc=$RC -> 中止"; break; fi
done

echo
echo "########## 各臂 eval200fix 曲线 ##########"
for A in $ARMS; do
  echo "--- $A"
  find exp-std/runs_branch -path "*${A}*" -name 'eval_stdskel_summary.csv' 2>/dev/null \
    | while read -r f; do echo "  [$f]"; cat "$f"; done
done
echo "[ab] 结束 $(date '+%F %T')"
