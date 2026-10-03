#!/bin/bash
# run_s32c_chain.sh — 串行实验链: s32c (长收敛) → s32d (超强REPA), 独立对照.
# 每个实验从 s31@42500 冷启 (不同假设单独对比), 各自 results 子目录, daemon 全程 watch s32c_chain.
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT${PYTHONPATH:+:$PYTHONPATH}
PY=/opt/conda/envs/cu121/bin/python
MAIN=assets/results/s30_dino_char_strong_pretrain/20260901-052520-s30-dino-char-strong-pretrain/checkpoints/0132500.pt
CTRL=assets/results/s31_ctrl_gt_skel_1px/20260901-135832-s31-ctrl-gt-skel-1px/checkpoints/0042500.pt
CHAIN_RES=assets/results/s32c_chain
mkdir -p "$CHAIN_RES"

# 全局 metrics daemon (看整个 s32c_chain; 修过: 目录不存在会 makedirs + 循环 try/except)
pkill -f 'eval_ctrl_metrics_daemon' 2>/dev/null
sleep 1
nohup /opt/conda/bin/python src/eval/eval_ctrl_metrics_daemon.py "$CHAIN_RES" > /tmp/s32c_chain_metrics.log 2>&1 &
echo "metrics daemon pid=$! -> $CHAIN_RES"

for EXP_CONFIG in \
  src/train/configs/s32c-repa-longconv.json \
  src/train/configs/s32d-repa-super.json ; do
  EXP_NAME=$(grep -oE '"experiment_name"[^,]*' "$EXP_CONFIG" | sed 's/"experiment_name"[[:space:]]*:[[:space:]]*"//;s/"//g')
  echo "========== [chain] $(date '+%F %T') 启动 $EXP_NAME =========="
  $PY src/train/train_repa.py \
      --config "$EXP_CONFIG" \
      --main-ckpt "$MAIN" \
      --ctrl-ckpt "$CTRL" \
      --compile true --compile-mode default \
      > "/tmp/${EXP_NAME}.log" 2>&1
  RC=$?
  echo "========== [chain] $(date '+%F %T') $EXP_NAME 退出 rc=$RC =========="
  [ $RC -ne 0 ] && tail -30 "/tmp/${EXP_NAME}.log"
done
echo "[chain] 全部完成: $CHAIN_RES"
echo "日志: /tmp/s32c-repa-longconv.log /tmp/s32d-repa-super.log"