#!/bin/bash
# _watchdog.sh — 全自动跑完 C/D/F/E 四个实验：训练 → 等 Done → eval_test 终评 → 下一个。
# 完全脱离 SSH：nohup bash _watchdog.sh > watchdog_out.log 2>&1 &
# 结果汇总到 experiment_summary.txt。

cd /root/Workspace/xy/DiT
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

SUMMARY=experiment_summary.txt
echo "" >> "$SUMMARY"
echo "=== watchdog start $(date '+%F %T') ===" >> "$SUMMARY"

# 队列: name|config(.json)|model|use_lora|lora_r|lora_target|pretrained|out
QUEUE=(
  "C|exp_xl_head_r8.json|DiT-3Cond-XL/2|1|8|all|pretrained_models/DiT-XL-2-256x256.pt|test_eval_C"
  "D|exp_xl_head_r32.json|DiT-3Cond-XL/2|1|32|all|pretrained_models/DiT-XL-2-256x256.pt|test_eval_D"
  "F|exp_xl_head_r64.json|DiT-3Cond-XL/2|1|64|all|pretrained_models/DiT-XL-2-256x256.pt|test_eval_F"
  "E|exp_xl_head_r32_attn.json|DiT-3Cond-XL/2|1|32|attn|pretrained_models/DiT-XL-2-256x256.pt|test_eval_E"
)

logmsg() { echo "[watchdog $(date '+%F %T')] $*"; }

# 训练进程是否还活着
training_alive() { ps aux | grep "[t]rain.py" | grep -q "$1"; }

# 启动训练
start_training() {
  local cfg="$1" log="$2"
  tmux kill-session -t exp 2>/dev/null
  pkill -9 -f "[t]rain.py" 2>/dev/null
  sleep 3
  rm -f "$log"
  tmux new-session -d -s exp "bash /root/Workspace/xy/DiT/_launch_exp.sh $cfg $log"
  sleep 5
}

# 等训练结束：成功(Done!)返回0，异常返回1
wait_done() {
  local cfg="$1" log="$2"
  local waited=0
  while true; do
    if grep -q "Done!" "$log" 2>/dev/null; then
      return 0
    fi
    if grep -qE "Error during training loop|CUDA out of memory|Traceback" "$log" 2>/dev/null; then
      logmsg "$cfg: ERROR in log"
      return 1
    fi
    # 进程没了但没 Done 也没报错：可能被外部杀掉
    if ! training_alive "$cfg"; then
      if [ -s "$log" ]; then
        logmsg "$cfg: training process gone without Done! (killed?)"
        return 1
      fi
    fi
    sleep 30
    waited=$((waited+30))
    if [ $((waited % 600)) -eq 0 ]; then
      local laststep=$(grep -oE "step=[0-9]+" "$log" 2>/dev/null | tail -1)
      logmsg "$cfg: still training ... $laststep (waited ${waited}s)"
    fi
  done
}

latest_ckpt() { ls -t results/"$1"/*/checkpoints/*.pt 2>/dev/null | head -1; }

run_one() {
  local entry="$1"
  IFS='|' read -r name cfg model use_lora lora_r lora_target pretrained out <<< "$entry"
  local log="${cfg%.json}.log"
  logmsg "===== 实验 $name ($cfg) ====="

  # 训练
  logmsg "$name: 启动训练"
  start_training "$cfg" "$log"
  if ! wait_done "$cfg" "$log"; then
    logmsg "$name: 训练异常，跳过终评"
    echo "${name}: TRAIN_FAIL ($(date '+%F %T'))" >> "$SUMMARY"
    return
  fi
  logmsg "$name: 训练完成 (Done!)"

  # 终评
  local ckpt
  ckpt=$(latest_ckpt "${cfg%.json}")
  if [ -z "$ckpt" ]; then
    logmsg "$name: 无 checkpoint"
    echo "${name}: NO_CKPT" >> "$SUMMARY"
    return
  fi
  logmsg "$name: 终评 ckpt=$ckpt"
  bash _eval_test.sh "$ckpt" "$model" "$use_lora" "$lora_r" "$pretrained" "$out" "$lora_target" > "${out}.eval_log" 2>&1
  if [ -f "${out}.json" ]; then
    local mse ssim
    mse=$(/opt/conda/bin/python -c "import json;print(json.load(open('${out}.json'))['mse'])" 2>/dev/null)
    ssim=$(/opt/conda/bin/python -c "import json;print(json.load(open('${out}.json'))['ssim'])" 2>/dev/null)
    echo "${name}: ckpt=$(basename $ckpt) MSE=$mse SSIM=$ssim" >> "$SUMMARY"
    logmsg "$name: 终评 DONE  MSE=$mse SSIM=$ssim"
  else
    logmsg "$name: 终评失败（无 ${out}.json）"
    echo "${name}: EVAL_FAIL" >> "$SUMMARY"
  fi
}

for entry in "${QUEUE[@]}"; do
  run_one "$entry"
done

echo "=== watchdog finished $(date '+%F %T') ===" >> "$SUMMARY"
logmsg "ALL DONE. summary -> $SUMMARY"
