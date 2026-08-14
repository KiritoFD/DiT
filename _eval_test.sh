#!/bin/bash
# 10k test 终评启动脚本：bash _eval_test.sh <ckpt> <model> <use_lora> <lora_r> <pretrained> <out> [lora_target]
cd /root/Workspace/xy/DiT
CKPT="$1"; MODEL="$2"; USE_LORA="$3"; LORA_R="$4"; PRETRAINED="$5"; OUT="$6"; LORA_TARGET="${7:-all}"
/opt/conda/bin/python eval_test.py \
  --ckpt "$CKPT" --model "$MODEL" --use-lora "$USE_LORA" --lora-r "$LORA_R" \
  --lora-target "$LORA_TARGET" \
  --pretrained "$PRETRAINED" --num-calligraphers 1873 --out "$OUT"
echo "EVAL_DONE=$?"
