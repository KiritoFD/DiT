#!/bin/bash
# 串行跑 D(exp_xl_head_r32) / F(exp_xl_head_r64) / E(exp_xl_head_r32_attn) 的 10k test 终评
cd /root/Workspace/xy/DiT
C1="results/exp_xl_head_r32/20260814-231414-DiT-3Cond-XL-2/checkpoints/0037000.pt"
C2="results/exp_xl_head_r64/20260815-012124-DiT-3Cond-XL-2/checkpoints/0037000.pt"
C3="results/exp_xl_head_r32_attn/20260815-032642-DiT-3Cond-XL-2/checkpoints/0037000.pt"
PRE="pretrained_models/DiT-XL-2-256x256.pt"

echo "===== D (r32) ====="
/opt/conda/bin/python eval_test.py --ckpt "$C1" --model DiT-3Cond-XL/2 --use-lora 1 --lora-r 32 --pretrained "$PRE" --num-calligraphers 1873 --out test_eval_D
echo "===== F (r64) ====="
/opt/conda/bin/python eval_test.py --ckpt "$C2" --model DiT-3Cond-XL/2 --use-lora 1 --lora-r 64 --pretrained "$PRE" --num-calligraphers 1873 --out test_eval_F
echo "===== E (r32 attn) ====="
/opt/conda/bin/python eval_test.py --ckpt "$C3" --model DiT-3Cond-XL/2 --use-lora 1 --lora-r 32 --lora-target attn --pretrained "$PRE" --num-calligraphers 1873 --out test_eval_E
echo "ALL_DONE"
