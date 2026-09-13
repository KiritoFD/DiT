#!/bin/bash
# 启动 gradio 公网链接（share=True），用当前最好 ckpt C（XL-r8）
cd /root/Workspace/xy/DiT
export PYTHONUNBUFFERED=1
CKPT="results/exp_xl_head_r8/20260814-211629-DiT-3Cond-XL-2/checkpoints/0037000.pt"
/opt/conda/bin/python gradio_app.py \
  --model-name DiT-3Cond-XL/2 --ckpt "$CKPT" \
  --use-lora 1 --lora-r 8 --lora-target all \
  --pretrained data/pretrained/DiT-XL-2-256x256.pt \
  --share --port 7860 > gradio_server.log 2>&1
echo "GRADIO_EXIT=$?" >> gradio_server.log
