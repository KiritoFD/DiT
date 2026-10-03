#!/bin/bash
# rapidocr on GPU —— onnxruntime-gpu 1.16.3 需要 CUDA 11.8 + cuDNN 8
# 系统只有 cuDNN 8.4，缺 libcublas.so.11 -> 用 pip 的 nvidia-*-cu11
cd /root/Workspace/xy/DiT
SP=/root/Workspace/xy/DiT/_venv_qwenvl/lib/python3.10/site-packages
P=""
for d in cublas cudnn cuda_runtime cuda_cupti curand cufft nvjitlink; do
  for v in cu11 cu12; do
    [ -d "$SP/nvidia/$d/lib" ] && P="$P:$SP/nvidia/$d/lib"
  done
done
export LD_LIBRARY_PATH="${P#:}:$LD_LIBRARY_PATH"
exec "$@"
