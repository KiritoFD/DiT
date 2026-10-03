#!/bin/bash
cd /root/Workspace/xy/DiT
export HF_ENDPOINT=https://hf-mirror.com
V=_venv_qwenvl/bin/python
echo "[dl] 开始下载 Qwen3-VL-4B-Instruct"
$V -c "
import os
os.environ.setdefault(\"HF_ENDPOINT\",\"https://hf-mirror.com\")
try:
    from modelscope import snapshot_download
    p=snapshot_download(\"Qwen/Qwen3-VL-4B-Instruct\", cache_dir=\"/root/Workspace/xy/DiT/_models\")
    print(\"[ok] modelscope\", p)
except Exception as e:
    print(\"[!] modelscope 失败:\", e)
    from huggingface_hub import snapshot_download as h
    p=h(\"Qwen/Qwen3-VL-4B-Instruct\", cache_dir=\"/root/Workspace/xy/DiT/_models\")
    print(\"[ok] hf-mirror\", p)
"
echo "[dl] DONE"
