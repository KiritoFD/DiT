#!/bin/bash
# 给 Qwen3-VL 建独立环境并下载模型。
# ⚠ 绝不能升级 /opt/conda/envs/cu121 的 transformers（4.36.2 -> 需要 >=4.49），
#   因为 v15b 正在用那个环境训练，升级可能把它搞崩。
set -u
cd /root/Workspace/xy/DiT

VENV=/root/Workspace/xy/DiT/_venv_qwenvl
if [ ! -d "$VENV" ]; then
    echo "[vlm] 建 venv ..."
    /opt/conda/envs/cu121/bin/python -m venv "$VENV"
fi
PY="$VENV/bin/python"
"$PY" -m pip install -q --upgrade pip

echo "[vlm] 装依赖 (transformers>=4.49, torch cpu) ..."
"$PY" -m pip install -q "transformers>=4.49" "torch" "torchvision" \
    "accelerate" "qwen-vl-utils" "pillow" "opencc-python-reimplemented" \
    "modelscope" 2>&1 | tail -5

echo "[vlm] 版本检查:"
"$PY" -c "
import transformers, torch
print('  transformers', transformers.__version__)
print('  torch', torch.__version__, 'cuda', torch.cuda.is_available())
"

echo "[vlm] 下载 Qwen3-VL-4B-Instruct ..."
"$PY" - <<'EOF'
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from modelscope import snapshot_download
try:
    p = snapshot_download("Qwen/Qwen3-VL-4B-Instruct",
                          cache_dir="/root/Workspace/xy/DiT/_models")
    print("  [ok] modelscope ->", p)
except Exception as e:
    print("  modelscope 失败:", e)
    print("  试 hf-mirror ...")
    from huggingface_hub import snapshot_download as hsd
    p = hsd("Qwen/Qwen3-VL-4B-Instruct",
            cache_dir="/root/Workspace/xy/DiT/_models",
            endpoint="https://hf-mirror.com")
    print("  [ok] hf-mirror ->", p)
EOF
echo "[vlm] DONE"
