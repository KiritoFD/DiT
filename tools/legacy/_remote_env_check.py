#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""远程环境检查 (Python 3.6 compatible)"""
import subprocess

def run(cmd):
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out = (r.stdout + r.stderr).strip()
        return out[:300]
    except Exception as e:
        return str(e)[:300]

print("=== which python3 ===")
print(run(["which", "python3"]))

print("=== conda python torch ===")
print(run(["/opt/conda/bin/python", "-c", "import torch; print('torch='+torch.__version__+' cuda='+str(torch.cuda.is_available()))"]))

print("=== conda python transformers ===")
print(run(["/opt/conda/bin/python", "-c", "import transformers; print('transformers='+transformers.__version__)"]))

print("=== GPU mem ===")
print(run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"]))

print("=== sample image ===")
print(run(["/opt/conda/bin/python", "-c", "from PIL import Image; im=Image.open('/root/Workspace/xy/DiT/final_images/0.png'); print('size='+str(im.size)+' mode='+im.mode)"]))
