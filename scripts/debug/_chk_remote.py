# -*- coding: utf-8 -*-
"""看 train_pixel.py 在远程的实际内容 (eval 调用段)."""
import subprocess
HOST="root@10.176.54.17"; PORT="36430"
r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,
    "grep -n 'eval\|ckpt_every\|_run_gpu\|_save_ckpt' /root/Workspace/xy/DiT/src/train_pixel.py | head -20"],
    capture_output=True,text=True,timeout=60)
print(r.stdout if r.returncode==0 else f"rc={r.returncode}")