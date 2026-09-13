# -*- coding: utf-8 -*-
"""停 pixel 训练, 手动 GPU eval 50000 ckpt, 存可视化."""
import subprocess, time
HOST="root@10.176.54.17"; PORT="36430"

def run(cmd, timeout=120):
    for i in range(5):
        try:
            r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,cmd],
                               capture_output=True,text=True,timeout=timeout)
            if r.returncode==0: return r.stdout
        except subprocess.TimeoutExpired:
            pass
        time.sleep(3)
    return None

# 停训练
print("kill:", run("tmux kill-session -t exp 2>/dev/null; pkill -9 -f 'src/train_pixel' 2>/dev/null; sleep 2; nvidia-smi --query-gpu=memory.used --format=csv,noheader", timeout=30))