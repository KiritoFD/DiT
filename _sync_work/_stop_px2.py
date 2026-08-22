# -*- coding: utf-8 -*-
import subprocess, time
HOST="root@10.176.54.17"; PORT="36430"
def run(cmd, timeout=60):
    for i in range(6):
        try:
            r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,cmd],
                               capture_output=True,text=True,timeout=timeout)
            if r.returncode==0: return r.stdout
        except subprocess.TimeoutExpired:
            pass
        time.sleep(3)
    return None
print(run("tmux kill-session -t exp 2>/dev/null", timeout=20))
print(run("pkill -9 -f 'src/train_pixel'", timeout=20))
time.sleep(2)
print(run("nvidia-smi --query-gpu=memory.used --format=csv,noheader", timeout=20))