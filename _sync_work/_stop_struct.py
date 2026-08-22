# -*- coding: utf-8 -*-
"""停训练(ep3 已在跑,用 ep2 best), 手动跑梯度健康检查 + 显存开销评估."""
import subprocess, time
HOST="root@10.176.54.17"; PORT="36430"

def run(cmd, timeout=60):
    for i in range(5):
        try:
            r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,cmd],
                               capture_output=True,text=True,timeout=timeout)
            if r.returncode==0: return r.stdout
        except subprocess.TimeoutExpired:
            pass
        time.sleep(3)
    return None

# 停训练 (ep2 已保存 best, ep3 在跑但不等了)
print("kill:", run("tmux kill-session -t structgpu 2>/dev/null; pkill -9 -f train_struct_decoder 2>/dev/null; sleep 2; echo K", timeout=30))
print("GPU:", run("nvidia-smi --query-gpu=memory.used --format=csv,noheader", timeout=20))