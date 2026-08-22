# -*- coding: utf-8 -*-
"""停 GPU 上一切 + 查 3px skel decoder 当前进度, 决定是否直接切 GPU 训."""
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

# GPU 状态
print("GPU:", run("nvidia-smi --query-gpu=memory.used --format=csv,noheader"))
# 3px 进度
print("3px:", run("tail -3 /root/Workspace/xy/DiT/run_skel_d3.log"))
print("3px HIST:", run("cat /root/Workspace/xy/DiT/5script/results/skel_decoder_d3/history.json 2>/dev/null"))
# tmux
print("TMUX:", run("tmux ls 2>/dev/null"))