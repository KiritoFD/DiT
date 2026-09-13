# -*- coding: utf-8 -*-
"""停 CPU 3px + 同步 GPU 脚本 + tmux 启动 GPU skel+canny 训练."""
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

# scp
r = subprocess.run(["scp","-o","ConnectTimeout=25","-P",PORT,
    r"G:\GitHub\DiT\tools\train_struct_decoder_gpu.py",
    "root@10.176.54.17:/root/Workspace/xy/DiT/tools/"],
    capture_output=True,timeout=60)
print("scp:", r.returncode==0)

# 停 CPU 3px
print("kill cpu:", run("tmux kill-session -t skel3 2>/dev/null; pkill -9 -f train_skel_decoder 2>/dev/null; sleep 2; echo K", timeout=30))

# 编译验证
print("compile:", run("cd /root/Workspace/xy/DiT && /opt/conda/bin/python -m py_compile tools/train_struct_decoder_gpu.py && echo OK", timeout=30))

# 启动 GPU 训练
print("launch:", run("cd /root/Workspace/xy/DiT && "
    "tmux new-session -d -s structgpu '"
    "/opt/conda/bin/python tools/train_struct_decoder_gpu.py "
    "--csv 5script/train.csv --epochs 20 --batch-size 512 --base 64 --depth 4 "
    "--skel-pos-weight 15 --canny-pos-weight 10 "
    "--out-dir 5script/results/struct_decoder_gpu "
    "--log-every 50 > run_struct_gpu.log 2>&1' && echo GPU_STARTED", timeout=30))

time.sleep(30)
print("log:", run("tail -10 /root/Workspace/xy/DiT/run_struct_gpu.log 2>/dev/null", timeout=30))