# -*- coding: utf-8 -*-
"""同步 v2 脚本 + 启动全量 GPU 训练 (329k, skel_d3+canny_d3, 30ep)."""
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

# 编译验证
print("compile:", run("cd /root/Workspace/xy/DiT && /opt/conda/bin/python -m py_compile tools/train_struct_decoder_gpu.py && echo OK", timeout=30))

# 清旧结果
print("clean:", run("rm -rf /root/Workspace/xy/DiT/5script/results/struct_decoder_gpu 2>/dev/null; echo C", timeout=30))

# 启动: 全量 329k, skel_d3+canny_d3, base=96 depth=6 (pixel-shuffle 解码), 30 epoch
print("launch:", run("cd /root/Workspace/xy/DiT && "
    "tmux new-session -d -s structgpu '"
    "/opt/conda/bin/python tools/train_struct_decoder_gpu.py "
    "--csv 5script/train_full.csv --epochs 30 --batch-size 512 "
    "--base 96 --depth 6 --skel-pos-weight 15 --canny-pos-weight 8 "
    "--out-dir 5script/results/struct_decoder_gpu --log-every 100 "
    "> run_struct_gpu.log 2>&1' && echo GPU_STARTED", timeout=30))

time.sleep(60)
print("log:", run("tail -12 /root/Workspace/xy/DiT/run_struct_gpu.log 2>/dev/null", timeout=30))