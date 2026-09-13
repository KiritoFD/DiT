# -*- coding: utf-8 -*-
"""scp + 重启, 分步执行避免 timeout."""
import subprocess, time
HOST="root@10.176.54.17"; PORT="36430"

# scp
r = subprocess.run(["scp","-o","ConnectTimeout=25","-P",PORT,
    r"G:\GitHub\DiT\tools\train_struct_decoder_gpu.py",
    "root@10.176.54.17:/root/Workspace/xy/DiT/tools/"],
    capture_output=True,timeout=60)
print("scp:", r.returncode==0)

# kill + clean (一条命令)
for i in range(5):
    try:
        r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,
            "tmux kill-session -t structgpu 2>/dev/null; pkill -9 -f train_struct_decoder 2>/dev/null; rm -rf /root/Workspace/xy/DiT/5script/results/struct_decoder_gpu; echo DONE"],
            capture_output=True,text=True,timeout=30)
        if r.returncode==0:
            print("clean:", r.stdout.strip()); break
    except subprocess.TimeoutExpired:
        pass
    time.sleep(3)

# launch
for i in range(5):
    try:
        r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,
            "cd /root/Workspace/xy/DiT && tmux new-session -d -s structgpu '/opt/conda/bin/python tools/train_struct_decoder_gpu.py --csv 5script/train_full.csv --epochs 30 --batch-size 256 --base 64 --depth 6 --skel-pos-weight 15 --canny-pos-weight 8 --out-dir 5script/results/struct_decoder_gpu --log-every 100 > run_struct_gpu.log 2>&1' && echo STARTED"],
            capture_output=True,text=True,timeout=30)
        if r.returncode==0:
            print("launch:", r.stdout.strip()); break
    except subprocess.TimeoutExpired:
        pass
    time.sleep(3)