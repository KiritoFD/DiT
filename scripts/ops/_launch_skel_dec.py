# -*- coding: utf-8 -*-
"""停旧 structnet + 同步并启动 skel decoder 训练。"""
import subprocess, time
HOST="root@10.176.54.17"; PORT="36430"

def run(cmd, timeout=90):
    for i in range(5):
        r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,cmd],
                           capture_output=True,text=True,timeout=timeout)
        if r.returncode==0: return r.stdout
        time.sleep(2)
    return None

# scp
r = subprocess.run(["scp","-o","ConnectTimeout=25","-P",PORT,
    r"G:\GitHub\DiT\tools\train_skel_decoder.py",
    "root@10.176.54.17:/root/Workspace/xy/DiT/tools/"],
    capture_output=True,timeout=60)
print("scp:", r.returncode==0)

# 停旧
print(run("pkill -9 -f train_struct_probe_256; pkill -9 -f train_struct_probe_256_v2; sleep 2; echo KILLED"))

# 启动 skel decoder
print(run("cd /root/Workspace/xy/DiT && tmux kill-session -t structnet 2>/dev/null; "
          "tmux new-session -d -s structnet '"
          "/opt/conda/bin/python tools/train_skel_decoder.py "
          "--csv 5script/train.csv --epochs 10 --batch-size 256 --num-threads 32 "
          "--log-every 50 > run_skel_dec.log 2>&1' && echo SKEL_STARTED"))

time.sleep(90)
print(run("tail -15 /root/Workspace/xy/DiT/run_skel_dec.log 2>/dev/null"))