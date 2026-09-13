# -*- coding: utf-8 -*-
"""scp 修复版脚本 + 杀旧 skel3 + 重启。"""
import subprocess, time
HOST="root@10.176.54.17"; PORT="36430"

def run(cmd, timeout=90):
    for i in range(5):
        r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,cmd],
                           capture_output=True,text=True,timeout=timeout)
        if r.returncode==0: return r.stdout
        time.sleep(2)
    return None

# scp 修复版
r = subprocess.run(["scp","-o","ConnectTimeout=25","-P",PORT,
    r"G:\GitHub\DiT\tools\train_skel_decoder.py",
    "root@10.176.54.17:/root/Workspace/xy/DiT/tools/"],
    capture_output=True,timeout=60)
print("scp:", r.returncode==0)

# 确认远程文件已更新
print(run("grep -c 'skel-root' /root/Workspace/xy/DiT/tools/train_skel_decoder.py"))

# 杠旧的 skel3 session + 重启
print(run("tmux kill-session -t skel3 2>/dev/null; "
          "cd /root/Workspace/xy/DiT && "
          "tmux new-session -d -s skel3 '"
          "/opt/conda/bin/python tools/train_skel_decoder.py "
          "--csv 5script/train.csv --epochs 10 --batch-size 256 --num-threads 16 "
          "--skel-root final_skeleton_d3 --pos-weight 15 --out-dir 5script/results/skel_decoder_d3 "
          "--log-every 50 > run_skel_d3.log 2>&1' && echo D3_STARTED"))

time.sleep(90)
print(run("tail -12 /root/Workspace/xy/DiT/run_skel_d3.log 2>/dev/null"))