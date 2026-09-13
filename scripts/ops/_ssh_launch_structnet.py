# -*- coding: utf-8 -*-
"""在远程 CPU 启动 struct probe 256 训练 (tmux 独立 session, 不影响 GPU pixel 训练)。"""
import subprocess, time

HOST = "root@10.176.54.17"
PORT = "36430"

def run(remote_cmd, timeout=120, retries=6):
    for i in range(retries):
        try:
            r = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=25", "-o", "ServerAliveInterval=10",
                 "-p", PORT, HOST, remote_cmd],
                capture_output=True, text=True, timeout=timeout)
            if r.returncode == 0:
                return r.stdout
            print(f"[ssh rc={r.returncode}] {r.stderr[:150]}")
        except subprocess.TimeoutExpired:
            print("[ssh timeout]")
        time.sleep(3)
    return None

# tmux 启动 CPU struct probe 训练 (独立 session structnet, 与 GPU exp 隔离)
cmd = ("cd /root/Workspace/xy/DiT && "
       "tmux kill-session -t structnet 2>/dev/null; "
       "tmux new-session -d -s structnet '"
       "/opt/conda/bin/python tools/train_struct_probe_256.py "
       "--csv 5script/train.csv --epochs 8 --batch-size 64 --num-threads 16 "
       "> run_structnet.log 2>&1' && echo STRUCTNET_STARTED")
out = run(cmd, timeout=60)
print("launch:", out)

time.sleep(30)
out = run("head -30 /root/Workspace/xy/DiT/run_structnet.log 2>/dev/null")
print("verify:\n", out)