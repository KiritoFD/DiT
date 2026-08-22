# -*- coding: utf-8 -*-
"""查看 structnet 训练日志。"""
import subprocess, time

HOST = "root@10.176.54.17"
PORT = "36430"

cmd = ("tail -25 /root/Workspace/xy/DiT/run_structnet.log 2>/dev/null; "
       "echo '=== tmux ==='; tmux ls 2>/dev/null; "
       "echo '=== procs ==='; pgrep -af train_struct_probe | head -3")
for i in range(6):
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=25", "-p", PORT, HOST, cmd],
            capture_output=True, text=True, timeout=90)
        if r.returncode == 0:
            print(r.stdout)
            break
        print(f"[rc={r.returncode}] {r.stderr[:150]}")
    except subprocess.TimeoutExpired:
        print("[timeout]")
    time.sleep(3)