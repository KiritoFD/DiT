# -*- coding: utf-8 -*-
"""列出并杀掉所有 train.py 进程 + 释放 MASTER_PORT 12355。"""
import subprocess, time

out = subprocess.run(["ps", "aux"], capture_output=True, text=True)
lines = out.stdout.splitlines()
procs = []
for line in lines:
    if "train.py" in line and "grep" not in line:
        parts = line.split()
        pid = parts[1]
        cmd = " ".join(parts[10:])
        procs.append((pid, cmd))
print("train.py 进程:")
for pid, cmd in procs:
    print(f"  PID {pid}: {cmd[:100]}")
for pid, _ in procs:
    subprocess.run(["kill", "-9", pid])
    print(f"  killed {pid}")
time.sleep(5)
# 检查 12355 端口
out2 = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True)
print("12355 端口监听:")
for line in out2.stdout.splitlines():
    if ":12355" in line:
        print(" ", line)
