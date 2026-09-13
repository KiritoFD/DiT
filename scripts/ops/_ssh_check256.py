# -*- coding: utf-8 -*-
"""查看远程 cache 构建日志全文 + 进程列表。"""
import subprocess

HOST = "root@10.176.54.17"
PORT = "36430"

cmd = ("cat /tmp/_build_cache256.log 2>/dev/null | tail -40; echo '=== PROC ==='; "
       "ps aux | grep build_latent_structure | grep -v grep | head -5")
for i in range(6):
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=25", "-p", PORT, HOST, cmd],
            capture_output=True, text=True, timeout=90)
        if r.returncode == 0:
            print(r.stdout)
            break
        print(f"[rc={r.returncode}] {r.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print("[timeout]")
    if i == 5:
        import sys; sys.exit(1)
    import time; time.sleep(3)