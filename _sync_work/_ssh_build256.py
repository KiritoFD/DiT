# -*- coding: utf-8 -*-
"""确认远程工具已更新 + 清 pyc + 重跑构建。"""
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

# 确认工具行内容更新了
out = run("grep -n 'expected 256x256' /root/Workspace/xy/DiT/tools/build_latent_structure_cache.py; "
          "grep -n 'expected \\${size} \\* 8' /root/Workspace/xy/DiT/tools/build_latent_structure_cache.py")
print("grep tool:\n", out)

# 清 pyc + 杀旧进程 + 重跑
out = run("find /root/Workspace/xy/DiT/tools/__pycache__ -name '*structure_cache*' -delete 2>/dev/null; "
          "pkill -9 -f build_latent_structure_cache.py 2>/dev/null; sleep 1; "
          "rm -f /root/Workspace/xy/DiT/5script/structnet/structure256_cache.npz*; "
          "cd /root/Workspace/xy/DiT && nohup /opt/conda/bin/python tools/build_latent_structure_cache.py "
          "--csv 5script/train.csv --out 5script/structnet/structure256_cache.npz "
          "--size 256 --workers 16 > /tmp/_build_cache256b.log 2>&1 & echo BUILD3_STARTED")
print("launch:", out)

time.sleep(15)
out = run("pgrep -f build_latent_structure_cache | head -3; echo ---; head -10 /tmp/_build_cache256b.log 2>/dev/null")
print("verify:\n", out)