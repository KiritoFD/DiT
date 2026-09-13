# -*- coding: utf-8 -*-
"""同步 eval 脚本 + 远程执行."""
import subprocess, time
HOST="root@10.176.54.17"; PORT="36430"

# scp
r = subprocess.run(["scp","-o","ConnectTimeout=25","-P",PORT,
    r"G:\GitHub\DiT\tools\eval_struct_decoder.py",
    "root@10.176.54.17:/root/Workspace/xy/DiT/tools/"],
    capture_output=True,timeout=60)
print("scp:", r.returncode==0)

# 运行 (前台, 等 5 分钟)
r2 = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,
    "cd /root/Workspace/xy/DiT && /opt/conda/bin/python tools/eval_struct_decoder.py 2>&1 | tail -60"],
    capture_output=True, text=True, timeout=360, encoding="utf-8", errors="replace")
print("RESULT:", r2.stdout if r2.returncode==0 else f"rc={r2.returncode}\n{r2.stderr[:500]}")