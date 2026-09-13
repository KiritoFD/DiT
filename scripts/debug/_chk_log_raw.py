# -*- coding: utf-8 -*-
"""直接看 50000-55600 区间的 log 行 (不加 grep)."""
import subprocess
HOST="root@10.176.54.17"; PORT="36430"
# 看 step 50000 保存后的几行 + 55000 保存后的几行
r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,
    "sed -n '305,310p' /root/Workspace/xy/DiT/run_px_s.log"],
    capture_output=True,text=True,timeout=60, encoding="utf-8", errors="replace")
print("305-310:", r.stdout if r.returncode==0 else f"rc={r.returncode}")

r2 = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,
    "wc -l /root/Workspace/xy/DiT/run_px_s.log; sed -n '553,560p' /root/Workspace/xy/DiT/run_px_s.log"],
    capture_output=True,text=True,timeout=60, encoding="utf-8", errors="replace")
print("553-560:", r2.stdout if r2.returncode==0 else f"rc={r2.returncode}")