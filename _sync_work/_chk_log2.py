# -*- coding: utf-8 -*-
"""查 run_px_s.log 从 50000 ckpt 开始的全部内容."""
import subprocess
HOST="root@10.176.54.17"; PORT="36430"
r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,
    "grep -n 'step=0050000\\|step=00500\\|eval\\|Saved\\|Trace\\|Error\\|warning' /root/Workspace/xy/DiT/run_px_s.log | tail -20"],
    capture_output=True,text=True,timeout=60, encoding="utf-8", errors="replace")
print(r.stdout if r.returncode==0 else f"rc={r.returncode}")