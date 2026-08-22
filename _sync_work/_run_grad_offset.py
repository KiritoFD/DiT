# -*- coding: utf-8 -*-
import subprocess
HOST="root@10.176.54.17"; PORT="36430"
r = subprocess.run(["scp","-o","ConnectTimeout=25","-P",PORT,
    r"G:\GitHub\DiT\tools\eval_grad_offset.py",
    "root@10.176.54.17:/root/Workspace/xy/DiT/tools/"],
    capture_output=True,timeout=60)
print("scp:", r.returncode==0)
r2 = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,
    "cd /root/Workspace/xy/DiT && /opt/conda/bin/python tools/eval_grad_offset.py 2>&1 | tail -40"],
    capture_output=True, text=True, timeout=360, encoding="utf-8", errors="replace")
print("RESULT:", r2.stdout if r2.returncode==0 else f"rc={r2.returncode}\n{r2.stderr[:500]}")