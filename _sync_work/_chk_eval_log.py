# -*- coding: utf-8 -*-
"""Check eval log (encoding safe)."""
import subprocess, time
HOST="root@10.176.54.17"; PORT="36430"

r = subprocess.run(["ssh","-o","ConnectTimeout=15","-p",PORT,HOST,
    "tail -10 /root/Workspace/xy/DiT/cpu_eval_ctrl.log 2>/dev/null"],
    capture_output=True,timeout=30)
print(r.stdout.decode("utf-8", errors="replace"))
