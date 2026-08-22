# -*- coding: utf-8 -*-
"""看 50500-55000 之间 log 里有没有 eval-gpu 行."""
import subprocess
HOST="root@10.176.54.17"; PORT="36430"
r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,
    "sed -n '305,560p' /root/Workspace/xy/DiT/run_px_s.log | grep -i 'eval\\|fail\\|error\\|warn\\|p_sample'"],
    capture_output=True,text=True,timeout=60, encoding="utf-8", errors="replace")
print(r.stdout if r.returncode==0 else f"rc={r.returncode}")