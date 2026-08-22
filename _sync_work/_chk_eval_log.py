# -*- coding: utf-8 -*-
import subprocess
HOST="root@10.176.54.17"; PORT="36430"
r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,
    "grep -i 'eval-gpu\\|eval-gpu.*fail\\|Saved checkpoint\\|p_sample' /root/Workspace/xy/DiT/run_px_s.log | tail -10"],
    capture_output=True,text=True,timeout=60, encoding="utf-8", errors="replace")
print(r.stdout if r.returncode==0 else f"rc={r.returncode}")