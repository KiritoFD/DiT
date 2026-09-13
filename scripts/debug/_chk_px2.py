# -*- coding: utf-8 -*-
import subprocess
HOST="root@10.176.54.17"; PORT="36430"
# 查 pixel train log 最后几行 (不含 eval 错误)
r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,
    "tail -6 /root/Workspace/xy/DiT/run_px_s.log; echo '==='; pgrep -af train_pixel | head -2; echo '==='; nvidia-smi --query-gpu=memory.used --format=csv,noheader"],
    capture_output=True,text=True,timeout=60)
print(r.stdout if r.returncode==0 else f"rc={r.returncode}")