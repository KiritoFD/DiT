# -*- coding: utf-8 -*-
"""重启 auto_eval_pixel (因 train 重启了)."""
import subprocess, time
HOST="root@10.176.54.17"; PORT="36430"
def run(cmd, timeout=90):
    for i in range(5):
        r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,cmd],
                           capture_output=True,text=True,timeout=timeout)
        if r.returncode==0: return r.stdout
        time.sleep(2)
    return None
print(run("pkill -9 -f auto_eval_pixel 2>/dev/null; sleep 1; "
          "cd /root/Workspace/xy/DiT && "
          "nohup /opt/conda/bin/python tools/auto_eval_pixel.py "
          "--results-dir 5script/results/px_s_scratch "
          "--device cpu --interval 20 > cpu_eval_px.log 2>&1 & echo EVAL_RESTARTED"))