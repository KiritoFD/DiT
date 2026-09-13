# -*- coding: utf-8 -*-
import subprocess, time
HOST="root@10.176.54.17"; PORT="36430"
def run(cmd, timeout=60):
    for i in range(6):
        r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,cmd],
                           capture_output=True,text=True,timeout=timeout)
        if r.returncode==0: return r.stdout
        print(f"rc={r.returncode}"); time.sleep(2)
    return None
# 简单版: 杀旧 eval, 用 nohup 重启
out = run("pkill -9 -f auto_eval_pixel 2>/dev/null")
print("kill:", out)
out = run("cd /root/Workspace/xy/DiT && nohup /opt/conda/bin/python tools/auto_eval_pixel.py --results-dir 5script/results/px_s_scratch --device cpu --interval 20 > cpu_eval_px.log 2>&1 &")
print("launch:", out)
time.sleep(5)
out = run("pgrep -af auto_eval_pixel | head -2")
print("verify:", out)