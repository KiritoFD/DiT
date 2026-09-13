# -*- coding: utf-8 -*-
"""停当前 skel 训练, 加 3px 膨胀, 调 pos_weight, 重启。"""
import subprocess, time
HOST="root@10.176.54.17"; PORT="36430"

def run(cmd, timeout=90):
    for i in range(5):
        r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,cmd],
                           capture_output=True,text=True,timeout=timeout)
        if r.returncode==0: return r.stdout
        time.sleep(2)
    return None

# 停当前
print(run("pkill -9 -f train_skel_decoder; sleep 2; echo KILLED"))