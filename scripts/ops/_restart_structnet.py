# -*- coding: utf-8 -*-
"""同步更新版 structnet 脚本并重启 (log-every=10, 32 线程)。"""
import subprocess, time
HOST = "root@10.176.54.17"; PORT = "36430"

def run(cmd, timeout=60):
    for i in range(5):
        r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,cmd],
                           capture_output=True,text=True,timeout=timeout)
        if r.returncode==0: return r.stdout
        print(f"rc={r.returncode} {r.stderr[:100]}"); time.sleep(2)
    return None

r = subprocess.run(["scp","-o","ConnectTimeout=25","-P",PORT,
    r"G:\GitHub\DiT\tools\train_struct_probe_256.py",
    "root@10.176.54.17:/root/Workspace/xy/DiT/tools/"],
    capture_output=True,timeout=60)
print("scp:", r.returncode==0)

print(run("pkill -9 -f train_struct_probe_256; sleep 2; echo KILLED"))
print(run("cd /root/Workspace/xy/DiT && tmux kill-session -t structnet 2>/dev/null; "
          "tmux new-session -d -s structnet '"
          "/opt/conda/bin/python tools/train_struct_probe_256.py "
          "--csv 5script/train.csv --epochs 8 --batch-size 64 --num-threads 32 "
          "--log-every 10 > run_structnet.log 2>&1' && echo STARTED"))