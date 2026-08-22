# -*- coding: utf-8 -*-
import subprocess, time
HOST="root@10.176.54.17"; PORT="36430"
# 简单命令先测连通
for i in range(8):
    try:
        r = subprocess.run(["ssh","-o","ConnectTimeout=15","-p",PORT,HOST,"echo OK"],
            capture_output=True,text=True,timeout=30)
        print(f"try{i}: rc={r.returncode} out='{r.stdout.strip()}' err='{r.stderr.strip()[:100]}'")
        if r.returncode==0: break
    except subprocess.TimeoutExpired:
        print(f"try{i}: timeout")
    time.sleep(5)