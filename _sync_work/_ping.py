# -*- coding: utf-8 -*-
import subprocess, time
HOST="root@10.176.54.17"; PORT="36430"
cmd = "echo PING"
for i in range(10):
    try:
        r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,cmd],
            capture_output=True,text=True,timeout=30)
        print(f"try{i}: rc={r.returncode} out='{r.stdout.strip()}'")
        if r.returncode==0: break
    except subprocess.TimeoutExpired:
        print(f"try{i}: timeout")
    time.sleep(4)