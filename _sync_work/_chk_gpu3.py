# -*- coding: utf-8 -*-
import subprocess, time
HOST="root@10.176.54.17"; PORT="36430"
for i in range(5):
    try:
        r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,
            "tail -15 /root/Workspace/xy/DiT/run_struct_gpu.log; echo '==='; cat /root/Workspace/xy/DiT/5script/results/struct_decoder_gpu/history.json 2>/dev/null"],
            capture_output=True,text=True,timeout=30, encoding="utf-8", errors="replace")
        if r.returncode==0:
            print(r.stdout); break
    except subprocess.TimeoutExpired:
        print("timeout")
    time.sleep(3)