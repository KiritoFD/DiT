# -*- coding: utf-8 -*-
import subprocess, time
HOST="root@10.176.54.17"; PORT="36430"
for i in range(10):
    try:
        r = subprocess.run(["ssh","-o","ConnectTimeout=15","-p",PORT,HOST,"echo OK"],
            capture_output=True,text=True,timeout=20)
        if r.returncode==0:
            print(f"SSH OK after {i} tries")
            r2 = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,
                "tail -15 /root/Workspace/xy/DiT/run_struct_gpu.log; echo '==='; cat /root/Workspace/xy/DiT/5script/results/struct_decoder_gpu/history.json 2>/dev/null || echo none"],
                capture_output=True,text=True,timeout=30, encoding="utf-8", errors="replace")
            print("LOG:", r2.stdout if r2.returncode==0 else f"rc={r2.returncode}")
            break
    except subprocess.TimeoutExpired:
        print(f"try{i}: timeout")
    time.sleep(5)