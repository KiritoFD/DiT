# -*- coding: utf-8 -*-
import subprocess, time
HOST="root@10.176.54.17"; PORT="36430"
for i in range(8):
    try:
        r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,"echo OK"],
            capture_output=True,text=True,timeout=20, encoding="utf-8", errors="replace")
        if r.returncode==0:
            print(f"try{i}: SSH OK")
            # 现在查 log
            r2 = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,
                "tail -15 /root/Workspace/xy/DiT/run_struct_gpu.log"],
                capture_output=True,text=True,timeout=30, encoding="utf-8", errors="replace")
            print("LOG:", r2.stdout if r2.returncode==0 else f"rc={r2.returncode}")
            r3 = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,
                "cat /root/Workspace/xy/DiT/5script/results/struct_decoder_gpu/history.json 2>/dev/null || echo none"],
                capture_output=True,text=True,timeout=30, encoding="utf-8", errors="replace")
            print("HIST:", r3.stdout if r3.returncode==0 else f"rc={r3.returncode}")
            break
        else:
            print(f"try{i}: rc={r.returncode}")
    except subprocess.TimeoutExpired:
        print(f"try{i}: timeout")
    time.sleep(5)