# -*- coding: utf-8 -*-
import subprocess, time
HOST="root@10.176.54.17"; PORT="36430"
for i in range(6):
    try:
        r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,
            "tail -15 /root/Workspace/xy/DiT/run_skel_dec.log; echo '---HIST---'; cat /root/Workspace/xy/DiT/5script/results/skel_decoder/history.json 2>/dev/null"],
            capture_output=True,text=True,timeout=90)
        if r.returncode==0:
            print(r.stdout); break
        print(f"rc={r.returncode} {r.stderr[:100]}")
    except subprocess.TimeoutExpired:
        print("[timeout]")
    time.sleep(3)