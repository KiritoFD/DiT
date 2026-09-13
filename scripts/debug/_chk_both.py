# -*- coding: utf-8 -*-
import subprocess, time
HOST="root@10.176.54.17"; PORT="36430"
cmd = ("echo '=== SKEL 1px ==='; tail -3 /root/Workspace/xy/DiT/run_skel_dec.log; "
       "cat /root/Workspace/xy/DiT/5script/results/skel_decoder/history.json 2>/dev/null; "
       "echo; echo '=== SKEL 3px ==='; tail -8 /root/Workspace/xy/DiT/run_skel_d3.log; "
       "cat /root/Workspace/xy/DiT/5script/results/skel_decoder_d3/history.json 2>/dev/null")
for i in range(6):
    try:
        r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,cmd],
            capture_output=True,text=True,timeout=90)
        if r.returncode==0:
            print(r.stdout); break
        print(f"rc={r.returncode}")
    except subprocess.TimeoutExpired:
        print("[timeout]")
    time.sleep(3)