# -*- coding: utf-8 -*-
import subprocess
HOST="root@10.176.54.17"; PORT="36430"
cmd = ("tail -15 /root/Workspace/xy/DiT/run_skel_dec.log; "
       "echo '---HIST---'; "
       "cat /root/Workspace/xy/DiT/5script/results/skel_decoder/history.json 2>/dev/null; "
       "echo '---PROC---'; pgrep -af train_skel | head -2")
r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,cmd],
    capture_output=True,text=True,timeout=90)
print(r.stdout if r.returncode==0 else f"rc={r.returncode} {r.stderr[:200]}")