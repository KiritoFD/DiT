# -*- coding: utf-8 -*-
import subprocess
HOST="root@10.176.54.17"; PORT="36430"
cmd = "tail -3 /root/Workspace/xy/DiT/run_skel_dec.log; cat /root/Workspace/xy/DiT/5script/results/skel_decoder/history.json 2>/dev/null; echo; tail -6 /root/Workspace/xy/DiT/run_skel_d3.log; cat /root/Workspace/xy/DiT/5script/results/skel_decoder_d3/history.json 2>/dev/null"
r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,cmd],
    capture_output=True,text=True,timeout=90)
print(r.stdout if r.returncode==0 else f"rc={r.returncode} {r.stderr[:200]}")