# -*- coding: utf-8 -*-
import subprocess
HOST="root@10.176.54.17"; PORT="36430"
cmd = "tail -3 /root/Workspace/xy/DiT/run_skel_dec.log"
r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,cmd],
    capture_output=True,text=True,timeout=60)
print(f"rc={r.returncode}")
print(repr(r.stdout))
print(repr(r.stderr[:300]))