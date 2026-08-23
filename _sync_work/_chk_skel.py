# -*- coding: utf-8 -*-
"""查远程 skel 目录名."""
import subprocess
HOST="root@10.176.54.17"; PORT="36430"
r = subprocess.run(["ssh","-o","ConnectTimeout=15","-p",PORT,HOST,
    "ls -d /root/Workspace/xy/DiT/final_skele* 2>/dev/null; ls -d /root/Workspace/xy/DiT/final_skeleton* 2>/dev/null; find /root/Workspace/xy/DiT -maxdepth 1 -name 'final_skel*' -type d 2>/dev/null"],
    capture_output=True,text=True,timeout=30, encoding="utf-8", errors="replace")
print(r.stdout if r.returncode==0 else f"rc={r.returncode}")
