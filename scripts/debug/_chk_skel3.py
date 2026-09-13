# -*- coding: utf-8 -*-
"""查远程 skel 文件数 (count all files)."""
import subprocess
HOST="root@10.176.54.17"; PORT="36430"
r = subprocess.run(["ssh","-o","ConnectTimeout=15","-p",PORT,HOST,
    "ls /root/Workspace/xy/DiT/final_skeleton_d3/ | wc -l; echo '---IMG_COUNT---'; ls /root/Workspace/xy/DiT/final_images/ | wc -l"],
    capture_output=True,text=True,timeout=30, encoding="utf-8", errors="replace")
print(r.stdout if r.returncode==0 else f"rc={r.returncode}")
