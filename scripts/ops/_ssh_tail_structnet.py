# -*- coding: utf-8 -*-
"""查 structnet 进程 CPU 占用 + 是否在跑 + 完整日志行数."""
import subprocess
HOST = "root@10.176.54.17"; PORT = "36430"
cmd = ("ps aux | grep train_struct_probe | grep -v grep | head -3; "
       "echo '---LOG---'; wc -l /root/Workspace/xy/DiT/run_structnet.log; tail -3 /root/Workspace/xy/DiT/run_structnet.log; "
       "echo '---FREE---'; free -g | head -2")
r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,cmd],
                   capture_output=True, text=True, timeout=60)
print(r.stdout if r.returncode==0 else f"rc={r.returncode} {r.stderr[:200]}")