# -*- coding: utf-8 -*-
"""查 structnet 训练进度 (log + 历史 metrics + 进程)."""
import subprocess
HOST = "root@10.176.54.17"; PORT = "36430"
cmd = ("echo '=== LOG (last 20) ==='; tail -20 /root/Workspace/xy/DiT/run_structnet.log; "
       "echo '=== HISTORY ==='; cat /root/Workspace/xy/DiT/5script/results/structnet256/history.json 2>/dev/null; "
       "echo '=== CKPT ==='; ls -la /root/Workspace/xy/DiT/5script/results/structnet256/*.pt 2>/dev/null; "
       "echo '=== PROC ==='; ps aux | grep train_struct_probe | grep -v grep | head -2; "
       "echo '=== CPU ==='; top -bn1 | head -5")
r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,cmd],
                   capture_output=True, text=True, timeout=90)
print(r.stdout if r.returncode==0 else f"rc={r.returncode} {r.stderr[:200]}")