# -*- coding: utf-8 -*-
import subprocess
HOST="root@10.176.54.17"; PORT="36430"
r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,
    "grep -i 'eval-gpu\\|fail\\|Error\\|Trace' /root/Workspace/xy/DiT/run_px_s.log | tail -10"],
    capture_output=True,text=True,timeout=60, encoding="utf-8", errors="replace")
print("EVAL ERR:", r.stdout if r.returncode==0 else f"rc={r.returncode}")
# 看是否有 eval json 在新 ckpt 目录
r2 = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,
    "ls -la /root/Workspace/xy/DiT/5script/results/px_s_scratch/20260822-145511-px-s-scratch-diff/checkpoints/eval_auto_* 2>/dev/null; echo 'none?'"],
    capture_output=True,text=True,timeout=60, encoding="utf-8", errors="replace")
print("EVAL JSON:", r2.stdout)