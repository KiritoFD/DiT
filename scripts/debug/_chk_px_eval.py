# -*- coding: utf-8 -*-
import subprocess
HOST="root@10.176.54.17"; PORT="36430"
# 查 log 里有没有 eval 相关的行 (grep eval)
r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,
    "grep -i 'eval' /root/Workspace/xy/DiT/run_px_s.log | tail -10"],
    capture_output=True,text=True,timeout=60)
print("EVAL LINES:", r.stdout if r.returncode==0 else f"rc={r.returncode}")
# 查 50000 ckpt 目录下有没有 eval json
r2 = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,
    "ls -la /root/Workspace/xy/DiT/5script/results/px_s_scratch/20260822-145511-px-s-scratch-diff/checkpoints/eval_auto_* 2>/dev/null; echo '---'; ls /root/Workspace/xy/DiT/5script/results/px_s_scratch/20260822-145511-px-s-scratch-diff/checkpoints/"],
    capture_output=True,text=True,timeout=60)
print("CKPT DIR:", r2.stdout if r2.returncode==0 else f"rc={r2.returncode}")