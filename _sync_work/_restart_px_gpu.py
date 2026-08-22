# -*- coding: utf-8 -*-
"""同步 GPU eval 版 train_pixel.py + 停 CPU eval + 重启 pixel 训练 (resume 45k)."""
import subprocess, time
HOST="root@10.176.54.17"; PORT="36430"

def run(cmd, timeout=90):
    for i in range(6):
        r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,cmd],
                           capture_output=True,text=True,timeout=timeout)
        if r.returncode==0: return r.stdout
        print(f"rc={r.returncode}"); time.sleep(2)
    return None

# scp
r = subprocess.run(["scp","-o","ConnectTimeout=25","-P",PORT,
    r"G:\GitHub\DiT\src\train_pixel.py","root@10.176.54.17:/root/Workspace/xy/DiT/src/"],
    capture_output=True,timeout=60)
print("scp:", r.returncode==0)

# 编译验证 + 停 CPU eval
print(run("cd /root/Workspace/xy/DiT && /opt/conda/bin/python -m py_compile src/train_pixel.py && echo COMPILE_OK"))
print(run("pkill -9 -f auto_eval_pixel 2>/dev/null; sleep 1; echo CPU_EVAL_KILLED"))

# 重启 pixel (从 45k resume, 这次 GPU eval)
print(run("cd /root/Workspace/xy/DiT && tmux kill-session -t exp 2>/dev/null; "
          "tmux new-session -d -s exp '"
          "/opt/conda/bin/python src/train_pixel.py --config exp_px_s_scratch.json > run_px_s.log 2>&1' && echo PX_RESTARTED"))

time.sleep(40)
print(run("tail -6 /root/Workspace/xy/DiT/run_px_s.log"))