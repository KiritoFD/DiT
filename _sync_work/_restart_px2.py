# -*- coding: utf-8 -*-
import subprocess, time
HOST="root@10.176.54.17"; PORT="36430"

def run(cmd, timeout=120):
    for i in range(6):
        try:
            r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,cmd],
                               capture_output=True,text=True,timeout=timeout)
            if r.returncode==0: return r.stdout
            print(f"rc={r.returncode}")
        except subprocess.TimeoutExpired:
            print("[timeout]")
        time.sleep(3)
    return None

# 1) 杀旧进程 (分离命令)
print("kill:", run("pkill -9 -f auto_eval_pixel 2>/dev/null; pkill -9 -f 'src/train_pixel' 2>/dev/null; sleep 2; echo K", timeout=30))
# 2) 启动 tmux (简单命令)
print("launch:", run("cd /root/Workspace/xy/DiT && tmux kill-session -t exp 2>/dev/null; tmux new-session -d -s exp '/opt/conda/bin/python src/train_pixel.py --config exp_px_s_scratch.json > run_px_s.log 2>&1' && echo R", timeout=30))
time.sleep(40)
print("log:", run("tail -6 /root/Workspace/xy/DiT/run_px_s.log", timeout=30))