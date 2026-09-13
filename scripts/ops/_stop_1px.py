# -*- coding: utf-8 -*-
"""停掉 1px skel decoder (tmux structnet), 让 CPU 给 3px。"""
import subprocess, time
HOST="root@10.176.54.17"; PORT="36430"

def run(cmd, timeout=60):
    for i in range(5):
        try:
            r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,cmd],
                               capture_output=True,text=True,timeout=timeout)
            if r.returncode==0: return r.stdout
        except subprocess.TimeoutExpired:
            pass
        time.sleep(2)
    return None

# 杀 1px 进程 + 杀 tmux session
print(run("tmux kill-session -t structnet 2>/dev/null; pkill -9 -f 'train_skel_decoder.py.*skel_root.*final_skeleton ' 2>/dev/null; sleep 1; echo KILLED"))
# 确认只剩 3px
print(run("pgrep -af train_skel_decoder | head -4; echo '---'; tmux ls 2>/dev/null"))