# -*- coding: utf-8 -*-
"""同步修复版 train_pixel.py + config, 重启 pixel 训练 (resume from 45000)."""
import subprocess, time
HOST="root@10.176.54.17"; PORT="36430"

def run(cmd, timeout=90):
    for i in range(5):
        r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,cmd],
                           capture_output=True,text=True,timeout=timeout)
        if r.returncode==0: return r.stdout
        time.sleep(2)
    return None

# scp 修复版
for f, remote in [
    (r"G:\GitHub\DiT\src\train_pixel.py", "/root/Workspace/xy/DiT/src/"),
    (r"G:\GitHub\DiT\exp_px_s_scratch.json", "/root/Workspace/xy/DiT/"),
]:
    r = subprocess.run(["scp","-o","ConnectTimeout=25","-P",PORT,f,
        f"root@10.176.54.17:{remote}"], capture_output=True,timeout=60)
    print(f"scp {f.split(chr(92))[-1]}: {r.returncode==0}")

# 编译验证
print(run("cd /root/Workspace/xy/DiT && /opt/conda/bin/python -m py_compile src/train_pixel.py && echo OK"))

# 重启 tmux
print(run("cd /root/Workspace/xy/DiT && tmux kill-session -t exp 2>/dev/null; "
          "tmux new-session -d -s exp '"
          "/opt/conda/bin/python src/train_pixel.py --config exp_px_s_scratch.json > run_px_s.log 2>&1' && echo PX_RESUMED"))

time.sleep(40)
print(run("tail -5 /root/Workspace/xy/DiT/run_px_s.log"))