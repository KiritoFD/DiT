# -*- coding: utf-8 -*-
"""同步 gen_skel_d3.py + train_skel_decoder.py 到远程, 后台生成 3px skel, 不停当前训练。"""
import subprocess, time
HOST="root@10.176.54.17"; PORT="36430"

def run(cmd, timeout=90):
    for i in range(5):
        r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,cmd],
                           capture_output=True,text=True,timeout=timeout)
        if r.returncode==0: return r.stdout
        time.sleep(2)
    return None

# 同步两个脚本
for f in ["tools/gen_skel_d3.py", "tools/train_skel_decoder.py"]:
    r = subprocess.run(["scp","-o","ConnectTimeout=25","-P",PORT,
        rf"G:\GitHub\DiT\{f}", f"root@10.176.54.17:/root/Workspace/xy/DiT/{f}"],
        capture_output=True,timeout=60)
    print(f"scp {f}: {r.returncode==0}")

# 确认当前 1px 训练还在跑
print(run("pgrep -af train_skel_decoder | head -2; echo '---'; tail -3 /root/Workspace/xy/DiT/run_skel_dec.log"))

# 后台启动 3px skel 生成 (nohup, 不影响训练)
print(run("cd /root/Workspace/xy/DiT && nohup /opt/conda/bin/python tools/gen_skel_d3.py "
          "--in-dir final_skeleton --out-dir final_skeleton_d3 --r 3 --workers 16 "
          "> /tmp/_gen_skel_d3.log 2>&1 & echo GEN_PID=$!"))

time.sleep(10)
print(run("head -5 /tmp/_gen_skel_d3.log; pgrep -af gen_skel_d3 | head -2"))