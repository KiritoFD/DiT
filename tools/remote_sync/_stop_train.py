# -*- coding: utf-8 -*-
"""安全停止所有 train.py（脚本方式避免 shell 自我匹配误杀）。"""
import subprocess, time

for pattern in ["train.py --config exp_s_5script_v3a_glyph_XL",
                "train.py --config exp_s_5script_v3a_glyph_cs"]:
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    pids = [p for p in out.stdout.split() if p]
    if pids:
        print(f"[{pattern}] 找到 PID:", pids)
        for p in pids:
            subprocess.run(["kill", "-TERM", p])
        time.sleep(3)
time.sleep(8)
out = subprocess.run(["pgrep", "-f", "train.py"], capture_output=True, text=True)
alive = [p for p in out.stdout.split() if p]
print("剩余 train.py:", alive or "无")
