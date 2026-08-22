# -*- coding: utf-8 -*-
"""v3c 训练收敛监控: 每 INTERVAL 秒 ssh 远程拉一次 train 日志的
StdMid/Diff/Total/X0Lat 与最新 auto-eval(SSIM/MSE), 输出到 stdout(供 job_output)。
退出条件: 检测到 max_steps 结束 或 SSIM 收敛(增量<阈值) 或训练进程消失。
用法: python monitor_v3c.py [interval_sec]
"""
import subprocess, sys, time, datetime, re

INTERVAL = int(sys.argv[1]) if len(sys.argv) > 1 else 1200  # 默认20分钟
SSH = ["ssh", "-o", "ConnectTimeout=20", "-o", "BatchMode=yes", "-p", "36430",
       "root@10.176.54.17"]
REMOTE_CMD = ('cd /root/Workspace/xy/DiT; '
              'grep -E "StdMid" exp_v3c_midstep.log | tail -1; '
              'grep -E "auto-eval" exp_v3c_midstep.log | tail -2; '
              'ps aux | grep -cE "[t]rain.py"; '
              'tail -1 exp_v3c_midstep.log')

def run():
    try:
        r = subprocess.run(SSH + [REMOTE_CMD], capture_output=True, text=True, timeout=90)
        return r.stdout
    except Exception as e:
        return f"ERR {e}"

def parse_stdmid(out):
    m = re.search(r"step=(\d+)", out)
    sm = re.search(r"StdMid: raw ([\d.e]+)", out)
    d = re.search(r"Diff: ([\d.e]+)", out)
    # 进程数: grep -c 输出独立一行(仅数字); 取 "(step=...)\n...\nN\n" 里最后的纯数字行
    proc = re.search(r"^\s*(\d+)\s*$", out, re.MULTILINE)
    return (int(m.group(1)) if m else None,
            float(sm.group(1)) if sm else None,
            float(d.group(1)) if d else None,
            int(proc.group(1)) if proc else None)

prev_ssim = None
print(f"[monitor_v3c] start, interval={INTERVAL}s", flush=True)
while True:
    out = run()
    step, stdmid, diff, procs = parse_stdmid(out)
    ev = re.findall(r"auto-eval\] step (\d+): free-sampling MSE=([\d.]+) SSIM=([\d.]+)", out)
    now = datetime.datetime.now().strftime("%H:%M:%S")
    evline = "; ".join(f"s{m[0]}:SSIM={m[2]}" for m in ev)
    print(f"[{now}] step={step} StdMid={stdmid} Diff={diff} procs={procs} eval[{evline}]", flush=True)
    # 收敛判定: 有 auto-eval 且连续两次变化 < 0.002
    if ev:
        latest = float(ev[-1][2])
        if prev_ssim is not None and abs(latest - prev_ssim) < 0.002:
            print(f"CONVERGED: SSIM {prev_ssim}->{latest} 增量<0.002", flush=True)
            sys.exit(0)
        prev_ssim = latest
    # 训练结束判定
    if procs == 0:
        print("TRAINING_ENDED (no train.py process)", flush=True)
        sys.exit(0)
    time.sleep(INTERVAL)
