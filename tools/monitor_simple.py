# -*- coding: utf-8 -*-
"""精简收敛监控：读本地 pull_monitor 更新的 train_data.json（每 INTERVAL 秒），
打印最新 step/diff/ssim，并在 SSIM 开始收敛或训练前进时输出提示。
用法: python monitor_simple.py [interval_sec]
"""
import json, sys, time, datetime, os

INTERVAL = int(sys.argv[1]) if len(sys.argv) > 1 else 600
JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_data.json")

prev = None
print(f"[monitor] start interval={INTERVAL}s ({JSON})", flush=True)
last_step = -1
stall = 0
while True:
    try:
        d = json.load(open(JSON, encoding="utf-8"))
        last = d["rows"][-1]
        now = datetime.datetime.now().strftime("%H:%M:%S")
        ss = last.get("ssim")
        progress = last["step"] > last_step
        print(f"[{now}] step={last['step']} diff={last.get('diff')} total={last.get('total')} "
              f"stdmid={last.get('stdmid')} ssim={ss}", flush=True)
        if progress:
            last_step = last["step"]
            stall = 0
        else:
            stall += 1
            if stall >= 15:
                print("TRAINING_STALLED: step 未推进", flush=True)
                stall = 0
        if ss is not None and prev is not None and abs(ss - prev) < 0.002:
            print(f"SSIM_CONVERGED: {prev}->{ss}", flush=True)
            # 不退出，继续打点；如需退出可改 sys.exit(0)
    except Exception as e:
        print(f"[monitor] err: {e}", flush=True)
    time.sleep(INTERVAL)
