# -*- coding: utf-8 -*-
"""s20 训练监控快照：进程 / GPU / loss / eval 指标 / daemon 状态。

    python tools/monitor_s20.py
"""
import glob
import io
import json
import os
import re
import subprocess
import time

RD = "5script/results/s20_midcommon_s_flow_v2"
LOG = "/tmp/s20_train.log"
DL = "/tmp/s20_eval_daemon.log"


def main():
    print("=" * 78)
    print("S20 监控快照  ", time.strftime("%H:%M:%S"))
    print("=" * 78)

    # --- 进程 ---
    r = subprocess.run(["ps", "-eo", "pid,etime,pcpu,rss,args", "--no-headers"],
                       capture_output=True, text=True)
    print("\n[进程]")
    found = False
    for ln in r.stdout.splitlines():
        if ("train.py" in ln or "eval_metrics_daemon" in ln) and "grep" not in ln:
            p = ln.split(None, 4)
            if len(p) < 5:
                continue
            found = True
            print(f"   pid={p[0]:>8s} up={p[1]:>10s} cpu={p[2]:>5s}% "
                  f"rss={int(p[3])/1e6:6.2f}GB")
            print(f"        {p[4][:100]}")
    if not found:
        print("   (无 train.py / daemon 进程)")

    # --- GPU ---
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True)
    print("\n[GPU]")
    for ln in r.stdout.strip().splitlines():
        u, t, g, tp = [x.strip() for x in ln.split(",")]
        print(f"   mem {u:>6s}/{t:>6s} MiB ({100*float(u)/float(t):5.1f}%)  "
              f"util {g:>3s}%  temp {tp}C")

    # --- train log ---
    if os.path.exists(LOG):
        with io.open(LOG, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        mtime = time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(LOG)))
        steps = [l for l in lines if "step=" in l and "Steps/Sec" in l]
        print(f"\n[train log] {len(lines)} 行, 最后写入 {mtime}")
        if steps:
            last = steps[-1]

            def g(pat, s=last):
                mm = re.search(pat, s)
                return mm.group(1) if mm else "?"

            p_step = r"step=(\d+)"
            p_loss = r"Total: ([\d.]+)"
            p_lr = r"LR: ([\d.e+-]+)"
            p_spd = r"Steps/Sec: ([\d.]+)"
            p_mu = r"Mem: ([\d.]+)G"
            p_mt = r"Mem: [\d.]+G/([\d.]+)G"
            print("   step={}  loss={}  lr={}  speed={}/s  mem={}/{}G".format(
                g(p_step), g(p_loss), g(p_lr), g(p_spd), g(p_mu), g(p_mt)))
            vals = []
            for x in steps[-20:]:
                mm = re.search(r"Total: ([\d.]+)", x)
                if mm:
                    vals.append(float(mm.group(1)))
            if vals:
                print(f"   最近 {len(vals)} 条 loss: 首 {vals[0]:.4f} "
                      f"末 {vals[-1]:.4f} 均值 {sum(vals)/len(vals):.4f} "
                      f"最小 {min(vals):.4f}")
        errs = [l for l in lines[-500:]
                if re.search(r"Error|Traceback|error", l)]
        if errs:
            print(f"   !! 最近 {len(errs)} 条含 error:")
            for e in errs[-3:]:
                print("      ", e.strip()[:140])

    # --- eval ---
    print(f"\n[eval 结果] {RD}")
    if os.path.isdir(RD):
        runs = sorted(glob.glob(os.path.join(RD, "*")))
        for run in runs[-2:]:
            evs = sorted(glob.glob(os.path.join(run, "eval_auto_*.json")))
            cks = sorted(glob.glob(os.path.join(run, "checkpoints", "*")))
            print(f"   {os.path.basename(run)}: ckpt={len(cks)} eval={len(evs)}")
            for p in evs[-10:]:
                try:
                    d = json.load(io.open(p, "r", encoding="utf-8"))
                except Exception:
                    continue
                keys = [k for k in ("ssim", "mse", "lpips", "skel_iou")
                        if isinstance(d.get(k), (int, float))]
                print(f"      step={d.get('step','?'):>7}  " +
                      "  ".join(f"{k}={d[k]:.4f}" for k in keys))
    else:
        print("   (尚未创建)")

    # --- daemon ---
    if os.path.exists(DL):
        with io.open(DL, "r", encoding="utf-8", errors="replace") as f:
            dls = f.readlines()
        print(f"\n[daemon log] {len(dls)} 行")
        for l in dls[-5:]:
            print("   ", l.rstrip()[:130])


if __name__ == "__main__":
    main()
