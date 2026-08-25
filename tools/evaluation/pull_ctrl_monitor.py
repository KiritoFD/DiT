# -*- coding: utf-8 -*-
"""
pull_ctrl_monitor.py — 精简版训练监控，适配 ControlNet 训练。

从远程拉取 ctrl 训练日志 → 解析 → 写 train_data.json → 供 train_dashboard.html 可视化。
ControlNet 训练日志格式: (step=NNNNNNN) loss=X.XXXX | LR: X.XXe-XX | Steps/Sec: X.XX | Mem: XX.XXG

用法:
  python pull_ctrl_monitor.py            # 拉一次
  python pull_ctrl_monitor.py --loop     # 每 --interval 秒循环（默认 30）
"""
import os
import re
import sys
import json
import time
import subprocess
import datetime

REMOTE_USER = "root"
REMOTE_HOST = "10.176.54.17"
REMOTE_PORT = "36430"
REMOTE_BASE = "/root/Workspace/xy/DiT"

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_CUR = os.path.join(HERE, "current_train.log")
OUT_JSON = os.path.join(HERE, "train_data.json")

import sys as _sys
if not any(_ in _sys.path for _ in (os.path.dirname(HERE), HERE)):
    _sys.path.insert(0, os.path.dirname(HERE))
DASH_HTML = os.path.join(HERE, "train_dashboard.html")

# ControlNet 日志行: (step=0000050) loss=0.0302 | LR: 2.71e-04 | Steps/Sec: 4.53 | Mem: 14.08G
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
ROW_RE = re.compile(
    r"\(step=(\d+)\)\s+loss=([\d.]+)\s*\|\s*LR:\s*([\d.eE\-]+)\s*\|\s*"
    r"Steps/Sec:\s*([\d.]+)\s*\|\s*Mem:\s*([\d.]+)G"
)

# eval_auto_*.json 中的 eval 行
EVAL_RE = re.compile(r"\[eval\].*?step\s+(\d+).*?MSE=([\d.]+).*?SSIM=([\d.]+)")


def num(v):
    try:
        return None if v in (None, "nan") else float(v)
    except (ValueError, TypeError):
        return None


def remote_find_latest_log():
    """找 ctrl 训练的最新日志文件 (run_ctrl_top30.log 或 run_ctrl_skel.log)."""
    cmd = ["ssh", "-o", "ConnectTimeout=25", "-o", "ServerAliveInterval=3",
           "-p", REMOTE_PORT, f"{REMOTE_USER}@{REMOTE_HOST}"]
    script = (
        f"ls -t {REMOTE_BASE}/run_ctrl_top30.log "
        f"{REMOTE_BASE}/run_ctrl_skel.log 2>/dev/null | head -1"
    )
    for attempt in range(5):
        try:
            r = subprocess.run(cmd + [script], capture_output=True, text=True, timeout=30)
            if r.returncode == 0:
                paths = [p.strip() for p in r.stdout.splitlines() if p.strip().startswith("/")]
                return paths[0] if paths else ""
        except subprocess.TimeoutExpired:
            pass
        time.sleep(5)
    return ""


def pull_log():
    """scp 远程日志到本地 current_train.log."""
    latest = remote_find_latest_log()
    if not latest:
        return None, False
    src = f"{REMOTE_USER}@{REMOTE_HOST}:{latest}"
    for attempt in range(3):
        try:
            r = subprocess.run(
                ["scp", "-o", "ConnectTimeout=20", "-P", REMOTE_PORT, src, LOCAL_CUR],
                capture_output=True, text=True, timeout=40)
            if r.returncode == 0:
                return latest, True
        except subprocess.TimeoutExpired:
            pass
        time.sleep(3)
    return latest, False


def parse():
    """解析本地 current_train.log → rows."""
    rows = []
    try:
        lines = open(LOCAL_CUR, encoding="utf-8", errors="ignore").read().splitlines()
    except OSError:
        return rows
    for raw in lines:
        line = ANSI_RE.sub("", raw)
        tm = TS_RE.search(line)
        ts = tm.group(1) if tm else None
        m = ROW_RE.search(line)
        if m:
            rows.append({
                "step": int(m.group(1)),
                "loss": float(m.group(2)),
                "lr": float(m.group(3)),
                "stepsPerSec": float(m.group(4)),
                "memCur": float(m.group(5)),
                "memPeak": None,
                "ts": ts,
                "is_eval": False,
                "mse": None,
                "ssim": None,
            })
    # 去重 (按 step)
    seen = {}
    for r in rows:
        seen[r["step"]] = r
    return sorted(seen.values(), key=lambda x: x["step"])


def pull_eval_jsons():
    """拉取远程 eval_auto_*.json, 合并 mse/ssim 进 rows."""
    cmd = ["ssh", "-o", "ConnectTimeout=25", "-o", "ServerAliveInterval=3",
           "-p", REMOTE_PORT, f"{REMOTE_USER}@{REMOTE_HOST}"]
    script = (
        f"for d in {REMOTE_BASE}/5script/results/ctrl_skel/*/checkpoints; do "
        f"  for f in $d/eval_auto_*.json; do "
        f"    if [ -f $f ]; then echo $f; fi; "
        f"  done; "
        f"done 2>/dev/null"
    )
    json_paths = []
    for attempt in range(3):
        try:
            r = subprocess.run(cmd + [script], capture_output=True, text=True, timeout=30)
            if r.returncode == 0:
                json_paths = [p.strip() for p in r.stdout.splitlines() if p.strip()]
                break
        except subprocess.TimeoutExpired:
            pass
        time.sleep(5)

    evals = {}
    for jp in json_paths:
        for attempt in range(2):
            try:
                r = subprocess.run(
                    ["scp", "-o", "ConnectTimeout=15", "-P", REMOTE_PORT,
                     f"{REMOTE_USER}@{REMOTE_HOST}:{jp}", "/tmp/_eval_tmp.json"],
                    capture_output=True, text=True, timeout=20)
                if r.returncode == 0:
                    with open("/tmp/_eval_tmp.json", encoding="utf-8") as f:
                        data = json.load(f)
                    step = data.get("step", 0)
                    evals[step] = {
                        "mse_base": data.get("mse_base"),
                        "ssim_base": data.get("ssim_base"),
                        "mse_ctrl": data.get("mse_ctrl"),
                        "ssim_ctrl": data.get("ssim_ctrl"),
                    }
                    break
            except Exception:
                pass
            time.sleep(2)
    return evals


def merge_evals(rows, evals):
    """把 eval 结果合并到对应 step 的 row."""
    for r in rows:
        if r["step"] in evals:
            ev = evals[r["step"]]
            r["is_eval"] = True
            r["mse"] = ev["mse_ctrl"]
            r["ssim"] = ev["ssim_ctrl"]
            r["mse_base"] = ev["mse_base"]
            r["ssim_base"] = ev["ssim_base"]


def write(rows, source=""):
    """写 train_data.json."""
    last = rows[-1] if rows else None
    data = {
        "source": source,
        "experiment": "ctrl-skel-top30-scratch",
        "rows": rows,
        "last_step": last["step"] if last else 0,
        "last_loss": last["loss"] if last else None,
        "last_lr": last["lr"] if last else None,
        "last_sps": last["stepsPerSec"] if last else None,
        "last_mem": last["memCur"] if last else None,
        "last_ts": last["ts"] if last else None,
        "updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def run_once(verbose=True):
    remote, ok = pull_log()
    if not ok:
        if verbose:
            print(f"[{datetime.datetime.now():%H:%M:%S}] 拉取失败: {remote}")
        return 0
    rows = parse()
    evals = pull_eval_jsons()
    merge_evals(rows, evals)
    write(rows, f"remote:{REMOTE_USER}@{REMOTE_HOST}:{remote}")
    last = rows[-1] if rows else None
    if verbose:
        tail = f"step={last['step']} loss={last['loss']:.4f}" if last else "无数据"
        eval_info = ""
        if evals:
            ev_steps = sorted(evals.keys())
            latest_ev = ev_steps[-1]
            ev = evals[latest_ev]
            eval_info = f" | eval@{latest_ev}: MSE={ev['mse_ctrl']:.4f} SSIM={ev['ssim_ctrl']:.4f}"
        print(f"[{datetime.datetime.now():%H:%M:%S}] rows={len(rows)} {tail}{eval_info}")
    return len(rows)


def main():
    interval = 30
    if "--interval" in sys.argv:
        try:
            interval = int(sys.argv[sys.argv.index("--interval") + 1])
        except Exception:
            pass
    if "--loop" in sys.argv:
        print(f"[pull_ctrl_monitor] loop 每 {interval}s")
        while True:
            t0 = time.time()
            try:
                run_once()
            except Exception as e:
                print(f"[pull_ctrl_monitor] error: {e}")
            time.sleep(max(1, interval - (time.time() - t0)))
    else:
        run_once()


if __name__ == "__main__":
    main()
