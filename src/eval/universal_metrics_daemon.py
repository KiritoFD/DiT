# -*- coding: utf-8 -*-
"""universal_metrics_daemon.py — 递归扫描全部实验系列的通用 eval 指标 daemon.

自动发现并消费两类 marker (无需按系列手工启动):
  * `eval_pending_ctrl_*.json`  → ControlNet base/ctrl 对比指标 (eval_ctrl_metrics_daemon 逻辑)
  * `eval_pending_*.json`       → 预训练单图指标 (eval_metrics_daemon 逻辑)
处理成功后删除 marker 并写 eval_auto_*.json 到对应 checkpoints 目录.

鲁棒性: 绝对路径 / flock 单实例 / 逐项 try-except / 仅成功后删 marker.
启动: tools/eval_supervisor.sh (tmux eval_supervisor, 崩溃自动重启).
"""
import os
import sys
import json
import glob
import time
import fcntl
import traceback

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.eval import eval_ctrl_metrics_daemon as ctrl_d
from src.eval import eval_metrics_daemon as pre_d

ctrl_d.BASE = ROOT            # 修复 ctrl daemon 的相对路径基准
pre_d.BASE = ROOT

SCAN_ROOTS = [os.path.join(ROOT, "5script", "results"), os.path.join(ROOT, "results")]
LOCK = "/tmp/universal_eval_daemon.lock"
POLL = 20

def log(msg):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] [universal] {msg}", flush=True)

def all_ckpt_dirs():
    dirs = []
    for root in SCAN_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, _ in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(('_', '.'))]
            if os.path.basename(dirpath) == "checkpoints":
                dirs.append(dirpath)
    return dirs

def pending_files(ckpt_dir):
    out = []
    for p in glob.glob(os.path.join(ckpt_dir, "eval_pending_ctrl_*.json")):
        out.append(("ctrl", p))
    for p in glob.glob(os.path.join(ckpt_dir, "eval_pending_*.json")):
        b = os.path.basename(p)
        if b.startswith("eval_pending_ctrl_"):
            continue
        # pretrain marker: eval_pending_<step>.json
        if b[len("eval_pending_"):-5].isdigit():
            out.append(("pre", p))
    return out

def main():
    log(f"universal daemon start, scanning {SCAN_ROOTS}")
    while True:
        try:
            for ckpt_dir in all_ckpt_dirs():
                for kind, marker in pending_files(ckpt_dir):
                    try:
                        if kind == "ctrl":
                            ctrl_d.process_pending(marker, ckpt_dir)
                        else:
                            pre_d.process_one(marker, ckpt_dir)
                    except Exception:
                        log(f"ERROR {marker}:\n{traceback.format_exc()}")
                        # 失败不删 marker, 但改名避免热循环
                        try:
                            os.replace(marker, marker + ".failed")
                        except Exception:
                            pass
        except Exception:
            log("scan error:\n" + traceback.format_exc())
        time.sleep(POLL)

if __name__ == "__main__":
    try:
        fh = open(LOCK, "w")
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log("another universal daemon is running — exit")
        sys.exit(0)
    main()
