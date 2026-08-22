#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sync.py —— 定时自动 pull（增量）+ build 静态 dashboard，只同步清单里的实验。

默认每 60 秒执行一次，只对 tools/active_exps.json 里的实验：
    pull_all.py           # 按 manifest 增量 scp 远程最新日志/eval
    build_dashboards.py   # 重新生成 tools/dashboards/<exp>.html + index.html

清单管理:
    python pull_all.py --set-active s8_structv2_b8all   # 设为当前实验
    python pull_all.py --set-active s8_structv2_b8all s5_xxx   # 多个

用法:
    python sync.py                 # 默认 60s，前台运行，Ctrl-C 退出
    python sync.py --interval 120  # 每 120 秒
    python sync.py --once          # 只跑一轮
    python sync.py --all           # 临时同步全部实验（一轮）
"""
import os
import sys
import time
import json
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "active_exps.json")
INTERVAL = 60


def active_list():
    if not os.path.isfile(MANIFEST):
        return []
    try:
        return json.load(open(MANIFEST, encoding="utf-8")).get("active", [])
    except Exception:
        return []


def run(cmd):
    print(f"[{time.strftime('%H:%M:%S')}] $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        print(f"  [warn] 返回码 {r.returncode}")
    return r.returncode


def sync_once(all_mode=False, extra=None):
    extra = list(extra or [])
    if all_mode:
        extra = ["--all"]
    run([sys.executable, "pull_all.py"] + extra)
    run([sys.executable, "build_dashboards.py"] + (["--all"] if all_mode else extra))


def main():
    interval = INTERVAL
    once = "--once" in sys.argv
    all_mode = "--all" in sys.argv
    exp_args = []
    if "--interval" in sys.argv:
        interval = int(sys.argv[sys.argv.index("--interval") + 1])
    if "--exp" in sys.argv:
        exp_args = ["--exp", sys.argv[sys.argv.index("--exp") + 1]]

    active = active_list() if not all_mode else []
    if not all_mode and not active and not exp_args:
        print("清单为空（tools/active_exps.json 无 active）。")
        print("  - 设置: python pull_all.py --set-active s8_structv2_b8all")
        print("  - 或临时全量: python sync.py --all")
        return

    if once:
        sync_once(all_mode, exp_args)
        return

    print(f"sync 启动，间隔 {interval}s，active={active or '(--all)'}，Ctrl-C 退出")
    try:
        while True:
            sync_once(all_mode, exp_args)
            print(f"[{time.strftime('%H:%M:%S')}] 下次同步 {interval}s 后")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n已停止 sync。")


if __name__ == "__main__":
    main()
