#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从远程定期拉取最新训练日志，解析为 train_data.json，供 train_dashboard.html 可视化。

日志位于远程 results/<最新实验目录>/log.txt（DiT 训练脚本把日志写到实验目录）。
脚本自动挑选 mtime 最新的 log.txt 下载。

用法:
  python pull_log.py            # 拉取一次
  python pull_log.py --loop     # 守护模式，每 --interval 秒拉一次（默认 60）
  python pull_log.py --interval 30
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

# 当前训练：DiT-3Cond-S/2 预训练，log 定向到 dit_s_pretrain_train.log
# results 目录：results/dit_s_pretrain
# 优先：dit_s_pretrain_train.log（当前预训练实验）
# 回退：results 下最新 log.txt
FIND_CMD = (
    "latest=$(ls -t %(base)s/exp_*.log 2>/dev/null | head -1); "
    "if [ -n \"$latest\" ]; then echo $latest; "
    "elif [ -f %(base)s/dit_s_pretrain_train.log ]; then "
    "  echo %(base)s/dit_s_pretrain_train.log; "
    "else "
    "  latest=$(ls -t %(base)s/results/*/log.txt "
    "%(base)s/results/dit_s_pretrain/*/log.txt 2>/dev/null | head -1); "
    "  if [ -z \"$latest\" ]; then "
    "    latest=$(ls -t %(base)s/train_run.log %(base)s/train_auto.log 2>/dev/null | head -1); "
    "  fi; "
    "  echo $latest; "
    "fi"
)

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_LOG = os.path.join(HERE, "train_run.log")
OUT_JSON = os.path.join(HERE, "train_data.json")

# 去掉 ANSI 颜色转义（日志里有 \x1b[34m 之类）
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# 行首时间戳：[2026-08-12 21:48:37]
TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
LINE_RE = re.compile(
    r"\(step=(\d+)\)\s+Total:\s+([\d.nan]+)\s*\|\s*Diff:\s+([\d.nan]+)\s*\|"
    r"\s*Canny:\s+raw\s+([\d.nan]+).*?Skel:\s+raw\s+([\d.nan]+).*?"
    r"REPA:\s+raw\s+([\d.nan]+).*?Steps/Sec:\s+([\d.nan]+)"
    r"(?:.*?Mem:\s*([\d.]+)G/([\d.]+)G)?"
)
# auto-eval 行（ckpt_every=1 模式下主要输出）：[auto-eval] step 5001: MSE=0.02995 SSIM=0.8860
AUTOEVAL_RE = re.compile(
    r"\[auto-eval\]\s+step\s+(\d+):\s*MSE=([\d.nan]+)\s*SSIM=([\d.nan]+)"
)


def to_num(v):
    return None if v == "nan" else float(v)


def remote_find_log():
    cmd = [
        "ssh", "-o", "ConnectTimeout=10", "-p", REMOTE_PORT,
        f"{REMOTE_USER}@{REMOTE_HOST}", FIND_CMD % {"base": REMOTE_BASE},
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        path = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        return path
    except Exception:
        return ""


# 最新 auto-eval 推理图（训练保存 ckpt 时生成，每次覆盖为 eval_latest.png）
FIND_EVAL_IMG = (
    "latest=$(ls -t %(base)s/results/exp_*/*/checkpoints/eval_latest.png "
    "%(base)s/results/dit_s_pretrain/*/checkpoints/eval_latest.png "
    "%(base)s/results/*/eval_latest.png %(base)s/results/*/*/eval_latest.png "
    "2>/dev/null | head -1); echo $latest;"
)
LOCAL_EVAL_IMG = os.path.join(HERE, "eval_latest.png")


def remote_find_eval_img():
    cmd = [
        "ssh", "-o", "ConnectTimeout=10", "-p", REMOTE_PORT,
        f"{REMOTE_USER}@{REMOTE_HOST}", FIND_EVAL_IMG % {"base": REMOTE_BASE},
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        path = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        return path
    except Exception:
        return ""


def pull_eval_img():
    """拉取最新 eval 推理图到本地；成功返回 True，否则 False。"""
    remote = remote_find_eval_img()
    if not remote:
        return False
    src = f"{REMOTE_USER}@{REMOTE_HOST}:{remote}"
    cmd = ["scp", "-o", "ConnectTimeout=10", "-P", REMOTE_PORT, src, LOCAL_EVAL_IMG]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return r.returncode == 0
    except Exception:
        return False


def pull_once():
    """拉取一次远程日志并解析。返回 (rows, source_str, used_cache:bool)。"""
    remote_log = remote_find_log()
    used_cache = False
    if not remote_log:
        print("  [WARN] 未找到远程日志，尝试本地已有副本。")
        used_cache = True
    else:
        src = f"{REMOTE_USER}@{REMOTE_HOST}:{remote_log}"
        cmd = ["scp", "-o", "ConnectTimeout=10", "-P", REMOTE_PORT, src, LOCAL_LOG]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                print(f"  [WARN] scp 失败({r.returncode}): {r.stderr.strip() or r.stdout.strip()}")
                print("         尝试使用已有本地副本（若存在）。")
                used_cache = True
        except Exception as e:
            print(f"  [WARN] scp 异常: {e}")
            used_cache = True

    rows = []
    if os.path.exists(LOCAL_LOG):
        with open(LOCAL_LOG, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = ANSI_RE.sub("", raw)
                ts_m = TS_RE.search(line)
                ts = ts_m.group(1) if ts_m else None
                m = LINE_RE.search(line)
                if m:
                    rows.append({
                        "step": int(m.group(1)),
                        "total": to_num(m.group(2)),
                        "diff": to_num(m.group(3)),
                        "canny": to_num(m.group(4)),
                        "skel": to_num(m.group(5)),
                        "repa": to_num(m.group(6)),
                        "stepsPerSec": to_num(m.group(7)),
                        "memCur": to_num(m.group(8)),
                        "memPeak": to_num(m.group(9)),
                        "mse": None,
                        "ssim": None,
                        "ts": ts,
                    })
                    continue
                a = AUTOEVAL_RE.search(line)
                if a:
                    rows.append({
                        "step": int(a.group(1)),
                        "total": None,
                        "diff": None,
                        "canny": None,
                        "skel": None,
                        "repa": None,
                        "stepsPerSec": None,
                        "memCur": None,
                        "memPeak": None,
                        "mse": to_num(a.group(2)),
                        "ssim": to_num(a.group(3)),
                        "ts": ts,
                    })
                    continue

    source = (f"remote:{REMOTE_USER}@{REMOTE_HOST}:{REMOTE_PORT}{remote_log}"
              if remote_log and not used_cache else
              f"{remote_log or 'local cache'}" + (" (缓存, 远程拉取失败)" if used_cache else ""))
    return rows, source, used_cache


def prepend_baseline(rows):
    """若存在 eval_baseline.json，则把 step=0 基模基准点固定 prepend 到最前。"""
    bp = os.path.join(HERE, "eval_baseline.json")
    if not os.path.exists(bp):
        return rows
    try:
        b = json.load(open(bp, encoding="utf-8"))
        if not b.get("mse") or not b.get("ssim"):
            return rows
        base_row = {k: None for k in ("step", "total", "diff", "canny", "skel",
                                       "repa", "stepsPerSec", "mse", "ssim", "ts")}
        base_row.update({"step": int(b.get("step", 0)),
                         "mse": float(b["mse"]),
                         "ssim": float(b["ssim"]),
                         "ts": None})
        # 避免重复 prepend：若已存在 step==0 的 auto 行则跳过
        if any(r.get("step") == base_row["step"] and r.get("mse") is not None for r in rows):
            return rows
        return [base_row] + rows
    except Exception:
        return rows


def write_json(rows, source):
    rows = prepend_baseline(rows)
    out = {
        "pulledAt": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "count": len(rows),
        "rows": rows,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def main():
    loop = "--loop" in sys.argv
    interval = 60
    if "--interval" in sys.argv:
        try:
            interval = int(sys.argv[sys.argv.index("--interval") + 1])
        except Exception:
            pass

    if not loop:
        print("[1/2] 拉取远程日志 ...")
        rows, source, _ = pull_once()
        write_json(rows, source)
        print(f"[2/2] 解析完成: {len(rows)} 条记录 -> {OUT_JSON}")
        if rows:
            last = rows[-1]
            print(f"      最新 step={last['step']}  diff={'NaN' if last['diff'] is None else round(last['diff'], 4)}")
        ok = pull_eval_img()
        print(f"[3/3] eval 推理图: {'已更新' if ok else '无/失败'} -> {LOCAL_EVAL_IMG}")
        return

    print(f"[loop] 每 {interval}s 拉取一次，Ctrl+C 退出。")
    while True:
        t0 = time.time()
        try:
            rows, source, used_cache = pull_once()
            write_json(rows, source)
            now = datetime.datetime.now().strftime("%H:%M:%S")
            if rows:
                last = rows[-1]
                msg = (f"[{now}] 拉取 OK | {len(rows)} 条 | step={last['step']} "
                       f"diff={'NaN' if last['diff'] is None else round(last['diff'], 4)}")
            else:
                msg = f"[{now}] 拉取 OK | 0 条（无匹配日志行）"
            if used_cache:
                msg += "  [远程拉取失败, 用缓存]"
            ok_img = pull_eval_img()
            if ok_img:
                msg += " | [图已更新]"
            print(msg)
        except Exception as e:
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 拉取异常: {e}")
        # 精确对齐间隔
        elapsed = time.time() - t0
        sleep_for = max(1, interval - elapsed)
        try:
            time.sleep(sleep_for)
        except KeyboardInterrupt:
            print("\n[loop] 已停止。")
            break


if __name__ == "__main__":
    main()
