#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""精简版训练监控：拉取当前实验训练日志 → 解析 → 写 train_data.json（供 dashboard）。
核心只做三件事，去掉旧的 eval配图/海报/show5 等繁复流程：
  1. 找远程当前实验目录的 log.txt（mtime 最新），scp 到本地 current_train.log
  2. 解析 (step=...) 损失行 + [auto-eval] 行 → rows（mse/ssim 前向填充）
  3. 写 train_data.json

用法:
  python pull_monitor.py            # 拉一次
  python pull_monitor.py --loop     # 每 --interval 秒循环（默认 60）
"""
import os, re, sys, json, time, glob
import subprocess, datetime

REMOTE_USER = "root"
REMOTE_HOST = "10.176.54.17"
REMOTE_PORT = "36430"
REMOTE_BASE = "/root/Workspace/xy/DiT"

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_CUR = os.path.join(HERE, "current_train.log")   # 最新的当前日志副本
OUT_JSON = os.path.join(HERE, "train_data.json")

# 找远程当前 log.txt（优先 5script/results 下实验目录，mtime 最新）
FIND_CMD = (
    "find %(base)s/5script/results -name log.txt 2>/dev/null "
    "| xargs ls -t 2>/dev/null | head -1; "
    "echo; ls -t %(base)s/exp_*.log %(base)s/exp_s2*.log 2>/dev/null | head -1"
)
LANRE = re.compile(r"\x1b\[[0-9;]*m")
# 训练损失行：用分栏切分更稳，避免长串 .*? 贪婪问题
ROW_RE = re.compile(r"\(step=(\d+)\)\s+(.*)")
EVAL_RE = re.compile(
    r"\[auto-eval\]\s+step\s+(\d+):\s*(?:free-sampling\s+)?MSE=([\d.nan]+)\s*SSIM=([\d.nan]+)"
)
TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")


def num(v):
    try:
        return None if v in (None, "nan") else float(v)
    except (ValueError, TypeError):
        return None


def remote_find_log():
    cmd = ["ssh", "-o", "ConnectTimeout=10", "-p", REMOTE_PORT,
           f"{REMOTE_USER}@{REMOTE_HOST}"]
    try:
        r = subprocess.run(cmd + [FIND_CMD % {"base": REMOTE_BASE}],
                           capture_output=True, text=True, timeout=30)
        paths = [p for p in r.stdout.splitlines() if p.strip() and p.startswith("/")]
        if paths:
            return paths[0]
    except Exception:
        pass
    return ""


def pull_log():
    """拉取当前日志到本地 current_train.log。返回 (last_modified_ts, ok)。"""
    remote = remote_find_log()
    if not remote:
        return None, False
    src = f"{REMOTE_USER}@{REMOTE_HOST}:{remote}"
    try:
        r = subprocess.run(
            ["scp", "-o", "ConnectTimeout=10", "-P", REMOTE_PORT, src, LOCAL_CUR],
            capture_output=True, text=True, timeout=90)
        if r.returncode != 0:
            return remote, False
        return remote, True
    except Exception:
        return remote, False


def parse():
    """解析本地 current_train.log → rows（已前向填充 mse/ssim）。"""
    rows = []
    cur_mse = cur_ssim = None
    try:
        lines = open(LOCAL_CUR, encoding="utf-8", errors="ignore").read().splitlines()
    except OSError:
        return rows
    for raw in lines:
        line = LANRE.sub("", raw)
        tm = TS_RE.search(line)
        ts = tm.group(1) if tm else None
        m = ROW_RE.search(line)
        if m:
            step = int(m.group(1))
            # 按 '|' 分栏解析各字段（Total/Diff/StdMid/X0Lat/Steps/Sec/Mem）
            fields = {}
            for seg in m.group(2).split("|"):
                seg = seg.strip()
                kv = re.match(r"([A-Za-z0-9 /]+?):\s*(.*)$", seg)
                if not kv:
                    continue
                key = kv.group(1).strip()
                if key.startswith("Steps/Sec"):
                    key = "stepsPerSec"; val = kv.group(2).strip()
                elif key == "Mem":
                    mm = re.match(r"([\d.]+)G/([\d.]+)G", kv.group(2))
                    fields["memCur"] = num(mm.group(1) if mm else None)
                    fields["memPeak"] = num(mm.group(2) if mm else None)
                    continue
                else:
                    val = re.sub(r"^raw\s+", "", kv.group(2)).strip()
                fields[key] = num(val)
            rows.append({
                "step": step, "total": fields.get("Total"), "diff": fields.get("Diff"),
                "stdmid": fields.get("StdMid"), "x0lat": fields.get("X0Lat"),
                "stepsPerSec": fields.get("stepsPerSec"),
                "memCur": fields.get("memCur"), "memPeak": fields.get("memPeak"),
                "mse": cur_mse, "ssim": cur_ssim, "ts": ts,
            })
            continue
        e = EVAL_RE.search(line)
        if e:
            # 独立 eval 行（也存）；同时更新前向填充值供后续 loss 行携带
            cur_mse = num(e.group(2)); cur_ssim = num(e.group(3))
            rows.append({
                "step": int(e.group(1)), "total": None, "diff": None,
                "stdmid": None, "x0lat": None, "stepsPerSec": None,
                "memCur": None, "memPeak": None,
                "mse": cur_mse, "ssim": cur_ssim, "ts": ts,
            })
    # 去重保序（同 step 损失行取最后，eval 行若与损失行同 step 后者覆盖）
    seen, out = set(), []
    for r in rows:
        if r["step"] in seen:
            continue
        seen.add(r["step"])
        out.append(r)
    out.sort(key=lambda r: r["step"])
    return out


def write(rows, source):
    out = {
        "source": source, "pulledAt": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "count": len(rows), "rows": rows,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


# ---- 海报：发现新 auto-eval step 时才重新拉取取样图并生成海报 ----
_last_eval_step = {"v": -1}


def _max_eval_step(rows):
    return max((r["step"] for r in rows if r.get("mse") is not None), default=-1)


def _find_remote_eval_samples():
    cmd = ["ssh", "-o", "ConnectTimeout=15", "-p", REMOTE_PORT,
           f"{REMOTE_USER}@{REMOTE_HOST}"]
    find = ("cd %s; D=$(ls -td 5script/results/%s/*/ 2>/dev/null | head -1); "
            "echo \"${D}checkpoints/eval_samples\"" % (REMOTE_BASE, "*"))
    try:
        r = subprocess.run(cmd + [find], capture_output=True, text=True, timeout=30)
        return r.stdout.strip().splitlines()[0].strip() if r.stdout.strip() else ""
    except Exception:
        return ""


def regen_poster_if_new_eval(rows, verbose=True):
    """有新 eval 步才更新海报（拉 eval_samples + 生成 eval_poster.png）。"""
    max_step = _max_eval_step(rows)
    if max_step <= _last_eval_step["v"]:
        return
    es_dir = os.path.join(HERE, "remote_eval_samples")
    os.makedirs(es_dir, exist_ok=True)
    remote = _find_remote_eval_samples()
    if not remote:
        return
    # 拉取该实验的 eval_samples（覆盖旧 run，避免混 step）
    _rmtree(es_dir)
    os.makedirs(es_dir, exist_ok=True)
    try:
        subprocess.run(["scp", "-o", "ConnectTimeout=15", "-P", REMOTE_PORT,
                        "-r", f"{REMOTE_USER}@{REMOTE_HOST}:{remote}/.", es_dir + "/"],
                       capture_output=True, text=True, timeout=180)
    except Exception as e:
        if verbose:
            print(f"[poster] scp 取样图失败: {e}")
        return
    try:
        args = [sys.executable, os.path.join(HERE, "make_eval_poster.py"), es_dir,
                "--gt-dir", os.path.join(HERE, "remote_gt"),
                "--show5-csv", os.path.join(HERE, "show5_eval.csv"),
                "-o", os.path.join(HERE, "eval_poster.png")]
        subprocess.run(args, capture_output=True, text=True, timeout=180)
        if verbose:
            print(f"[poster] 已更新到 eval step {max_step}")
    except Exception as e:
        if verbose:
            print(f"[poster] 海报生成失败: {e}")
    _last_eval_step["v"] = max_step


def _rmtree(path):
    import shutil
    shutil.rmtree(path, ignore_errors=True)


def run_once(verbose=True, regen_poster=True):
    remote, ok = pull_log()
    if not ok:
        if verbose:
            print(f"[{datetime.datetime.now():%H:%M:%S}] 拉取失败: {remote}")
        return 0
    rows = parse()
    write(rows, f"remote:{REMOTE_USER}@{REMOTE_HOST}:{remote}")
    if regen_poster:
        try:
            regen_poster_if_new_eval(rows, verbose)
        except Exception:
            pass
    last = rows[-1] if rows else None
    if verbose:
        tail = f"step={last['step']} diff={last['diff']} total={last['total']} stdmid={last['stdmid']}" if last else "无数据"
        print(f"[{datetime.datetime.now():%H:%M:%S}] rows={len(rows)} {tail}")
    return len(rows)


def main():
    interval = 60
    if "--interval" in sys.argv:
        try:
            interval = int(sys.argv[sys.argv.index("--interval") + 1])
        except Exception:
            pass
    if "--loop" in sys.argv:
        print(f"[pull_monitor] loop 每 {interval}s")
        while True:
            t0 = time.time()
            try:
                run_once()
            except Exception as e:
                print(f"[pull_monitor] error: {e}")
            time.sleep(max(1, interval - (time.time() - t0)))
    else:
        run_once()


if __name__ == "__main__":
    main()
