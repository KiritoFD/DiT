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

# 当前训练：XL 甲2 标准字形条件 · 楷隶（exp_v3b_glyphcond.log 最新）。
# 只取 mtime 最新一个日志（当前正在进行的 run），避免误合并旧 run 重复 step。
FIND_CMD = (
    "latest=$(ls -t %(base)s/exp_v3b_glyphcond.log "
    "%(base)s/exp_xl_skelhead_c*.log %(base)s/exp_xl_highdim_cs*.log "
    "%(base)s/exp_v3a_glyph_cs*.log %(base)s/exp_xl_lora_cs.log 2>/dev/null | head -1); "
    "if [ -n \"$latest\" ]; then echo $latest; fi; "
    "ls -t %(base)s/results/v3*/log.txt %(base)s/results/v3*/**/log.txt 2>/dev/null | head -1"
)

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_LOG = os.path.join(HERE, "train_run.log")
LOCAL_LOGS_DIR = os.path.join(HERE, "remote_logs")  # 按远程日志名持久保存，不覆盖
OUT_JSON = os.path.join(HERE, "train_data.json")

# 去掉 ANSI 颜色转义（日志里有 \x1b[34m 之类）
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# 行首时间戳：[2026-08-12 21:48:37]
TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
LINE_RE = re.compile(
    r"\(step=(\d+)\)\s+Total:\s+([\d.nan]+)\s*\|\s*Diff:\s+([\d.nan]+)\s*\|"
    r"\s*Canny:\s+raw\s+([\d.nan]+).*?Skel:\s+raw\s+([\d.nan]+).*?"
    r"REPA:\s+raw\s+([\d.nan]+)"
    r"(?:.*?X0Lat:\s+raw\s+([\d.nan]+))?"
    r".*?Steps/Sec:\s+([\d.nan]+)"
    r"(?:.*?Mem:\s*([\d.]+)G/([\d.]+)G)?"
)
# auto-eval 行：[auto-eval] step 5001: MSE=0.02995 SSIM=0.8860 或 free-sampling 版
AUTOEVAL_RE = re.compile(
    r"\[auto-eval\]\s+step\s+(\d+):\s*(?:free-sampling\s+)?MSE=([\d.nan]+)\s*SSIM=([\d.nan]+)"
)


def to_num(v):
    try:
        if v is None or v == "nan":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def remote_find_log():
    cmd = [
        "ssh", "-o", "ConnectTimeout=10", "-p", REMOTE_PORT,
        f"{REMOTE_USER}@{REMOTE_HOST}", FIND_CMD % {"base": REMOTE_BASE},
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        paths = [p for p in r.stdout.strip().splitlines() if p.strip()]
        # 去重（FIND_CMD 的多个分支可能重复输出同一路径），保持 mtime 顺序
        seen, out = set(), []
        for p in paths:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out
    except Exception:
        return []


# 最新 auto-eval 推理图（训练保存 ckpt 时生成，每次覆盖为 eval_latest.png）
FIND_EVAL_IMG = (
    "latest=$(ls -t %(base)s/5script/results/*/*/checkpoints/eval_latest.png "
    "%(base)s/results/exp_*/*/checkpoints/eval_latest.png "
    "%(base)s/results/dit_s_pretrain/*/checkpoints/eval_latest.png "
    "%(base)s/results/*/eval_latest.png %(base)s/results/*/*/eval_latest.png "
    "2>/dev/null | head -1); echo $latest;"
)
# 最新实验的 eval_samples 目录（含所有 step 的历史取样图）
FIND_EVAL_SAMPLES = (
    "latest=$(ls -td %(base)s/5script/results/*/*/checkpoints/eval_samples 2>/dev/null | head -1); "
    "if [ -z \"$latest\" ]; then "
    "  latest=$(ls -td %(base)s/results/*/*/checkpoints/eval_samples 2>/dev/null | head -1); "
    "fi; echo $latest;"
)
LOCAL_EVAL_IMG = os.path.join(HERE, "eval_latest.png")
LOCAL_EVAL_SAMPLES_DIR = os.path.join(HERE, "remote_eval_samples")


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
    """拉取最新 eval 推理图到本地；成功后本地生成六宫格四联图。
    成功返回 True，否则 False。"""
    remote = remote_find_eval_img()
    if not remote:
        return False
    src = f"{REMOTE_USER}@{REMOTE_HOST}:{remote}"
    cmd = ["scp", "-o", "ConnectTimeout=10", "-P", REMOTE_PORT, src, LOCAL_EVAL_IMG]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return False
    except Exception:
        return False
    # 本地生成六宫格：pred|GT / pred-canny|GT-canny / pred-skel|GT-skel
    try:
        quad = os.path.join(HERE, "make_eval_quad.py")
        subprocess.run([sys.executable, quad, LOCAL_EVAL_IMG],
                       capture_output=True, text=True, timeout=120)
    except Exception as e:
        print(f"  [WARN] 本地六宫格生成失败: {e}")
    return True


def remote_find_eval_samples():
    cmd = [
        "ssh", "-o", "ConnectTimeout=10", "-p", REMOTE_PORT,
        f"{REMOTE_USER}@{REMOTE_HOST}", FIND_EVAL_SAMPLES % {"base": REMOTE_BASE},
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        path = r.stdout.strip().splitlines()[-1].strip() if r.stdout.strip() else ""
        return path
    except Exception:
        return ""


def ensure_show5_gt_once():
    """show5 的 GT 真值（canny/skel）+ show5_eval.csv 固定拉一次，之后不再拉。
    marker 绑定 eval_csv 文件名：实验切换(eval_csv 变)会触发重建，避免 GT 与样本错位。
    """
    # 当前 v3b 实验用 kailishu_eval.csv
    eval_csv = "kailishu_eval.csv"
    marker = os.path.join(HERE, "remote_gt", f".pulled_{eval_csv}")
    show5_csv = os.path.join(HERE, "show5_eval.csv")
    if os.path.exists(marker) and os.path.exists(show5_csv):
        return True  # 该 eval_csv 已拉过，不再重复
    try:
        script = os.path.join(HERE, "pull_show5_gt.py")
        r = subprocess.run([sys.executable, script, eval_csv],
                           capture_output=True, text=True, timeout=120)
        ok = r.returncode == 0
        if ok:
            os.makedirs(os.path.join(HERE, "remote_gt"), exist_ok=True)
            open(marker, "w").close()
            print(f"[gt-pull] show5 GT 已拉取(eval_csv={eval_csv}, 仅此一次)")
        return ok
    except Exception as e:
        print(f"  [WARN] show5 GT 拉取异常: {e}")
        return False


def pull_eval_samples_make_poster():
    """拉取最新实验的 eval_samples 目录（含各 step 完整历史取样图），
    本地生成 poster（一行=一个 ckpt，每样本 img|canny|skel，末行 GT）。
    成功返回 True，否则 False。"""
    remote = remote_find_eval_samples()
    if not remote or not remote.startswith("/"):
        return False
    # 确保 show5 GT 真值已固定拉取（一次性）
    ensure_show5_gt_once()
    src = f"{REMOTE_USER}@{REMOTE_HOST}:{remote}"
    # 拉取 eval_samples 目录内容到本地（合并，不删除旧 step）
    os.makedirs(LOCAL_EVAL_SAMPLES_DIR, exist_ok=True)
    cmd_scp = ["scp", "-o", "ConnectTimeout=15", "-P", REMOTE_PORT,
               "-r", src + "/.", LOCAL_EVAL_SAMPLES_DIR]
    try:
        r = subprocess.run(cmd_scp, capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            return False
    except Exception:
        return False
    # 生成 poster
    poster = os.path.join(HERE, "eval_poster.png")
    gt_dir = os.path.join(HERE, "remote_gt")
    show5_csv = os.path.join(HERE, "show5_eval.csv")
    args = [sys.executable, os.path.join(HERE, "make_eval_poster.py"),
            LOCAL_EVAL_SAMPLES_DIR]
    if os.path.isdir(gt_dir):
        args += ["--gt-dir", gt_dir]
    if os.path.exists(show5_csv):
        args += ["--show5-csv", show5_csv]
    args += ["-o", poster]
    try:
        subprocess.run(args, capture_output=True, text=True, timeout=120)
    except Exception as e:
        print(f"  [WARN] poster 生成失败: {e}")
    return True


def pull_once():
    """拉取一次远程日志并解析。返回 (rows, source_str, used_cache:bool)。

    每个远程日志按 basename（附内容哈希后缀以防不同实验同名）持久保存到
    LOCAL_LOGS_DIR，因此 resume 后新日志、以及历史实验日志都不会互相覆盖。
    解析时读取目录内全部日志按 step 去重合并，得到从 0 开始的完整曲线。
    """
    remote_logs = remote_find_log()
    used_cache = False
    os.makedirs(LOCAL_LOGS_DIR, exist_ok=True)
    if not remote_logs:
        print("  [WARN] 未找到远程日志，使用本地已有副本。")
        used_cache = True
    else:
        for remote_log in remote_logs:
            base = os.path.basename(remote_log)
            local_dst = os.path.join(LOCAL_LOGS_DIR, base)
            src = f"{REMOTE_USER}@{REMOTE_HOST}:{remote_log}"
            cmd = ["scp", "-o", "ConnectTimeout=10", "-P", REMOTE_PORT, src, local_dst]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
                if r.returncode != 0:
                    print(f"  [WARN] scp 失败({r.returncode}) {remote_log}")
                    used_cache = True
            except Exception as e:
                print(f"  [WARN] scp 异常 {remote_log}: {e}")
                used_cache = True

    # 解析目录内全部本地日志
    rows = parse_local_logs()
    src_list = ",".join(remote_logs) if remote_logs else ""
    source = (f"remote:{REMOTE_USER}@{REMOTE_HOST}:{REMOTE_PORT}[{src_list}]"
              if src_list and not used_cache else
              f"{src_list or 'local cache'}" + (" (缓存, 远程拉取失败)" if used_cache else ""))
    return rows, source, used_cache


def parse_local_logs():
    """读取 LOCAL_LOGS_DIR 顶层 .log 文件与旧单文件缓存，按 step 去重合并。离线可用。
    子目录（如 _other_exp/ 归档区）不扫描。"""
    os.makedirs(LOCAL_LOGS_DIR, exist_ok=True)
    log_files = sorted(f for f in os.listdir(LOCAL_LOGS_DIR)
                       if os.path.isfile(os.path.join(LOCAL_LOGS_DIR, f))
                       and (f.endswith(".log") or f == "log.txt"))
    log_files = [os.path.join(LOCAL_LOGS_DIR, f) for f in log_files]
    # 兼容旧的单文件缓存
    if os.path.exists(LOCAL_LOG):
        log_files.append(LOCAL_LOG)

    all_rows = []
    for local in log_files:
        if not os.path.exists(local):
            continue
        try:
            with open(local, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception:
            continue
        for raw in lines:
            line = ANSI_RE.sub("", raw)
            ts_m = TS_RE.search(line)
            ts = ts_m.group(1) if ts_m else None
            m = LINE_RE.search(line)
            if m:
                all_rows.append({
                    "step": int(m.group(1)),
                    "total": to_num(m.group(2)),
                    "diff": to_num(m.group(3)),
                    "canny": to_num(m.group(4)),
                    "skel": to_num(m.group(5)),
                    "repa": to_num(m.group(6)),
                    "x0lat": to_num(m.group(7)),
                    "stepsPerSec": to_num(m.group(8)),
                    "memCur": to_num(m.group(9)),
                    "memPeak": to_num(m.group(10)),
                    "mse": None,
                    "ssim": None,
                    "ts": ts,
                })
                continue
            a = AUTOEVAL_RE.search(line)
            if a:
                all_rows.append({
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

    # 合并：按 (step, 类型) 去重，同 key 保留后出现（更新日志优先）；再按 step 排序
    merged = {}
    for r in all_rows:
        key = (r["step"], "eval" if r["mse"] is not None else "loss")
        merged[key] = r
    rows = sorted(merged.values(), key=lambda r: (r["step"], 0 if r["mse"] is None else 1))
    return rows


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
                                       "repa", "x0lat", "stepsPerSec", "mse", "ssim", "ts")}
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
    local_only = "--local-only" in sys.argv
    interval = 60
    if "--interval" in sys.argv:
        try:
            interval = int(sys.argv[sys.argv.index("--interval") + 1])
        except Exception:
            pass

    if local_only:
        # 纯离线：只解析本地 remote_logs/（已在网络可用时拉回），不访问远程
        rows = parse_local_logs()
        source = f"local offline [{','.join(sorted(os.listdir(LOCAL_LOGS_DIR)))}]"
        write_json(rows, source)
        print(f"[local-only] 解析完成: {len(rows)} 条 -> {OUT_JSON}")
        return

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
        okp = pull_eval_samples_make_poster()
        print(f"[4/4] eval 历史取样海报: {'已更新' if okp else '无/失败'} -> {os.path.join(HERE, 'eval_poster.png')}")
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
            okp = pull_eval_samples_make_poster()
            if okp:
                msg += " | [海报已更新]"
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
