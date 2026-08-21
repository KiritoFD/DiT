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
import glob
import subprocess
import datetime

REMOTE_USER = "root"
REMOTE_HOST = "10.176.54.17"
REMOTE_PORT = "36430"
REMOTE_BASE = "/root/Workspace/xy/DiT"

# 当前训练：V3C MIDSTEP · 楷隶。权威日志是实验目录下的 log.txt（FileHandler，
# 无缓冲、最新）；exps/ 下的 *.log 是 stdout 重定向（可能缓冲，不作为首选）。
# 只取 mtime 最新一个日志（当前正在进行的 run），避免误合并旧 run 重复 step。
FIND_CMD = (
    # 0) 当前正在跑的实验：s5_2factor_B_latentstruct*（2026-08-17 起，含 _pixelsk_opt 等变体），第一优先
    "latest=$(ls -t %(base)s/5script/results/s5_2factor_B_latentstruct*/*/log.txt 2>/dev/null | head -1); "
    # 1) 其次：5script/results/v3c_*/<run>/log.txt（每次运行一个，mtime 最新为准）
    "if [ -z \"$latest\" ]; then "
    "latest=$(ls -t %(base)s/5script/results/v3c_xl_glyphcond_midstep/*/log.txt "
    "%(base)s/5script/results/v3b_xl_glyphcond/*/log.txt 2>/dev/null | head -1); "
    "fi; "
    # 2) 其次：顶层 exp_*.log（仅当上面没找到时）
    "if [ -z \"$latest\" ]; then "
    "latest=$(ls -t %(base)s/exp_v3c_midstep.log %(base)s/exp_v3b_glyphcond.log "
    "%(base)s/exp_v3c*.log %(base)s/exp_xl_skelhead_c*.log "
    "%(base)s/exp_xl_highdim_cs*.log "
    "%(base)s/exp_v3a_glyph_cs*.log %(base)s/exp_xl_lora_cs.log 2>/dev/null | head -1); "
    "fi; "
    "if [ -n \"$latest\" ]; then echo $latest; fi; "
    # 3) 兜底：结果目录任意 log.txt
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
# 兼容新旧日志格式：
#  旧(v3c/v3b): (step=..) Total: X | Diff: X | Canny: raw X | Skel: raw X | REPA: raw X | X0Lat: raw X | Steps/Sec: X | Mem: XG/XG
#  新(s5 2factor): ... Canny: raw 0.0000 x 0.00 = 0.0000 | Skel: raw ... | LatC: raw ... | LatS: raw ... | REPA: raw ... | SkelH: raw ... | StdMid: raw ... | X0Lat: raw ... | Steps/Sec: X | Mem: XG/XG
# 每个字段用 .+? 宽松前置（跨过 "x 0.00 = 0.0000" 权重乘式），全部可选以兼容旧格式不存在的字段。
LINE_RE = re.compile(
    r"\(step=(\d+)\)\s+Total:\s+([\d.nan]+)\s*\|\s*Diff:\s+([\d.nan]+)"
    r".+?Canny:\s+raw\s+([\d.nan]+)"
    r".+?Skel:\s+raw\s+([\d.nan]+)"
    r"(?:.+?LatC:\s+raw\s+([\d.nan]+))?"
    r"(?:.+?LatS:\s+raw\s+([\d.nan]+))?"
    r".+?REPA:\s+raw\s+([\d.nan]+)"
    r"(?:.+?SkelH:\s+raw\s+([\d.nan]+))?"
    r"(?:.+?StdMid:\s+raw\s+([\d.nan]+))?"
    r"(?:.+?X0Lat:\s+raw\s+([\d.nan]+))?"
    r".+?Steps/Sec:\s+([\d.nan]+)"
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
LOCAL_SEEN_SAMPLES_DIR = os.path.join(HERE, "remote_seen_samples")
LOCAL_EVAL_JSONS_DIR = os.path.join(HERE, "remote_eval_jsons")


# 最新实验的 checkpoints 目录（含 CPU eval 写的 eval_auto_*.json 指标）
FIND_CKPT_DIR = (
    "latest=$(ls -td %(base)s/5script/results/*/*/checkpoints 2>/dev/null | head -1); "
    "if [ -z \"$latest\" ]; then "
    "  latest=$(ls -td %(base)s/results/*/*/checkpoints 2>/dev/null | head -1); "
    "fi; echo $latest;"
)
# 最新实验的 seen_samples 目录（seen5：训练集样本展示，eval 每个 ckpt 画一次）
FIND_SEEN_SAMPLES = (
    "latest=$(ls -td %(base)s/5script/results/*/*/checkpoints/seen_samples 2>/dev/null | head -1); "
    "if [ -z \"$latest\" ]; then "
    "  latest=$(ls -td %(base)s/results/*/*/checkpoints/seen_samples 2>/dev/null | head -1); "
    "fi; echo $latest;"
)
LOCAL_SEEN_SAMPLES_DIR = os.path.join(HERE, "remote_seen_samples")


def _exp_eval_dir(exp):
    """按实验隔离的本地目录（不同实验的 step 号会重叠，必须分开，避免旧步混入海报）。"""
    return os.path.join(LOCAL_EVAL_SAMPLES_DIR, exp, "eval_samples")


def _exp_seen_dir(exp):
    return os.path.join(LOCAL_SEEN_SAMPLES_DIR, exp, "seen_samples")


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


def remote_find_ckpt_dir():
    cmd = [
        "ssh", "-o", "ConnectTimeout=10", "-p", REMOTE_PORT,
        f"{REMOTE_USER}@{REMOTE_HOST}", FIND_CKPT_DIR % {"base": REMOTE_BASE},
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        path = r.stdout.strip().splitlines()[-1].strip() if r.stdout.strip() else ""
        return path if path.startswith("/") else ""
    except Exception:
        return ""


def remote_find_seen_samples():
    cmd = [
        "ssh", "-o", "ConnectTimeout=10", "-p", REMOTE_PORT,
        f"{REMOTE_USER}@{REMOTE_HOST}", FIND_SEEN_SAMPLES % {"base": REMOTE_BASE},
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        path = r.stdout.strip().splitlines()[-1].strip() if r.stdout.strip() else ""
        return path if path.startswith("/") else ""
    except Exception:
        return ""


def remote_complete_steps(ckpt_dir, sub):
    """远程 <ckpt>/<sub>/ 下已写完 samples.json 的 step 号集合（sub=eval_samples|seen_samples）。
    samples.json 是 eval 画完一个 step 后最后落盘的标记，只有它才算"已完成"。"""
    if not ckpt_dir:
        return set()
    cmd = (
        "ls -1 %s/%s/step*/samples.json 2>/dev/null"
        % (ckpt_dir, sub)
    )
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", "-p", REMOTE_PORT,
             f"{REMOTE_USER}@{REMOTE_HOST}", cmd],
            capture_output=True, text=True, timeout=30)
        steps = set()
        for line in r.stdout.splitlines():
            m = re.search(r"step(\d+)/samples\.json$", line.strip())
            if m:
                steps.add(int(m.group(1)))
        return steps
    except Exception:
        return set()


def local_complete_steps(local_dir):
    """本地 <local_dir>/ 下已写完 samples.json 的 step 号集合。"""
    if not os.path.isdir(local_dir):
        return set()
    out = set()
    for d in glob.glob(os.path.join(local_dir, "step*")):
        if os.path.exists(os.path.join(d, "samples.json")):
            m = re.search(r"step(\d+)", os.path.basename(d))
            if m:
                out.add(int(m.group(1)))
    return out


def _scp_dir_contents(src, local_dir, timeout=300):
    """scp 远程目录内容到本地 local_dir（合并，不删旧）。成功返回 True。"""
    os.makedirs(local_dir, exist_ok=True)
    cmd = ["scp", "-o", "ConnectTimeout=15", "-P", REMOTE_PORT, "-r", src + "/.", local_dir]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0
    except Exception:
        return False


def _prune_incomplete(local_dir):
    """删除本地缺 samples.json 的 step 目录（中断的 scp / eval 未写完的残留），
    保证前端只看到已完成 step，解耦 eval 进行中的写盘与展示。"""
    if not os.path.isdir(local_dir):
        return
    for d in glob.glob(os.path.join(local_dir, "step*")):
        if not os.path.exists(os.path.join(d, "samples.json")):
            try:
                for f in os.listdir(d):
                    os.remove(os.path.join(d, f))
                os.rmdir(d)
            except Exception:
                pass


def local_latest_exp():
    """离线时从本地 remote_eval_samples/<exp>/ 找最新实验名（远程不可达时的 exp 回退）。"""
    if not os.path.isdir(LOCAL_EVAL_SAMPLES_DIR):
        return ""
    try:
        exps = [(os.path.getmtime(os.path.join(LOCAL_EVAL_SAMPLES_DIR, e)), e)
                for e in os.listdir(LOCAL_EVAL_SAMPLES_DIR)
                if os.path.isdir(os.path.join(LOCAL_EVAL_SAMPLES_DIR, e))]
    except Exception:
        return ""
    return max(exps)[1] if exps else ""


def pull_eval_jsons():
    """拉取最新实验 checkpoints/eval_auto_*.json（CPU eval 写的指标）。"""
    ckpt_dir = remote_find_ckpt_dir()
    if not ckpt_dir:
        return False
    os.makedirs(LOCAL_EVAL_JSONS_DIR, exist_ok=True)
    src = f"{REMOTE_USER}@{REMOTE_HOST}:{ckpt_dir}/eval_auto_*.json"
    cmd = ["scp", "-o", "ConnectTimeout=15", "-P", REMOTE_PORT, src, LOCAL_EVAL_JSONS_DIR]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        return r.returncode == 0
    except Exception:
        return False


def parse_local_eval_jsons():
    """解析本地 eval_auto_*.json（CPU eval 的 MSE/SSIM 指标点）。"""
    rows = []
    if not os.path.isdir(LOCAL_EVAL_JSONS_DIR):
        return rows
    for f in sorted(glob.glob(os.path.join(LOCAL_EVAL_JSONS_DIR, "eval_auto_*.json"))):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if "step" not in d:
            continue
        rows.append({
            "step": int(d["step"]),
            "total": None, "diff": None, "canny": None, "skel": None,
            "latc": None, "lats": None, "repa": None, "skelh": None,
            "stdmid": None, "x0lat": None, "stepsPerSec": None,
            "memCur": None, "memPeak": None,
            "mse": to_num(d.get("mse")), "ssim": to_num(d.get("ssim")),
            "ts": None,
        })
    return rows


def merge_eval_jsons(rows):
    """把 CPU eval 的指标点合并进 rows（eval_auto json 优先，覆盖日志行）。"""
    jr = parse_local_eval_jsons()
    if not jr:
        return rows
    merged = {}
    for r in rows:
        merged[(r["step"], "eval" if r.get("mse") is not None else "loss")] = r
    for j in jr:
        merged[(j["step"], "eval")] = j
    return sorted(merged.values(), key=lambda r: (r["step"], 0 if r.get("mse") is None else 1))


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


def pull_poster_data():
    """拉取最新实验的 eval_samples(show5) + seen_samples(seen5) 到本地，
    只拉完整 step，再生成 10×3 海报（文件名带实验名，不互相覆盖）。
    返回 (exp, poster_name, changed)；changed=False 表示没有新 step、无需刷新。"""
    ckpt_dir = remote_find_ckpt_dir()
    if not ckpt_dir:
        return None, None, False
    exp = os.path.basename(os.path.dirname(ckpt_dir.rstrip("/")))
    eval_remote = os.path.join(ckpt_dir, "eval_samples")
    seen_remote = os.path.join(ckpt_dir, "seen_samples")
    local_eval = _exp_eval_dir(exp)
    local_seen = _exp_seen_dir(exp)

    # 快路径：远程完整 step 集合 == 本地 => 无新产出，跳过拉取/生成
    r_eval = remote_complete_steps(ckpt_dir, "eval_samples")
    r_seen = remote_complete_steps(ckpt_dir, "seen_samples")
    l_eval = local_complete_steps(local_eval)
    l_seen = local_complete_steps(local_seen)
    if r_eval == l_eval and r_seen == l_seen:
        poster = os.path.join(HERE, f"eval_poster_{exp}.png")
        return exp, poster if os.path.exists(poster) else None, False
    if not r_eval and not r_seen:
        return exp, None, False

    # 确保 show5 GT 真值已固定拉取（一次性）
    ensure_show5_gt_once()

    ok_e = ok_s = True
    if r_eval:
        ok_e = _scp_dir_contents(f"{REMOTE_USER}@{REMOTE_HOST}:{eval_remote}",
                                 local_eval)
    if r_seen:
        ok_s = _scp_dir_contents(f"{REMOTE_USER}@{REMOTE_HOST}:{seen_remote}",
                                 local_seen)
    if not (ok_e or ok_s):
        return exp, None, True
    _prune_incomplete(local_eval)
    _prune_incomplete(local_seen)

    # 生成 10×3 海报（show5+seen5），文件名带实验名不覆盖
    poster = os.path.join(HERE, f"eval_poster_{exp}.png")
    gt_dir = os.path.join(HERE, "remote_gt")
    args = [sys.executable, os.path.join(HERE, "make_eval_poster.py"),
            "--show5-dir", local_eval,
            "--seen5-dir", local_seen,
            "--exp", exp]
    if os.path.isdir(gt_dir):
        args += ["--gt-dir", gt_dir]
    show5_csv = os.path.join(HERE, "show5_eval.csv")
    if os.path.exists(show5_csv):
        args += ["--show5-csv", show5_csv]
    seen5_csv = os.path.join(HERE, "..", "5script", "seen5_top30.csv")
    if os.path.exists(seen5_csv):
        args += ["--seen5-csv", seen5_csv]
    args += ["-o", poster]
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=180)
        if r.stdout.strip():
            print("  " + r.stdout.strip().splitlines()[-1])
    except Exception as e:
        print(f"  [WARN] poster 生成失败: {e}")
    return exp, poster, True


def pull_once():
    """拉取一次远程日志并解析。返回 (rows, source_str, used_cache:bool)。

    每个远程日志按 basename（附内容哈希后缀以防不同实验同名）持久保存到
    LOCAL_LOGS_DIR。默认只解析**本次拉取的当前日志**(filenames)，避免把
    历史 run(v3b 等) 混进当前 v3c 曲线；离线(拉取失败)时回退到全部本地副本。
    """
    remote_logs = remote_find_log()
    used_cache = False
    fetched = []
    os.makedirs(LOCAL_LOGS_DIR, exist_ok=True)
    if not remote_logs:
        print("  [WARN] 未找到远程日志，使用本地已有副本。")
        used_cache = True
    else:
        for remote_log in remote_logs:
            base = os.path.basename(remote_log)
            fetched.append(base)
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

    # 解析：有成功拉取的当前日志 => 只解析它们(当前 run)；否则回退只解析 log.txt
    # （log.txt 是当前实验日志名；旧实验是 exp_*.log，离线时混入会把旧 step 曲线
    #   并进当前实验，导致前端数据错乱）。
    if fetched and not used_cache:
        rows = parse_local_logs(filenames=fetched)
    else:
        rows = parse_local_logs(filenames=["log.txt"])
    # CPU eval 的指标（eval_auto_*.json），不依赖训练日志里的 [auto-eval] 行
    pull_eval_jsons()
    rows = merge_eval_jsons(rows)
    src_list = ",".join(remote_logs) if remote_logs else ""
    source = (f"remote:{REMOTE_USER}@{REMOTE_HOST}:{REMOTE_PORT}[{src_list}]"
              if src_list and not used_cache else
              f"{src_list or 'local cache'}" + (" (缓存, 远程拉取失败)" if used_cache else ""))
    return rows, source, used_cache


def parse_local_logs(filenames=None):
    """读取 LOCAL_LOGS_DIR 顶层 .log 文件与旧单文件缓存，按 step 去重合并。离线可用。
    子目录（如 _other_exp/ 归档区）不扫描。
    filenames: 若给定(如当前正在拉的日志 basename 列表)，只解析这些文件，
     避免把已结束的历史 run(v3b 等) 混进当前 v3c 曲线。默认 None=解析全部。"""
    os.makedirs(LOCAL_LOGS_DIR, exist_ok=True)
    if filenames:
        # 只解析指定日志文件(可能带子目录名，取 basename 与之匹配)
        log_files = []
        for f in filenames:
            b = os.path.basename(f)
            p = os.path.join(LOCAL_LOGS_DIR, b)
            if os.path.isfile(p):
                log_files.append(p)
            elif os.path.isfile(f):
                log_files.append(f)
        log_files = sorted(log_files)
    else:
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
                    "latc": to_num(m.group(6)),
                    "lats": to_num(m.group(7)),
                    "repa": to_num(m.group(8)),
                    "skelh": to_num(m.group(9)),
                    "stdmid": to_num(m.group(10)),
                    "x0lat": to_num(m.group(11)),
                    "stepsPerSec": to_num(m.group(12)),
                    "memCur": to_num(m.group(13)),
                    "memPeak": to_num(m.group(14)),
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
                    "latc": None,
                    "lats": None,
                    "repa": None,
                    "skelh": None,
                    "stdmid": None,
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


def write_json(rows, source, exp=""):
    rows = prepend_baseline(rows)
    # eval 指标(mse/ssim)只在 eval step 单点出现，其余 loss 行均为 None。
    # 前向填充：把每个 eval 点的值往后传播到后续所有行，使最新一行(last)始终带当前
    # 已算出的 MSE/SSIM，前端 renderStats/图表不再取到 null("抓不到"的根因)。
    rows = forward_fill_eval(rows)
    out = {
        "pulledAt": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "expName": exp,
        "count": len(rows),
        "rows": rows,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def forward_fill_eval(rows):
    """把 mse/ssim 前向填充：遇到新 eval 值则更新，后续行沿用最近一次 eval 值。
    返回新列表，不改入参。保序。"""
    cur_mse = cur_ssim = None
    out = []
    for r in rows:
        nr = dict(r)
        if nr.get("mse") is not None:
            cur_mse = nr["mse"]
        if nr.get("ssim") is not None:
            cur_ssim = nr["ssim"]
        # 若当前行是丢失型的(eval 行 mse 非 None)则保留；否则若我们已知道某个 eval 值，
        # 就把它也填到本行，保证每行都有"当前已知"的评估指标。
        if cur_mse is not None and nr.get("mse") is None:
            nr["mse"] = cur_mse
        if cur_ssim is not None and nr.get("ssim") is None:
            nr["ssim"] = cur_ssim
        out.append(nr)
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
        ckpt_dir = remote_find_ckpt_dir()
        exp = os.path.basename(os.path.dirname(ckpt_dir.rstrip("/"))) if ckpt_dir else local_latest_exp()
        write_json(rows, source, exp)
        print(f"[2/2] 解析完成: {len(rows)} 条记录 -> {OUT_JSON}")
        if rows:
            last = rows[-1]
            print(f"      最新 step={last['step']}  diff={'NaN' if last['diff'] is None else round(last['diff'], 4)}")
        ok = pull_eval_img()
        print(f"[3/3] eval 推理图: {'已更新' if ok else '无/失败'} -> {LOCAL_EVAL_IMG}")
        exp, poster, changed = pull_poster_data()
        print(f"[4/4] eval 历史取样海报: {poster or '无/失败'}" +
              ("" if changed else " (无新 step, 跳过)"))
        return

    print(f"[loop] 每 {interval}s 拉取一次，Ctrl+C 退出。")
    while True:
        t0 = time.time()
        try:
            rows, source, used_cache = pull_once()
            ckpt_dir = remote_find_ckpt_dir()
            exp = os.path.basename(os.path.dirname(ckpt_dir.rstrip("/"))) if ckpt_dir else local_latest_exp()
            write_json(rows, source, exp)
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
            exp2, poster, changed = pull_poster_data()
            if changed and poster:
                msg += f" | [海报已更新: {os.path.basename(poster)}]"
            elif changed:
                msg += " | [海报: 拉取中/失败]"
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
