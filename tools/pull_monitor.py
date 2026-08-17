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

# 当前实验的数据集规模与 batch（用于算"数据过几遍"与"一遍需几小时"）
# 按 exp_s5_2factor_top30: train_top30=128842 样本, batch=192
DATASET_SIZE = 128842
BATCH_SIZE = 192

# 找当前实验的结果目录（5script/results 下，mtime 最新 -> 但取该系列目录：同一 experiment 名的多次 run 要拼接）
FIND_CMD = (
    "find %(base)s/5script/results -name log.txt 2>/dev/null "
    "| xargs ls -t 2>/dev/null | head -1; "
    "echo; ls -t %(base)s/exp_*.log %(base)s/exp_s2*.log 2>/dev/null | head -1"
)
# 找到最新 log.txt 后，取其"实验系列目录"（上级的上级），把该系列所有 log.txt cat 合并（支持 resume 拼接 0->N）
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


def remote_find_latest():
    """找最新 log.txt 的绝对路径（用于识别实验系列目录）。"""
    cmd = ["ssh", "-o", "ConnectTimeout=10", "-p", REMOTE_PORT,
           f"{REMOTE_USER}@{REMOTE_HOST}"]
    try:
        r = subprocess.run(cmd + [FIND_CMD % {"base": REMOTE_BASE}],
                           capture_output=True, text=True, timeout=30)
        paths = [p for p in r.stdout.splitlines() if p.strip() and p.startswith("/")]
        return paths[0] if paths else ""
    except Exception:
        return ""


def pull_log():
    """拉取当前实验系列的**所有** log.txt（cat 合并）到本地 current_train.log。
    这样 resume run 能拼接前一个 run 的曲线（step 0 -> N -> 继续），parse 按 step 去重。"""
    latest = remote_find_latest()
    if not latest:
        return None, False
    # 实验系列目录 = log.txt 所在 run 目录的上级（含同一实验名的多个 run）
    # latest 形如 .../results/<exp>/<run>/log.txt -> 系列目录 = .../results/<exp>
    series_dir = "/".join(latest.split("/")[:-2])
    cmd = ["ssh", "-o", "ConnectTimeout=15", "-p", REMOTE_PORT,
           f"{REMOTE_USER}@{REMOTE_HOST}"]
    merge = (f"find {series_dir} -name log.txt 2>/dev/null | sort "
             f"| xargs cat > /tmp/_cur_train.log 2>/dev/null; "
             f"echo {series_dir}")
    try:
        r = subprocess.run(cmd + [merge], capture_output=True, text=True, timeout=60)
        series = r.stdout.splitlines()[0].strip() if r.stdout else ""
    except Exception:
        return latest, False
    src = f"{REMOTE_USER}@{REMOTE_HOST}:/tmp/_cur_train.log"
    try:
        rr = subprocess.run(
            ["scp", "-o", "ConnectTimeout=15", "-P", REMOTE_PORT, src, LOCAL_CUR],
            capture_output=True, text=True, timeout=90)
        if rr.returncode == 0:
            return latest, True
    except Exception:
        pass
    return latest, False


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
                val_str = kv.group(2).strip()
                if key.startswith("Steps/Sec"):
                    key = "stepsPerSec"; fields[key] = num(val_str.split()[0] if val_str else None)
                elif key == "Mem":
                    mm = re.match(r"([\d.]+)G/([\d.]+)G", val_str)
                    fields["memCur"] = num(mm.group(1) if mm else None)
                    fields["memPeak"] = num(mm.group(2) if mm else None)
                    continue
                else:
                    # value 形如 "raw 0.1423 x 0.50 = 0.0712" -> 取第一个数值(raw)
                    nm = re.search(r"-?[\d.]+", val_str)
                    fields[key] = num(nm.group(0) if nm else None)
            rows.append({
                "step": step, "total": fields.get("Total"), "diff": fields.get("Diff"),
                "canny": fields.get("Canny"), "skel": fields.get("Skel"),
                "latc": fields.get("LatC"), "lats": fields.get("LatS"),
                "stdmid": fields.get("StdMid"), "x0lat": fields.get("X0Lat"),
                "stepsPerSec": fields.get("stepsPerSec"),
                "memCur": fields.get("memCur"), "memPeak": fields.get("memPeak"),
                "mse": cur_mse, "ssim": cur_ssim, "ts": ts, "is_eval": False,
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
                "mse": cur_mse, "ssim": cur_ssim, "ts": ts, "is_eval": True,
            })
    # 去重保序：同 step 优先保留真实 eval 行(is_eval)，否则保留后出现的损失行(带最新前向填充)
    seen, out = {}, []
    for r in rows:
        s = r["step"]
        if s in seen:
            if r.get("is_eval") and not out[seen[s]].get("is_eval"):
                out[seen[s]] = r  # eval 行覆盖同 step 的损失行
            continue
        seen[s] = len(out)
        out.append(r)
    out.sort(key=lambda r: r["step"])
    return out


def write(rows, source):
    # 每 row 附 epoch（已过数据遍数 = step*batch/数据集大小），顶层给最新 epoch + 每遍耗时
    steps_per_epoch = max(1, int(DATASET_SIZE // BATCH_SIZE) + (1 if DATASET_SIZE % BATCH_SIZE else 0))
    out_rows = []
    latest_sps = None
    for r in rows:
        nr = dict(r)
        nr["epoch"] = (r["step"] * BATCH_SIZE) / DATASET_SIZE
        if r.get("stepsPerSec"):
            latest_sps = r["stepsPerSec"]
        out_rows.append(nr)
    sps = latest_sps or 4.0
    sec_per_epoch = DATASET_SIZE / (sps * BATCH_SIZE)
    last_step = out_rows[-1]["step"] if out_rows else 0
    total_steps = 200000  # 当前配置 max_steps（供前端算剩余训练 ETA）
    steps_per_epoch = steps_per_epoch
    eta_hours = max(0, total_steps - last_step) / (sps * 3600) if sps else 0
    out = {
        "source": source, "pulledAt": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "count": len(rows), "rows": out_rows,
        "dataset_size": DATASET_SIZE, "batch": BATCH_SIZE, "steps_per_epoch": steps_per_epoch,
        "epoch_now": (last_step * BATCH_SIZE) / DATASET_SIZE,
        "sec_per_epoch": sec_per_epoch, "hr_per_epoch": sec_per_epoch / 3600.0,
        "stepsPerSec": sps, "total_steps": total_steps, "eta_hours": eta_hours,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


# ---- 海报：按实验隔离 + 增量追加。每个实验开始 => 归档旧海报、清空取样缓存、
#      重新开始；之后每次出现新 auto-eval step => 只拉该 step 取样图并重新生成海报，
#      旧 step 行保留(行 = step，时间升序向下)。----
STATE_F = os.path.join(HERE, "poster_state.json")
ARCHIVE_DIR = os.path.join(HERE, "poster_archive")
LOCAL_ES = os.path.join(HERE, "remote_eval_samples")
POSTER = os.path.join(HERE, "eval_poster.png")


def _experiment_key(remote_log):
    """实验身份 = 实验系列目录名（results/<exp>），而非 run 目录。
    resume run（同 <exp> 新 run_dir）共享同一 key → 海报跨 run 累积，不 reset。"""
    exp = os.path.basename(os.path.dirname(os.path.dirname(remote_log.rstrip("/"))))
    return exp


def _load_state():
    try:
        return json.load(open(STATE_F, encoding="utf-8")) if os.path.exists(STATE_F) else {}
    except Exception:
        return {}


def _save_state(s):
    try:
        json.dump(s, open(STATE_F, "w", encoding="utf-8"))
    except Exception:
        pass


def _rmtree(path):
    import shutil
    shutil.rmtree(path, ignore_errors=True)


def _new_archive_dir(exp):
    """归档目录：poster_archive/<exp>/，静态留存该实验的海报/html/数据/取样图。"""
    d = os.path.join(ARCHIVE_DIR, exp.replace("/", "__"))
    os.makedirs(d, exist_ok=True)
    return d


def _archive_exp(exp, verbose=True):
    """归档上一实验的产物（New 实验开始时调用，静态留存，live 文件不受影响）：
    - html/data json：复制到 post_archive/<exp>/ 作静态快照（live 的仍在 tools/ 供服务）
    - 旧海报 eval_poster.png：移动（新实验将覆盖该文件名）
    - 旧 eval 取样图：复制归档
    """
    dest = _new_archive_dir(exp)
    import shutil
    for src, name in ((os.path.join(HERE, "train_dashboard.html"), "train_dashboard.html"),
                      (os.path.join(HERE, "train_data.json"), "train_data.json")):
        if os.path.exists(src):
            shutil.copy(src, os.path.join(dest, name))
    if os.path.exists(POSTER):
        shutil.move(POSTER, os.path.join(dest, "eval_poster.png"))
    if os.path.isdir(LOCAL_ES):
        shutil.rmtree(os.path.join(dest, "eval_samples"), ignore_errors=True)
        shutil.copytree(LOCAL_ES, os.path.join(dest, "eval_samples"))
    if verbose:
        print(f"[poster] 旧实验 {exp} 已归档到 {dest}")


def _reset_experiment(exp, verbose=True):
    """新实验开始：把上一实验的海报/html/数据归档静态留存，清空取样缓存重新开始。"""
    state = _load_state()
    old_exp = state.get("experiment")
    if old_exp and old_exp != exp:
        _archive_exp(old_exp, verbose)
    _rmtree(LOCAL_ES)
    os.makedirs(LOCAL_ES, exist_ok=True)
    _save_state({"experiment": exp, "done_steps": []})
    if verbose:
        print(f"[poster] 新实验 {exp} 开始（旧实验已归档）")
    _rmtree(LOCAL_ES)
    os.makedirs(LOCAL_ES, exist_ok=True)
    _save_state({"experiment": exp, "done_steps": []})
    if verbose:
        print(f"[poster] 新实验 {exp}: 归档旧海报, 重置取样缓存")


def _existing_steps():
    try:
        return sorted(int(d.name.replace("step", "")) for d in os.listdir(LOCAL_ES)
                      if d.startswith("step"))
    except Exception:
        return []


def _pull_step_stepdir(remote_base, step, verbose=True):
    """从远程 eval_samples/<stepNNNNNNN> 拉取该 step 的取样图到本地。"""
    import glob as _glob
    # 远程 eval_samples 目录在给定实验的 checkpoints/eval_samples 下
    src_dir = "%s/checkpoints/eval_samples/step%07d" % (remote_base, step)
    local_dir = os.path.join(LOCAL_ES, "step%07d" % step)
    os.makedirs(local_dir, exist_ok=True)
    try:
        r = subprocess.run(
            ["scp", "-o", "ConnectTimeout=15", "-P", REMOTE_PORT, "-r",
             f"{REMOTE_USER}@{REMOTE_HOST}:{src_dir}/.", local_dir + "/"],
            capture_output=True, text=True, timeout=180)
        return r.returncode == 0
    except Exception:
        return False


def regen_poster_if_new_eval(rows, remote_log, verbose=True):
    """实验感知 + 增量：实验切换则 reset；新 eval step 则拉取样图并重生成海报。"""
    if not remote_log:
        return
    exp = _experiment_key(remote_log)
    state = _load_state()
    if state.get("experiment") != exp:
        _reset_experiment(exp, verbose)
        state = _load_state()

    # 当前已处理过的 step：state 记录 + 本地已存在取样目录（防止 state 被意外清空时重拉）
    done = set(state.get("done_steps", [])) | set(_existing_steps())
    # 新出现的真实 eval 步（is_eval=True，即 [auto-eval] 行；不是前向填充的损失行）
    new_steps = sorted(r["step"] for r in rows
                       if r.get("is_eval") and r["step"] not in done)
    if not new_steps:
        return  # 没有新 eval，海报不用动

    remote_base = "/".join(remote_log.split("/")[:-1])  # 实验目录绝对路径（去掉 log.txt）
    # 增量拉取本地还没有的 step（只有真拿到该 step 取样图才记 done，否则下轮重试）
    have = set(_existing_steps())
    really_done = set(done)
    for s in new_steps:
        if s not in have:
            ok_pull = _pull_step_stepdir(remote_base, s, verbose)
            # pull 后确认本地确实出现该 step 目录（内容非空）
            sd = os.path.join(LOCAL_ES, "step%07d" % s)
            if ok_pull and os.path.isdir(sd) and os.listdir(sd):
                really_done.add(s)
        else:
            really_done.add(s)
    # 重新生成（make_eval_poster 读 LOCAL_ES 全部 step => 行追加）
    args = [sys.executable, os.path.join(HERE, "make_eval_poster.py"), LOCAL_ES,
            "--gt-dir", os.path.join(HERE, "remote_gt"),
            "--show5-csv", os.path.join(HERE, "eval5_top30.csv"),
            "-o", POSTER]
    try:
        subprocess.run(args, capture_output=True, text=True, timeout=180)
        _save_state({"experiment": exp,
                     "done_steps": sorted(really_done)})
        if verbose:
            print(f"[poster] 已更新 {exp}，含 eval steps {sorted(really_done)}")
    except Exception as e:
        if verbose:
            print(f"[poster] 海报生成失败: {e}")


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
            regen_poster_if_new_eval(rows, remote, verbose)
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
