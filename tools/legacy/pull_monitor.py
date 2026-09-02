#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练监控：拉取远程日志+eval → 写 train_data.json + 生成 eval poster + 更新 dashboard。

每轮只用 2 次 SSH 连接（1 ssh 收集元数据 + 1 scp 拉文件），避免 sshd 限流。

用法:
  python pull_monitor.py            # 拉一次
  python pull_monitor.py --loop    # 每 --interval 秒循环（默认 120）
"""
import os, re, sys, json, time, glob, io, tarfile
import subprocess, datetime, sys

REMOTE_USER = "root"
REMOTE_HOST = "10.176.54.17"
REMOTE_PORT = "36430"
REMOTE_BASE = "/root/Workspace/xy/DiT"

HERE = os.path.dirname(os.path.abspath(__file__))
# 本地实验名覆盖: 用于 poster/dashboard 文件命名 (不从远程路径推断)
# 设为 None 则自动从远程 source 路径推断
EXP_NAME = "s10_b4_grey_clear"

def _exp_tag_from_source(source_path=None):
    """返回实验标识。优先用 EXP_NAME 覆盖, 否则从远程路径推断。"""
    if EXP_NAME:
        return EXP_NAME
    if not source_path:
        return "unknown"
    parts = source_path.replace("\\", "/").rstrip("/").split("/")
    if len(parts) >= 3:
        return parts[-3]
    return "unknown"

LOCAL_CUR = os.path.join(HERE, "current_train.log")
OUT_JSON = os.path.join(HERE, "train_data.json")
LOCAL_ES = os.path.join(HERE, "remote_eval_samples")
LOCAL_SEEN = os.path.join(HERE, "remote_seen_samples")
LOCAL_EVAL_JSON = os.path.join(HERE, "remote_eval_auto.json")
STATE_F = os.path.join(HERE, "poster_state.json")

DATASET_SIZE = 104751
BATCH_SIZE = 96
TOTAL_STEPS = 600000

LANRE = re.compile(r"\x1b\[[0-9;]*m")
ROW_RE = re.compile(r"\(step=(\d+)\)\s+(.*)")
EVAL_RE = re.compile(
    r"\[(?:auto-eval|eval)\]\s+step\s+(\d+):\s*"
    r"(?:free-sampling\s+)?MSE=([\d.nan]+)\s*SSIM=([\d.nan]+)"
)
TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")


def num(v):
    try:
        return None if v in (None, "nan") else float(v)
    except (ValueError, TypeError):
        return None


def _ssh(remote_cmd, timeout=30):
    """单次 SSH 调用，返回 stdout 字符串。ssh 用 -p (小写) 端口。"""
    cmd = ["ssh", "-o", "ConnectTimeout=15", "-p", REMOTE_PORT,
           f"{REMOTE_USER}@{REMOTE_HOST}", remote_cmd]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else ""
    except Exception as e:
        print(f"[ssh] error: {e}", file=sys.stderr)
        return ""


def _scp_dir(remote_path, local_dir, timeout=120):
    """scp -r 一个远程目录到本地。scp 用 -P (大写) 端口。"""
    os.makedirs(local_dir, exist_ok=True)
    try:
        r = subprocess.run(
            ["scp", "-o", "ConnectTimeout=15", "-P", REMOTE_PORT, "-r",
             f"{REMOTE_USER}@{REMOTE_HOST}:{remote_path}", local_dir + "/"],
            capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0
    except Exception:
        return False


def collect_remote():
    """一次 SSH 调用完成所有远程信息收集：
    1. 找最新 log.txt 路径
    2. cat 该实验系列所有 log.txt
    3. cat 所有 eval_auto_*.json
    4. ls eval_samples/ 和 seen_samples/ 目录列表
    返回 (latest_log_path, log_content, eval_json_content, ckpt_dir)"""
    # Step 1: 找最新 log.txt
    latest = _ssh(
        f"find {REMOTE_BASE}/5script/results -name log.txt 2>/dev/null "
        f"| xargs ls -t 2>/dev/null | head -1", timeout=20)
    latest = latest.strip()
    if not latest:
        return None, None, None, None, "", None, [], []

    # 实验目录 = log.txt 的上级目录 (去掉 /log.txt)
    run_dir = "/".join(latest.split("/")[:-1])
    series_dir = "/".join(latest.split("/")[:-2])  # results/<exp>
    ckpt_dir = f"{run_dir}/checkpoints"

    # Step 2: 一次性 cat log.txt + cat cpu_eval_state.json + cat eval_auto_*.json
    # + ls eval_samples + ls seen_samples
    # cpu_eval_state.json 包含 eval 时间戳，用于时间轴对齐
    # 注意: 只读当前 run_dir 的 log，不再 cat 整个 series（避免 s7 旧 eval 污染 s8）
    # 关键修正: EVAL_LOG 不再读全局 auto_eval_cpu.log (里面混有旧 s7 的 eval 行)，
    # 只读 s8 自己的 ckpt_dir/cpu_eval_state.json + eval_auto_*.json (它们是 per-ckpt 的)
    SEP = "===_DSH_SEP_==="
    script = (
        f"echo '{SEP}LOG'; "
        f"cat {run_dir}/log.txt 2>/dev/null; "
        f"echo '{SEP}EVAL'; "
        f"cat {series_dir}/*/checkpoints/eval_auto_*.json 2>/dev/null; "
        f"echo '{SEP}EVAL_STATE'; "
        f"cat {ckpt_dir}/gpu_eval_state.json 2>/dev/null; "
        f"cat {ckpt_dir}/cpu_eval_state.json 2>/dev/null; "
        f"echo '{SEP}EVAL_SAMPLES'; "
        f"ls {ckpt_dir}/eval_samples/ 2>/dev/null; "
        f"echo '{SEP}SEEN_SAMPLES'; "
        f"ls {ckpt_dir}/seen_samples/ 2>/dev/null; "
        f"echo '{SEP}END'"
    )
    combined = _ssh(script, timeout=60)

    # 解析分隔段: 每段格式为 "LABEL\ncontent"
    # strip 开头的 label 行 (LOG/EVAL/EVAL_STATE/EVAL_SAMPLES/SEEN_SAMPLES/END)
    parts = combined.split(SEP)
    log_content = ""
    eval_json = ""
    eval_state_json = ""
    eval_log_lines = ""
    eval_sample_steps = []
    seen_sample_steps = []

    def _strip_label(s):
        """Remove the first line if it's a known section label."""
        lines = s.strip().splitlines()
        if lines and lines[0].strip() in ("LOG", "EVAL", "EVAL_STATE",
                                          "EVAL_SAMPLES", "SEEN_SAMPLES", "END"):
            return "\n".join(lines[1:]).strip()
        return s.strip()

    if len(parts) >= 2:
        log_content = _strip_label(parts[1])
    if len(parts) >= 3:
        eval_json = _strip_label(parts[2])
    if len(parts) >= 4:
        eval_state_json = _strip_label(parts[3])
    if len(parts) >= 5:
        _es = _strip_label(parts[4])
        eval_sample_steps = [l.strip() for l in _es.splitlines() if l.strip()]
    if len(parts) >= 6:
        _ss = _strip_label(parts[5])
        seen_sample_steps = [l.strip() for l in _ss.splitlines() if l.strip()]

    return latest, log_content, eval_json, eval_state_json, eval_log_lines, ckpt_dir, eval_sample_steps, seen_sample_steps


def parse(log_content):
    """解析 log.txt 内容 → rows。
    训练行和 eval 行分开解析：
    - 训练行 (step=N): 来自 train.py log.txt，包含 loss/mem 等
    - eval 行 (MSE=.. SSIM=..): 来自 auto_eval_cpu.log，不混入训练行
    两者在 merge_eval_json 中按 step 对齐合并。"""
    rows = []
    for raw in log_content.splitlines():
        line = LANRE.sub("", raw)
        tm = TS_RE.search(line)
        ts = tm.group(1) if tm else None
        m = ROW_RE.search(line)
        if m:
            step = int(m.group(1))
            fields = {}
            for seg in m.group(2).split("|"):
                seg = seg.strip()
                kv = re.match(r"([A-Za-z0-9 /]+?):\s*(.*)$", seg)
                if not kv:
                    continue
                key = kv.group(1).strip()
                val_str = kv.group(2).strip()
                if key.startswith("Steps/Sec"):
                    fields["stepsPerSec"] = num(val_str.split()[0] if val_str else None)
                elif key == "Mem":
                    mm = re.match(r"([\d.]+)G/([\d.]+)G", val_str)
                    fields["memCur"] = num(mm.group(1) if mm else None)
                    fields["memPeak"] = num(mm.group(2) if mm else None)
                else:
                    nm = re.search(r"-?[\d.]+", val_str)
                    fields[key] = num(nm.group(0) if nm else None)
            rows.append({
                "step": step, "total": fields.get("Total"), "diff": fields.get("Diff"),
                "canny": fields.get("Canny"), "skel": fields.get("Skel"),
                "latc": fields.get("LatC"), "lats": fields.get("LatS"),
                "stdmid": fields.get("StdMid"), "x0lat": fields.get("X0Lat"),
                "stepsPerSec": fields.get("stepsPerSec"),
                "memCur": fields.get("memCur"), "memPeak": fields.get("memPeak"),
                "mse": None, "ssim": None, "ts": ts, "is_eval": False,
            })
            continue
        e = EVAL_RE.search(line)
        if e:
            rows.append({
                "step": int(e.group(1)), "total": None, "diff": None,
                "stdmid": None, "x0lat": None, "stepsPerSec": None,
                "memCur": None, "memPeak": None,
                "mse": num(e.group(2)), "ssim": num(e.group(3)),
                "ts": ts, "is_eval": True,
            })
    # 去重保序
    seen, out = {}, []
    for r in rows:
        s = r["step"]
        if s in seen:
            if r.get("is_eval") and not out[seen[s]].get("is_eval"):
                out[seen[s]] = r
            continue
        seen[s] = len(out)
        out.append(r)
    out.sort(key=lambda r: r["step"])
    return out


def merge_eval_json(rows, eval_json_str):
    """从 eval_auto_*.json 内容解析 MSE/SSIM/SkelIoU，标记 is_eval=True 并填入值。
    支持任意字段顺序（json.loads 解析每个独立 JSON 对象）。"""
    if not eval_json_str:
        return
    ev_map = {}
    # 尝试逐个解析 JSON 对象（eval_auto_*.json 被拼接为多个 {} 块）
    import json as _json
    decoder = _json.JSONDecoder()
    idx = 0
    text = eval_json_str.strip()
    while idx < len(text):
        # 跳过空白和分隔符
        while idx < len(text) and text[idx] in ' \t\n\r':
            idx += 1
        if idx >= len(text):
            break
        try:
            obj, end_idx = decoder.raw_decode(text, idx)
            idx = end_idx
            if isinstance(obj, dict) and "step" in obj and "mse" in obj:
                step = int(obj["step"])
                ev_map[step] = (
                    float(obj["mse"]),
                    float(obj.get("ssim", 0)),
                    float(obj["skel_iou"]) if obj.get("skel_iou") is not None else None,
                )
        except Exception:
            idx += 1  # skip one char and retry
    if not ev_map:
        return
    by_step = {r["step"]: r for r in rows}
    for step, (mse, ssim, skel_iou) in ev_map.items():
        if step in by_step:
            by_step[step]["mse"] = mse
            by_step[step]["ssim"] = ssim
            by_step[step]["is_eval"] = True
            if skel_iou is not None:
                by_step[step]["skel_iou"] = skel_iou
        else:
            rows.append({
                "step": step, "total": None, "diff": None,
                "canny": None, "skel": None, "latc": None, "lats": None,
                "stdmid": None, "x0lat": None, "stepsPerSec": None,
                "memCur": None, "memPeak": None,
                "mse": mse, "ssim": ssim, "skel_iou": skel_iou,
                "ts": None, "is_eval": True,
            })


def parse_eval_log_ts(eval_log_lines):
    """解析 auto_eval_cpu.log 中的 MSE/SSIM 行，提取 step + 时间戳。
    格式: [2026-08-23 21:15:56] [eval] step 5000: MSE=1.37596 SSIM=0.3481 (290s, ...)
    返回 {step: (mse, ssim, timestamp_str)}"""
    result = {}
    for raw in eval_log_lines.splitlines():
        line = LANRE.sub("", raw)
        tm = TS_RE.search(line)
        ts = tm.group(1) if tm else None
        e = EVAL_RE.search(line)
        if e:
            step = int(e.group(1))
            mse = num(e.group(2))
            ssim = num(e.group(3))
            if mse is not None and ssim is not None:
                result[step] = (mse, ssim, ts)
    return result


def write(rows, source):
    out_rows = []
    latest_sps = None
    for r in rows:
        nr = dict(r)
        nr["epoch"] = (r["step"] * BATCH_SIZE) / DATASET_SIZE
        if r.get("stepsPerSec"):
            latest_sps = r["stepsPerSec"]
        out_rows.append(nr)
    sps = latest_sps or 3.5
    last_step = out_rows[-1]["step"] if out_rows else 0
    eta_hours = max(0, TOTAL_STEPS - last_step) / (sps * 3600) if sps else 0
    out = {
        "source": source, "pulledAt": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "count": len(rows), "rows": out_rows,
        "dataset_size": DATASET_SIZE, "batch": BATCH_SIZE,
        "stepsPerSec": sps, "total_steps": TOTAL_STEPS, "eta_hours": eta_hours,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


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


def _existing_steps(local_dir):
    try:
        return sorted(int(d.replace("step", "")) for d in os.listdir(local_dir)
                      if d.startswith("step"))
    except Exception:
        return []


def pull_eval_samples(ckpt_dir, eval_sample_steps, verbose=True):
    """一次 scp -r 拉取整个 eval_samples 目录。
    scp -r 远程目录会在本地创建一层嵌套，需要展平。"""
    if not eval_sample_steps:
        return
    # 清理旧的嵌套目录
    nested = os.path.join(LOCAL_ES, "eval_samples")
    if os.path.isdir(nested):
        import shutil
        # 把嵌套目录里的 step 子目录移到上层
        for d in os.listdir(nested):
            src = os.path.join(nested, d)
            dst = os.path.join(LOCAL_ES, d)
            if os.path.isdir(src) and not os.path.exists(dst):
                shutil.move(src, dst)
        shutil.rmtree(nested, ignore_errors=True)
    remote = f"{ckpt_dir}/eval_samples/"
    _scp_dir(remote, LOCAL_ES, timeout=120)
    # scp -r 可能在 LOCAL_ES 下再建一层 eval_samples/
    nested = os.path.join(LOCAL_ES, "eval_samples")
    if os.path.isdir(nested):
        import shutil
        for d in os.listdir(nested):
            src = os.path.join(nested, d)
            dst = os.path.join(LOCAL_ES, d)
            if os.path.isdir(src) and not os.path.exists(dst):
                shutil.move(src, dst)
            elif os.path.isdir(src) and os.path.isdir(dst):
                # 合并：拷贝缺失的文件
                for f in os.listdir(src):
                    sf = os.path.join(src, f)
                    df = os.path.join(dst, f)
                    if not os.path.exists(df):
                        shutil.move(sf, df)
        shutil.rmtree(nested, ignore_errors=True)
    if verbose:
        steps = _existing_steps(LOCAL_ES)
        print(f"[poster] eval_samples: {len(steps)} steps pulled")


def pull_seen_samples(ckpt_dir, verbose=True):
    """一次 scp -r 拉取整个 seen_samples 目录。"""
    # 清理旧的嵌套目录
    nested = os.path.join(LOCAL_SEEN, "seen_samples")
    if os.path.isdir(nested):
        import shutil
        for d in os.listdir(nested):
            src = os.path.join(nested, d)
            dst = os.path.join(LOCAL_SEEN, d)
            if os.path.isdir(src) and not os.path.exists(dst):
                shutil.move(src, dst)
        shutil.rmtree(nested, ignore_errors=True)
    remote = f"{ckpt_dir}/seen_samples/"
    _scp_dir(remote, LOCAL_SEEN, timeout=120)
    nested = os.path.join(LOCAL_SEEN, "seen_samples")
    if os.path.isdir(nested):
        import shutil
        for d in os.listdir(nested):
            src = os.path.join(nested, d)
            dst = os.path.join(LOCAL_SEEN, d)
            if os.path.isdir(src) and not os.path.exists(dst):
                shutil.move(src, dst)
            elif os.path.isdir(src) and os.path.isdir(dst):
                for f in os.listdir(src):
                    sf = os.path.join(src, f)
                    df = os.path.join(dst, f)
                    if not os.path.exists(df):
                        shutil.move(sf, df)
        shutil.rmtree(nested, ignore_errors=True)
    if verbose:
        steps = _existing_steps(LOCAL_SEEN)
        print(f"[poster] seen_samples: {len(steps)} steps pulled")


def regen_poster(remote_log, ckpt_dir, verbose=True, exp_tag=None):
    """生成 eval poster（show5 + seen5 + GT 行）。输出文件名含实验标识。"""
    if not os.path.exists(os.path.join(HERE, "make_eval_poster.py")):
        if verbose:
            print("[poster] make_eval_poster.py 不存在，跳过")
        return None

    tag = exp_tag or _exp_tag_from_source(remote_log)
    poster_out = os.path.join(HERE, f"poster_{tag}.png")

    show5_steps = [s for s in _existing_steps(LOCAL_ES)
                   if os.path.exists(os.path.join(LOCAL_ES, "step%07d" % s, "samples.json"))]
    if not show5_steps:
        if verbose:
            print("[poster] 无带 samples.json 的 eval step，跳过海报生成")
        return None

    args = [sys.executable, os.path.join(HERE, "make_eval_poster.py"),
            "--show5-dir", LOCAL_ES,
            "--gt-dir", os.path.join(HERE, "remote_gt"),
            "--show5-csv", os.path.join(HERE, "show5_top30.csv"),
            "--eval-json-dir", ckpt_dir,
            "--exp", tag,
            "-o", poster_out]
    seen5_steps = [s for s in _existing_steps(LOCAL_SEEN)
                   if os.path.exists(os.path.join(LOCAL_SEEN, "step%07d" % s, "samples.json"))]
    if seen5_steps:
        args.extend(["--seen5-dir", LOCAL_SEEN,
                     "--seen5-csv", os.path.join(HERE, "seen5_top30.csv")])
    try:
        subprocess.run(args, capture_output=True, text=True, timeout=180)
        if verbose:
            print(f"[poster] 海报已生成: {poster_out} (show5={len(show5_steps)} seen5={len(seen5_steps)})")
        return poster_out
    except Exception as e:
        if verbose:
            print(f"[poster] 海报生成失败: {e}")
        return None


def build_dashboard(poster_name=None, exp_tag=None):
    """生成静态 HTML dashboard。poster_name=实验专属 poster 文件名, exp_tag=实验标识。"""
    template_path = os.path.join(HERE, "train_dashboard.html")
    if not os.path.exists(template_path) or not os.path.exists(OUT_JSON):
        return
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            t = f.read()
        with open(OUT_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        data_js = json.dumps(data, ensure_ascii=False)
        t = t.replace("const COLORS = {", f"const __DATA__ = {data_js};\nconst COLORS = {{", 1)
        old_fetch = (
            "    const res = await fetch('train_data.json?t='+Date.now());\n"
            "    if(!res.ok) throw new Error('HTTP '+res.status);\n"
            "    const data = await res.json();"
        )
        t = t.replace(old_fetch, "    const data = __DATA__;", 1)
        # Poster: 用实验专属文件名 (如 poster_s9_b4_clean_dino.png)
        tag = exp_tag or "unknown"
        p_name = poster_name or f"poster_{tag}.png"
        old_poster = (
            "async function loadPoster(){\n"
            "  await loadImg('latestImg', ['eval_latest.png?t='+Date.now()]);\n"
            "  await loadImg('posterImg', ['eval_poster.png?t='+Date.now()]);\n"
            "}"
        )
        new_poster = (
            f"function loadPoster(){{\n"
            f"  const li=document.getElementById('latestImg'); if(li){{li.src='{p_name}';li.onerror=()=>{{li.style.display='none';}};}}\n"
            f"  const pi=document.getElementById('posterImg'); if(pi){{pi.src='{p_name}';pi.onerror=()=>{{pi.style.display='none';}};}}\n"
            f"}}"
        )
        t = t.replace(old_poster, new_poster, 1)
        t = t.replace("load();\nsyncAuto();", "load();", 1)
        dash_dir = os.path.join(HERE, "dashboards")
        os.makedirs(dash_dir, exist_ok=True)
        # 复制 poster 到 dashboards 目录, 让 html 相对路径能找到
        import shutil as _sh
        _src_poster = os.path.join(HERE, p_name)
        if os.path.exists(_src_poster):
            _sh.copy2(_src_poster, os.path.join(dash_dir, p_name))
        out = os.path.join(dash_dir, f"{tag}.html")
        with open(out, "w", encoding="utf-8") as f:
            f.write(t)
        return out
    except Exception:
        return None


def build_compare_dashboard():
    """生成 s7 vs s8 对比 dashboard。s7 数据来自 s7_history.json 静态快照，
    s8 数据来自当前 train_data.json（实时更新）。"""
    cmp_tpl = os.path.join(HERE, "compare_dashboard.html")
    cmp_builder = os.path.join(HERE, "build_compare_dashboard.py")
    s7_hist = os.path.join(HERE, "s7_history.json")
    s8_json = os.path.join(HERE, "train_data.json")
    if not (os.path.exists(cmp_tpl) and os.path.exists(s7_hist) and os.path.exists(s8_json)):
        return None
    try:
        args = [sys.executable, cmp_builder]
        r = subprocess.run(args, capture_output=True, text=True, timeout=30, cwd=HERE)
        if r.returncode == 0:
            return os.path.join(HERE, "dashboards", "s7_vs_s8.html")
        return None
    except Exception:
        return None


def run_once(verbose=True, do_poster=True):
    """一轮：收集远程 → 解析 → 写 JSON → 可选拉取样图 + 生成海报 + dashboard。"""
    result = collect_remote()
    if result[0] is None:
        if verbose:
            print(f"[{datetime.datetime.now():%H:%M:%S}] SSH 收集失败")
        return 0
    latest, log_content, eval_json, eval_state_json, eval_log_lines, ckpt_dir, eval_steps, seen_steps = result

    # 写本地 log
    if log_content:
        open(LOCAL_CUR, "w", encoding="utf-8").write(log_content)
    if eval_json:
        open(LOCAL_EVAL_JSON, "w", encoding="utf-8").write(eval_json)

    rows = parse(log_content or "")
    merge_eval_json(rows, eval_json or "")

    # 用 eval_log_lines（auto_eval_cpu.log 的 MSE/SSIM 行）补充 eval 行
    # 这些行带真实时间戳，且可能包含 train log 里没有的 eval step
    eval_log_ts = parse_eval_log_ts(eval_log_lines or "")
    # 把 eval_log 中 step 不在 rows 的追加进去；在的则确保 is_eval/mse/ssim 准确
    existing_steps = {r["step"] for r in rows}
    for step, (mse, ssim, ts) in eval_log_ts.items():
        if step not in existing_steps:
            rows.append({
                "step": step, "total": None, "diff": None,
                "stdmid": None, "x0lat": None, "stepsPerSec": None,
                "memCur": None, "memPeak": None,
                "mse": mse, "ssim": ssim, "ts": ts, "is_eval": True,
            })
            existing_steps.add(step)
        else:
            for r in rows:
                if r["step"] == step:
                    r["is_eval"] = True
                    r["mse"] = mse
                    r["ssim"] = ssim
                    if ts:
                        r["ts"] = ts
                    break
    rows.sort(key=lambda r: r["step"])

    write(rows, f"remote:{REMOTE_HOST}:{latest}")

    eval_rows = [r for r in rows if r.get("is_eval")]
    last = rows[-1] if rows else None

    poster_path = None
    if do_poster and eval_rows:
        # 拉取 eval_samples 和 seen_samples（各一次 scp -r）
        pull_eval_samples(ckpt_dir, eval_steps, verbose)
        pull_seen_samples(ckpt_dir, verbose)
        poster_path = regen_poster(latest, ckpt_dir, verbose, exp_tag=_exp_tag_from_source(latest))

    # 生成 dashboard (实时实验独立, 不合并其他)
    tag = _exp_tag_from_source(latest)
    dash = build_dashboard(poster_name=os.path.basename(poster_path) if poster_path else f"poster_{tag}.png",
                           exp_tag=tag)

    if verbose:
        if last and last.get("diff") is not None:
            tail = f"step={last['step']} diff={last['diff']:.4f}"
        elif last:
            tail = f"step={last['step']} diff=?"
        else:
            tail = "无数据"
        ev_info = f" evals={len(eval_rows)}" if eval_rows else ""
        print(f"[{datetime.datetime.now():%H:%M:%S}] rows={len(rows)} {tail}{ev_info}")
        if dash:
            print(f"  dashboard → {dash}")
    return len(rows)


def main():
    interval = 120
    if "--interval" in sys.argv:
        try:
            interval = int(sys.argv[sys.argv.index("--interval") + 1])
        except Exception:
            pass
    do_poster = "--no-poster" not in sys.argv
    if "--loop" in sys.argv:
        print(f"[pull_monitor] loop 每 {interval}s (poster={'on' if do_poster else 'off'})")
        while True:
            t0 = time.time()
            try:
                run_once(do_poster=do_poster)
            except Exception as e:
                print(f"[pull_monitor] error: {e}")
            time.sleep(max(1, interval - (time.time() - t0)))
    else:
        run_once(do_poster=do_poster)


if __name__ == "__main__":
    main()
