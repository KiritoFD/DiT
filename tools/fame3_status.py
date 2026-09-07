#!/usr/bin/env python3
"""fame3 std-skel 实验一键状态查看.

用法:
    python tools/fame3_status.py [exp_name] [--compare baseline]
    exp_name: 结果目录名 (默认 v10b_stdskel_fame3_d01)
    --compare: 对照实验名 (默认 v10b_stdskel_fame3)

单次 ssh 批量采集: 训练日志 + GPU + ckpt + eval json + follow 诊断.
"""
import argparse, json, os, re, subprocess, sys
from datetime import datetime

REMOTE = "4090"
REMOTE_ROOT = "/root/Workspace/xy/DiT"
RESULTS_ROOT = f"{REMOTE_ROOT}/5script/results"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(s):
    return ANSI_RE.sub("", s)


def ssh_batch(cmd, timeout=30):
    try:
        r = subprocess.run(["ssh", REMOTE, cmd], capture_output=True, text=True, timeout=timeout)
        return r.stdout
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[ERR: {e}]"


def build_remote_script(exp, compare, logpath):
    parts = []
    parts.append(f"echo '===SECTION:TRAINLOG==='; tail -40 {logpath} 2>/dev/null")
    parts.append(f"echo '===SECTION:GPU==='; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null")
    parts.append(f"echo '===SECTION:ACTIVE==='; cat {RESULTS_ROOT}/{exp}/_active_ckpt_dir.txt 2>/dev/null")
    parts.append(f"echo '===SECTION:CKPT_LIST==='; ACTIVE=$(cat {RESULTS_ROOT}/{exp}/_active_ckpt_dir.txt 2>/dev/null); CKPT_DIR={REMOTE_ROOT}/$ACTIVE; ls -1 $CKPT_DIR/*.pt 2>/dev/null | sort | tail -6")
    parts.append(f"echo '===SECTION:EVAL_LIST==='; ACTIVE=$(cat {RESULTS_ROOT}/{exp}/_active_ckpt_dir.txt 2>/dev/null); CKPT_DIR={REMOTE_ROOT}/$ACTIVE; for f in $(ls -1 $CKPT_DIR/eval_auto_*.json 2>/dev/null | sort); do echo '---EVALFILE---'; cat $f; done")
    if compare:
        parts.append(f"echo '===SECTION:CMP_ACTIVE==='; cat {RESULTS_ROOT}/{compare}/_active_ckpt_dir.txt 2>/dev/null")
        parts.append(f"echo '===SECTION:CMP_EVAL==='; CACTIVE=$(cat {RESULTS_ROOT}/{compare}/_active_ckpt_dir.txt 2>/dev/null); CKPT_DIR={REMOTE_ROOT}/$CACTIVE; for f in $(ls -1 $CKPT_DIR/eval_auto_*.json 2>/dev/null | sort | tail -3); do echo '---EVALFILE---'; cat $f; done")
    parts.append(f"echo '===SECTION:FOLLOW_DIAG==='; cat {RESULTS_ROOT}/{exp}/debug_stdskel.json 2>/dev/null")
    parts.append("echo '===SECTION:END==='")
    return " ; ".join(parts)


def parse_sections(raw):
    sections = {}
    current = None
    buf = []
    for line in raw.splitlines():
        m = re.match(r"===SECTION:(\w+)===", line)
        if m:
            if current:
                sections[current] = "\n".join(buf)
            current = m.group(1)
            buf = []
        else:
            buf.append(line)
    if current:
        sections[current] = "\n".join(buf)
    return sections


def parse_train_log(raw):
    if not raw.strip():
        return None
    lines = [strip_ansi(l) for l in raw.splitlines()]
    step_lines = [l for l in lines if "(step=" in l]
    epoch_lines = [l for l in lines if "Beginning epoch" in l]
    latest_steps = step_lines[-5:] if len(step_lines) >= 5 else step_lines
    latest_epoch = epoch_lines[-1] if epoch_lines else ""
    parsed = []
    for l in latest_steps:
        m_step = re.search(r"step=(\d+)", l)
        m_total = re.search(r"Total:\s+([\d.]+)", l)
        m_diff = re.search(r"Diff:\s+([\d.]+)", l)
        m_sps = re.search(r"Steps/Sec:\s+([\d.]+)", l)
        m_lr = re.search(r"LR:\s+([\d.eE+-]+)", l)
        m_ema = re.search(r"EMA:\s+([\d.]+)", l)
        m_mem = re.search(r"Mem:\s+([\d.]+G/[\d.]+G)", l)
        m_repaw = re.search(r"REPA.*?w=([\d.]+)", l)
        m_stdmid = re.search(r"StdMid:\s+raw\s+([\d.]+)", l)
        ts = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", l)
        parsed.append({
            "step": int(m_step.group(1)) if m_step else 0,
            "total": float(m_total.group(1)) if m_total else 0,
            "diff": float(m_diff.group(1)) if m_diff else 0,
            "sps": float(m_sps.group(1)) if m_sps else 0,
            "lr": m_lr.group(1) if m_lr else "?",
            "ema": float(m_ema.group(1)) if m_ema else 0,
            "mem": m_mem.group(1) if m_mem else "?",
            "repaw": float(m_repaw.group(1)) if m_repaw else 0,
            "stdmid": float(m_stdmid.group(1)) if m_stdmid else 0,
            "ts": ts.group(1) if ts else "?",
        })
    return {"steps": parsed, "latest_epoch": latest_epoch.strip()}


def parse_eval_list(raw):
    if not raw.strip():
        return []
    results = []
    for block in raw.split("---EVALFILE---"):
        block = block.strip()
        if not block:
            continue
        try:
            results.append(json.loads(block))
        except json.JSONDecodeError:
            pass
    return results


def fmt_eval_row(r):
    step = r.get("step", "?")
    ssim = r.get("ssim", 0)
    skel = r.get("skel_iou", 0)
    lpips = r.get("lpips", 0)
    cfg = r.get("cfg", "?")
    eng = r.get("engine", "?")
    return f"  step={step:>6}  SSIM={ssim:.4f}  skel_iou={skel:.4f}  LPIPS={lpips:.4f}  cfg={cfg}  eng={eng}"


def main():
    ap = argparse.ArgumentParser(description="fame3 std-skel 实验状态查看")
    ap.add_argument("exp", nargs="?", default="v10b_stdskel_fame3_d01",
                    help="实验结果目录名 (默认 v10b_stdskel_fame3_d01)")
    ap.add_argument("--compare", default="v10b_stdskel_fame3",
                    help="对照实验名 (默认 v10b_stdskel_fame3)")
    ap.add_argument("--logname", default=None,
                    help="训练日志文件名 (默认自动推断)")
    args = ap.parse_args()

    if args.logname:
        logname = args.logname
    elif args.exp == "v10b_stdskel_fame3":
        logname = "v10bstdskel3"
    elif args.exp == "v10b_stdskel_fame3_d01":
        logname = "v10bstdskel3d01"
    else:
        logname = args.exp.replace("-", "").replace("_", "")
    logpath = f"/tmp/{logname}_train.log"

    print(f"{'='*80}")
    print(f"  fame3 std-skel 实验状态  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  实验: {args.exp}    对照: {args.compare}")
    print(f"{'='*80}")

    remote_script = build_remote_script(args.exp, args.compare, logpath)
    raw = ssh_batch(remote_script, timeout=25)
    if raw == "[TIMEOUT]" or raw.startswith("[ERR"):
        print(f"\n[ssh 失败: {raw}]")
        return

    sec = parse_sections(raw)

    # 1. 训练进度
    print(f"\n[1] 训练进度  (log: {logpath})")
    info = parse_train_log(sec.get("TRAINLOG", ""))
    if info and info["steps"]:
        if info["latest_epoch"]:
            print(f"  {info['latest_epoch']}")
        for s in info["steps"]:
            print(f"  [{s['ts']}] step={s['step']:>6}  Total={s['total']:.4f}  Diff={s['diff']:.4f}  "
                  f"sps={s['sps']:.2f}  LR={s['lr']}  EMA={s['ema']:.6f}  Mem={s['mem']}  "
                  f"REPAw={s['repaw']:.2f}  StdMid={s['stdmid']:.4f}")
        last = info["steps"][-1]
        if last["sps"] > 0:
            remaining = 60000 - last["step"]
            eta_h = remaining / last["sps"] / 3600
            print(f"  -> ETA 60k: ~{eta_h:.1f}h ({remaining} steps @ {last['sps']:.2f} sps)")
    else:
        print("  [无训练日志]")

    # 2. GPU
    print(f"\n[2] GPU")
    gpu = sec.get("GPU", "").strip()
    print(f"  {gpu or '[n/a]'}")

    # 3. ckpt
    active = sec.get("ACTIVE", "").strip()
    print(f"\n[3] ckpt  (active: {active or '?'})")
    ckpt_list = sec.get("CKPT_LIST", "").strip()
    if ckpt_list:
        for line in ckpt_list.splitlines():
            print(f"  {os.path.basename(line)}")
    else:
        print("  [无 ckpt]")

    # 4. eval
    print(f"\n[4] eval (auto)")
    evals = parse_eval_list(sec.get("EVAL_LIST", ""))
    if evals:
        for r in evals:
            print(fmt_eval_row(r))
    else:
        print("  [无 eval]")

    # 5. 对照
    cmp_active = sec.get("CMP_ACTIVE", "").strip()
    if cmp_active:
        print(f"\n[5] 对照 eval  ({args.compare}, active: {cmp_active})")
        cmp_evals = parse_eval_list(sec.get("CMP_EVAL", ""))
        if cmp_evals:
            for r in cmp_evals:
                print(fmt_eval_row(r))
        else:
            print("  [无 eval]")

    # 6. follow 诊断
    print(f"\n[6] follow IoU3 诊断  ({args.exp})")
    diag_raw = sec.get("FOLLOW_DIAG", "").strip()
    if diag_raw:
        try:
            diag = json.loads(diag_raw)
            gs = diag.get("glyph_scale", {})
            gi = diag.get("glyph_influence_pct", 0)
            fiou = diag.get("follow_iou3", 0)
            print(f"  glyph_scale: init={gs.get('init','?')}  final={gs.get('final','?')}  delta={gs.get('delta','?')}")
            print(f"  g 注入作用: {gi:.2f}%")
            print(f"  follow IoU3: {fiou:.4f}")
            layers = diag.get("per_layer_g_influence", [])
            if layers:
                print(f"  逐层 g 衰减: {' -> '.join(f'{v:.1f}%' for v in layers)}")
        except json.JSONDecodeError:
            print("  [debug_stdskel.json 解析失败]")
    else:
        print(f"  [无 debug_stdskel.json — 用 tools/debug_fame3_stdskel.py 生成]")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
