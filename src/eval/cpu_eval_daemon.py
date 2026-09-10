# -*- coding: utf-8 -*-
"""cpu_eval_daemon.py — 常驻 CPU eval 守护 (双 socket 负载均衡切分).

调度 (按实测单样本成本加权: base 0.092 min/样本, ctrl 0.160 min/样本, doc34 §7):
  node0 (CPU 0-31):  base[0:B0] + ctrl[0:C0]
  node1 (CPU 32-63): base[B0:100] + ctrl[C0:100]
  B0/C0 由 _balance() 求解, 使两 node 墙钟相等 (总功 ~23.6 min → 墙钟 ~12-13 min).

每 ckpt 流程: newest-missing 选点 → 双 worker (LD_PRELOAD jemalloc, taskset 绑物理核)
→ 等 .part_*.json → 逐样本列表精确合并 (mean/std/分位数) →
写 eval_auto_ctrl_{step}.json (eval_facade 同 schema, 早停/daemon/collect 零改动兼容).

用法:
  常驻: python -u src/eval/cpu_eval_daemon.py --watch-root 5script/results/v9c_skel_joint
  验证: python -u src/eval/cpu_eval_daemon.py --once <ckpt.pt> --report /tmp/r.json
"""
import os, sys, json, time, glob, argparse, subprocess
import numpy as np

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _root)
os.chdir(_root)
sys.stdout.reconfigure(encoding="utf-8")

COST = {"base": 0.098, "ctrl": 0.190}  # min/样本 (采样+decode+metrics 份额, doc34)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [cpu-eval] {msg}", flush=True)


def find_numactl():
    for p in ("/usr/bin/numactl", "/usr/local/bin/numactl"):
        if os.path.exists(p):
            return p
    return None


def _balance(n=100):
    """求 node0 的 base 数 B0 与 ctrl 数 C0, 使两 node 墙钟相等."""
    best, b0, c0 = 1e9, 0, 0
    for B0 in range(0, n + 1, 2):
        # node0 时间 = 0.092*B0 + 0.16*C0 + fixed; 令 = node1 时间
        # 0.092*B0 + 0.16*C0 = 0.092*(n-B0) + 0.16*(n-C0)
        C0 = int(round((COST["base"] * (n - 2 * B0) + COST["ctrl"] * n) / (2 * COST["ctrl"])))
        C0 = max(0, min(n, C0))
        l0 = COST["base"] * B0 + COST["ctrl"] * C0
        l1 = COST["base"] * (n - B0) + COST["ctrl"] * (n - C0)
        diff = abs(l0 - l1)
        if diff < best:
            best, b0, c0 = diff, B0, C0
    return b0, c0


def latest_active_ckpt_dir(root):
    p = os.path.join(root, "_active_ckpt_dir.txt")
    if not os.path.exists(p):
        return None
    d = open(p, encoding="utf-8").read().strip()
    return d if os.path.isdir(d) else None


def newest_missing(ckpt_dir, state_last, flat=False):
    """flat=True: train.py 预训练 (eval_auto_{step}.json 平铺 json, .pt 可无 .done)."""
    cands = {}
    for f in glob.glob(os.path.join(ckpt_dir, "*.pt.done")):
        cands[int(os.path.basename(f).replace(".pt.done", ""))] = True
    for f in glob.glob(os.path.join(ckpt_dir, "*.pt")):
        st = int(os.path.basename(f).replace(".pt", ""))
        if st not in cands and time.time() - os.path.getmtime(f) > 90:
            cands[st] = True
    for step in sorted(cands, reverse=True):
        ckpt = os.path.join(ckpt_dir, f"{step:07d}.pt")
        if not os.path.isfile(ckpt):
            continue
        done_json = (f"eval_auto_{step}.json" if flat
                     else f"eval_auto_ctrl_{step}.json")
        if os.path.exists(os.path.join(ckpt_dir, done_json)):
            continue
        if step <= state_last:
            continue
        if glob.glob(os.path.join(ckpt_dir, f"{step:07d}.cpu_eval.*")):
            continue
        return ckpt, step
    return None


def merge_parts(part_files, n):
    """逐样本列表精确合并 → eval_facade 同 schema 的 res dict."""
    agg = {}
    cfg = steps = None
    t_sample = t_decode = t_load = 0.0
    for pf in part_files:
        d = json.load(open(pf, encoding="utf-8"))
        t_load += d.get("t_load", 0)
        for p in d["parts"]:
            a = agg.setdefault(p["arm"], {"mse": [], "ssim": [], "skel_iou": [], "lpips": []})
            a["mse"] += p["lists"]["mse"]
            a["ssim"] += p["lists"]["ssim"]
            a["skel_iou"] += p["lists"]["skel_iou"]
            if p["lists"].get("lpips"):
                a["lpips"] += p["lists"]["lpips"]
    res = {}
    for arm, a in agg.items():
        m = {"n": len(a["mse"])}
        if not a["mse"]:
            continue
        m["mse_mean"] = float(np.mean(a["mse"]))
        m["mse_std"] = float(np.std(a["mse"]))
        m["mse_q25"], m["mse_q50"], m["mse_q75"] = [float(q) for q in np.percentile(a["mse"], [25, 50, 75])]
        m["ssim_mean"] = float(np.mean(a["ssim"]))
        m["ssim_std"] = float(np.std(a["ssim"]))
        m["ssim_p10"], m["ssim_q25"], m["ssim_med"], m["ssim_q75"], m["ssim_p90"] = \
            [float(q) for q in np.percentile(a["ssim"], [10, 25, 50, 75, 90])]
        m["skel_iou_mean"] = float(np.mean(a["skel_iou"]))
        m["skel_iou_std"] = float(np.std(a["skel_iou"]))
        if a["lpips"]:
            m["lpips_mean"] = float(np.mean(a["lpips"]))
        res[arm] = m
    if "ctrl" in res and "base" in res:   # delta 仅 ctrl_pair 双臂模式存在
        for k in ("mse", "ssim", "lpips"):
            if f"{k}_mean" in res["ctrl"] and f"{k}_mean" in res["base"]:
                res[f"delta_{k}"] = res["ctrl"][f"{k}_mean"] - res["base"][f"{k}_mean"]
    return res


def run_pair(ckpt, run_dir, threads, dit_batch, vae_batch, numactl, je, report=None,
             mode="ctrl_pair", log_dir=None, eval_sets=None, strict_every=0):
    ckpt_dir = os.path.dirname(ckpt)
    step = int(os.path.basename(ckpt).split(".")[0])
    lock = os.path.join(ckpt_dir, f"{step:07d}.cpu_eval.lock")
    open(lock, "w").close()
    env = dict(os.environ)
    if je:
        env["LD_PRELOAD"] = je
    env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               _eval_step=str(step))
    py = sys.executable
    n = 100
    n_map = {}
    eval_sets_str = ""
    if mode == "pretrain_g":
        # 多 eval 集: seen (每 ckpt) + strict (每 strict_every 步)。
        # 每集 n = 该 csv 实际行数; 两 worker 按集各切一半。
        try:
            import torch as _t
            _na = _t.load(ckpt, map_location="cpu", weights_only=False).get("args", {})
            _a = vars(_na) if isinstance(_na, argparse.Namespace) else (_na or {})
            if not eval_sets:
                _csv0 = _a.get("gpu_eval_csv") or _a.get("eval_csv") or _a.get("data_csv") or ""
                eval_sets = [("g", _csv0)]
        except Exception:
            eval_sets = eval_sets or [("g", "")]
        active = []
        for _nm, _cp in eval_sets:
            _cp_full = _cp if os.path.isabs(_cp) else os.path.join("/root/Workspace/xy/DiT", _cp)
            if _nm != "seen" and _nm != "g" and strict_every > 0 and step % strict_every != 0:
                continue
            active.append((_nm, _cp_full))
        seg0s, seg1s = [], []
        for _nm, _cp_full in active:
            _rows = max(sum(1 for _ in open(_cp_full, encoding="utf-8")) - 1, 2)
            n_map[_nm] = _rows
            seg0s.append(f"{_nm}:0:{_rows // 2}")
            seg1s.append(f"{_nm}:{_rows // 2}:{_rows}")
        seg0, seg1 = ",".join(seg0s), ",".join(seg1s)
        eval_sets_str = ",".join(f"{_nm}={_cp_full}" for _nm, _cp_full in active)
        n = n_map.get("seen") or n_map.get("g") or 100
        log(f"step {step}: pretrain_g 双 worker, 集={list(n_map.keys())} "
            f"n={n_map} (strict 每 {strict_every} 步)")
    else:
        b0, c0 = _balance(n)
        seg0 = f"base:0:{b0},ctrl:0:{c0}"
        seg1 = f"base:{b0}:{n},ctrl:{c0}:{n}"
        log(f"step {step}: 双臂启动 (balance base[:{b0}]/ctrl[:{c0}] @node0 | rest @node1)")

    cmd0 = (["numactl", "-N", "0", "-m", "0"] if numactl else ["taskset", "-c", "0-31"])
    cmd1 = (["numactl", "-N", "1", "-m", "1"] if numactl else ["taskset", "-c", "32-63"])
    common = ["--ckpt", ckpt, "--out-dir", run_dir, "--threads", str(threads),
              "--dit-batch", str(dit_batch), "--vae-batch", str(vae_batch), "--n", str(n),
              "--mode", mode]
    if eval_sets_str:
        common += ["--eval-sets", eval_sets_str]
    t0 = time.time()
    # worker 日志: logs/<exp>/ (不再写 /tmp)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    wlog = lambda tag: os.path.join(
        log_dir or "/tmp", f"cpu_eval_w_{step}_{tag}.log")
    procs = []
    for tag, cmd, segs in (("p0", cmd0, seg0), ("p1", cmd1, seg1)):
        procs.append((tag, subprocess.Popen(
            cmd + [py, "-u", "src/eval/cpu_eval_worker.py",
                   "--segments", segs, "--part-tag", tag] + common,
            env=env, stdout=open(wlog(tag), "w"),
            stderr=subprocess.STDOUT)))
    ok = True
    for tag, p in procs:
        rc = None
        try:
            rc = p.wait(timeout=30 * 60)
        except subprocess.TimeoutExpired:
            log(f"step {step}: {tag} 超时 kill")
            p.kill()
            ok = False
        if rc is None or rc != 0:
            log(f"step {step}: {tag} rc={rc} (日志 {wlog(tag)})")
            ok = False
    part_files = [os.path.join(ckpt_dir, f".part_{t}.json") for t in ("p0", "p1")]
    if not ok or not all(os.path.exists(p) for p in part_files):
        os.remove(lock)
        return False

    res = merge_parts(part_files, n)
    if res is None:
        os.remove(lock)
        return False
    if mode == "pretrain_g":
        # seen (旧平铺键, 兼容既有报表) + strict (嵌套字典) 双集落盘
        _seen = res.get("seen") or res.get("g")
        if _seen is None:
            os.remove(lock)
            return False
        flat = {"n": _seen["n"], "ssim": _seen["ssim_mean"], "ssim_std": _seen["ssim_std"],
                "mse": _seen["mse_mean"], "mse_std": _seen["mse_std"],
                "lpips": _seen.get("lpips_mean"), "skel_iou": _seen["skel_iou_mean"],
                "ssim_p10": _seen["ssim_p10"], "ssim_q25": _seen["ssim_q25"],
                "ssim_med": _seen["ssim_med"], "ssim_q75": _seen["ssim_q75"],
                "ssim_p90": _seen["ssim_p90"],
                "step": step, "elapsed_s": round(time.time() - t0, 1),
                "engine": "cpu_2sock_g"}
        if "strict" in res:
            flat["strict"] = res["strict"]
    else:
        flat = res
    flat.update({"n": n, "elapsed_s": round(time.time() - t0, 1)})
    # cfg/steps 从 ckpt args 带回 (与 eval_facade json 字段一致)
    try:
        import torch as _t
        _na = _t.load(ckpt, map_location="cpu", weights_only=False).get("args", {})
        a = vars(_na) if isinstance(_na, argparse.Namespace) else (_na or {})
        _cfg = float(a.get("eval_cfg", a.get("gpu_eval_cfg", 0.7)))
        _steps = int(a.get("eval_steps", a.get("gpu_eval_steps", 50)))
        if mode == "pretrain_g":
            flat["cfg"], flat["ddim_steps"] = _cfg, _steps
        else:
            res["cfg"], res["ddim_steps"] = _cfg, _steps
    except Exception:
        pass
    out_json = report or os.path.join(
        ckpt_dir, f"eval_auto_{step}.json" if mode == "pretrain_g"
        else f"eval_auto_ctrl_{step}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(flat if mode == "pretrain_g" else res, f, ensure_ascii=False)
    for p in part_files:
        if os.path.exists(p):
            os.remove(p)
    os.remove(lock)
    _r = flat if mode == "pretrain_g" else res
    _ssim = _r.get("ssim") or (_r.get("ctrl", {}).get("ssim_mean"))
    log(f"step {step}: DONE {time.time()-t0:.0f}s ssim={_ssim} -> {os.path.basename(out_json)}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch-root", action="append", default=[])
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--dit-batch", type=int, default=16)
    ap.add_argument("--vae-batch", type=int, default=24)
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--once", default="")
    ap.add_argument("--mode", choices=["ctrl_pair", "pretrain_g"], default="ctrl_pair")
    ap.add_argument("--report", default="")
    ap.add_argument("--eval-sets", default="seen=5script/eval_seen_v10.csv,strict=5script/eval_fame3_strict_clean_v9.csv",
                    help="pretrain_g 多 eval 集 'name=csv[,name=csv...]'; seen 每 ckpt, 其他集按 --strict-every")
    ap.add_argument("--strict-every", type=int, default=10000,
                    help="非 seen 集 (strict) 的评测间隔 (按 ckpt step); 0 = 每 ckpt 都评")
    args = ap.parse_args()

    numactl = find_numactl()
    je = "/opt/conda/envs/cu121/lib/libjemalloclocal.so.2"
    if not os.path.exists(je):
        je = ""
    log(f"daemon 启动: numactl={'y' if numactl else 'n(taskset)'} jemalloc={'y' if je else 'n'} "
        f"threads={args.threads} watch={args.watch_root or args.once}")

    _es = []
    for _spec in args.eval_sets.split(","):
        if "=" in _spec:
            _nm, _cp = _spec.split("=", 1)
            _es.append((_nm, _cp))

    if args.once:
        ckpt = os.path.abspath(args.once)
        run_dir = os.path.dirname(os.path.dirname(ckpt))
        _ld = os.path.join(
            "/root/Workspace/xy/DiT/logs",
            os.path.basename(os.path.normpath(os.path.dirname(run_dir))))
        ok = run_pair(ckpt, run_dir, args.threads, args.dit_batch, args.vae_batch,
                      numactl, je, report=args.report or None, mode=args.mode,
                      log_dir=_ld, eval_sets=_es, strict_every=args.strict_every)
        sys.exit(0 if ok else 1)

    state = {}
    while True:
        for root in args.watch_root:
            ckpt_dir = latest_active_ckpt_dir(root)
            if not ckpt_dir:
                continue
            last = state.get(ckpt_dir, -1)
            hit = newest_missing(ckpt_dir, last, flat=(args.mode == "pretrain_g"))
            if not hit:
                continue
            ckpt, step = hit
            _ld = os.path.join("/root/Workspace/xy/DiT/logs",
                               os.path.basename(os.path.normpath(root)))
            if run_pair(ckpt, os.path.dirname(ckpt_dir), args.threads, args.dit_batch,
                        args.vae_batch, numactl, je, mode=args.mode, log_dir=_ld,
                        eval_sets=_es, strict_every=args.strict_every):
                state[ckpt_dir] = step
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
