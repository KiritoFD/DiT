# -*- coding: utf-8 -*-
"""gpu_eval_loop.py — GPU in-mem 评测循环.

设计: **评测时暂停训练 (SIGSTOP) -> in-mem GPU 采样 -> 恢复训练 (SIGCONT)**。
  - 不再与训练抢 SM; eval 用独立进程加载 ckpt (训练进程显存仍占用, 故 batch 仍小)
  - in-mem 指标 (无 PNG 必需; --save-samples 落盘供 poster)
  - 自动写 eval_auto_<step>.json + 生成 poster
  - --device cuda|cpu 开关

用法:
  python tools/eval/gpu_eval_loop.py --results-dir assets/results/<exp> \
      [--device cuda] [--strict-every 10000] [--poll 30] [--pause-train]
"""
import argparse
import csv
import glob
import json
import os
import signal
import subprocess
import sys
import time

BASE = "/root/Workspace/xy/DiT"
sys.path.insert(0, BASE)
os.chdir(BASE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] [gpu-eval] {m}", flush=True)


def train_pids(pattern):
    try:
        r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
        return [int(x) for x in r.stdout.split() if x.strip()]
    except Exception:
        return []


def pause_procs(pids):
    for p in pids:
        try:
            os.kill(p, signal.SIGSTOP)
        except Exception:
            pass


def resume_procs(pids):
    for p in pids:
        try:
            os.kill(p, signal.SIGCONT)
        except Exception:
            pass



def active_ckpt_dir(root):
    p = os.path.join(root, "_active_ckpt_dir.txt")
    if os.path.exists(p):
        d = open(p, encoding="utf-8").read().strip()
        if os.path.isdir(d):
            return d
    ds = [d for d in glob.glob(os.path.join(root, "*", "checkpoints")) if os.path.isdir(d)]
    return max(ds, key=os.path.getmtime) if ds else None


def newest_missing(ckpt_dir):
    cands = {}
    for f in glob.glob(os.path.join(ckpt_dir, "*.pt.done")):
        cands[int(os.path.basename(f).replace(".pt.done", ""))] = True
    for f in glob.glob(os.path.join(ckpt_dir, "*.pt")):
        st = int(os.path.basename(f).replace(".pt", ""))
        if st not in cands and time.time() - os.path.getmtime(f) > 90:
            cands[st] = True
    for st in sorted(cands, reverse=True):
        ck = os.path.join(ckpt_dir, f"{st:07d}.pt")
        if os.path.isfile(ck) and not os.path.exists(
                os.path.join(ckpt_dir, f"eval_auto_{st}.json")):
            return ck, st
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--strict-every", type=int, default=10000)
    ap.add_argument("--seen-csv", default="assets/eval_seen_v10.csv")
    ap.add_argument("--strict-csv", default="assets/eval_fame3_strict_clean_v9.csv")
    ap.add_argument("--run", action="store_true", help="跑一次遍历后退出 (debug)")
    ap.add_argument("--pause-train", action="store_true", default=True,
                    help="评测期间 SIGSTOP 训练进程, 结束 SIGCONT (默认开)")
    ap.add_argument("--no-pause-train", dest="pause_train", action="store_false")
    ap.add_argument("--pause-match", default="src/train/train.py",
                    help="训练进程匹配串 (pgrep -f)")
    ap.add_argument("--self-cond", action="store_true", default=True,
                    help="同时运行自条件评测并对比记录 (默认开)")
    ap.add_argument("--no-self-cond", dest="self_cond", action="store_false")
    ap.add_argument("--blend-alpha", type=float, default=0.5,
                    help="自条件骨架混合系数 (默认 0.5)")
    args = ap.parse_args()

    root = os.path.abspath(args.results_dir)
    log(f"watch {root} device={args.device} strict_every={args.strict_every}")
    while True:
        ckpt_dir = active_ckpt_dir(root)
        if not ckpt_dir:
            time.sleep(args.poll)
            continue
        seg = os.path.dirname(ckpt_dir)             # .../<seg>
        sum_path = os.path.join(root, "eval_stdskel_summary.csv")
        while True:
            ck, step = newest_missing(ckpt_dir)
            if not ck:
                break
            sets = [f"seen:{args.seen_csv}:10"]
            if args.strict_every <= 0 or step % args.strict_every == 0:
                sets.append(f"strict:{args.strict_csv}:237")
            # 训练 SIGSTOP 后显存仍占用; batch192 训练 ~18G -> 留 ~6G, dit16/vae8 (~2.2G)
            batches = ["--dit-batch", "16", "--vae-batch", "8"]
            cmd = [sys.executable, "-u", "-m", "src.eval.batch_eval",
                   "--results-dir", root, "--ckpt-override", ck, "--device", args.device,
                   "--save-samples", "--sets", *sets] + batches
            _env = {**os.environ, "PYTHONPATH": BASE}
            log(f"step {step}: eval {sets} on {args.device} ...")
            t0 = time.time()
            pids = train_pids(args.pause_match) if args.pause_train else []
            if pids:
                log(f"step {step}: PAUSE training pids={pids} -> eval in-mem")
                pause_procs(pids)
            try:
                # 1) Origin baseline 评测
                r = subprocess.run(cmd, capture_output=True, text=True, env=_env)
                if r.returncode != 0:
                    log(f"step {step}: eval FAILED rc={r.returncode} :: {r.stderr.strip()[-300:]}")
                    break
                tail = [l for l in r.stdout.splitlines() if "step" in l or "peak" in l][-4:]
                for l in tail:
                    log("  [orig] " + l)

                # 2) Self-Conditioning 评测 (如果启用)
                if args.self_cond:
                    sets_sc = [f"seen_sc:{args.seen_csv}:10"]
                    if args.strict_every <= 0 or step % args.strict_every == 0:
                        sets_sc.append(f"strict_sc:{args.strict_csv}:237")
                    cmd_sc = [sys.executable, "-u", "-m", "src.eval.batch_eval",
                              "--results-dir", root, "--ckpt-override", ck, "--device", args.device,
                              "--self-cond", "--blend-alpha", str(args.blend_alpha),
                              "--sets", *sets_sc] + batches
                    r_sc = subprocess.run(cmd_sc, capture_output=True, text=True, env=_env)
                    if r_sc.returncode == 0:
                        tail_sc = [l for l in r_sc.stdout.splitlines() if "step" in l or "peak" in l][-4:]
                        for l in tail_sc:
                            log("  [self-cond] " + l)
                    else:
                        log(f"step {step}: self-cond eval failed rc={r_sc.returncode}")
            finally:
                if pids:
                    resume_procs(pids)
                    log(f"step {step}: RESUME training pids={pids}")

            # 汇总 eval_auto json (seen + strict, origin + self-cond)
            flat = {"step": step, "elapsed_s": round(time.time() - t0, 1),
                    "engine": f"gpu_inmem_{args.device}"}
            seen_orig = None
            seen_sc = None
            strict_orig = None
            strict_sc = None
            if os.path.exists(sum_path):
                for row in csv.DictReader(open(sum_path, encoding="utf-8")):
                    if int(row["step"]) != step:
                        continue
                    sname = row["set"]
                    if sname == "seen":
                        seen_orig = {"ssim": float(row["ssim_mean"]), "mse": float(row["mse_mean"]),
                                     "med": float(row.get("ssim_med", 0)), "q3": float(row.get("ssim_q3", 0))}
                    elif sname == "seen_sc":
                        seen_sc = {"ssim": float(row["ssim_mean"]), "mse": float(row["mse_mean"]),
                                   "med": float(row.get("ssim_med", 0)), "q3": float(row.get("ssim_q3", 0))}
                    elif sname == "strict":
                        strict_orig = {"n": int(row["n"]), "ssim": float(row["ssim_mean"]),
                                       "mse": float(row["mse_mean"]), "med": float(row.get("ssim_med", 0)),
                                       "q3": float(row.get("ssim_q3", 0))}
                    elif sname == "strict_sc":
                        strict_sc = {"n": int(row["n"]), "ssim": float(row["ssim_mean"]),
                                     "mse": float(row["mse_mean"]), "med": float(row.get("ssim_med", 0)),
                                     "q3": float(row.get("ssim_q3", 0))}

            # 结构化字段
            flat["origin"] = {"seen": seen_orig, "strict": strict_orig}
            flat["self_cond"] = {"seen": seen_sc, "strict": strict_sc, "blend_alpha": args.blend_alpha}
            if seen_orig:
                flat.update({"ssim": seen_orig["ssim"], "mse": seen_orig["mse"]})
            if strict_orig:
                flat["strict"] = {"n": strict_orig["n"], "ssim_mean": strict_orig["ssim"], "mse_mean": strict_orig["mse"]}
            if seen_orig and seen_sc:
                flat["delta_seen_ssim"] = round(seen_sc["ssim"] - seen_orig["ssim"], 4)
            if strict_orig and strict_sc:
                flat["delta_strict_ssim"] = round(strict_sc["ssim"] - strict_orig["ssim"], 4)

            # 写入单独的 eval_auto_{step}.json
            with open(os.path.join(ckpt_dir, f"eval_auto_{step}.json"), "w", encoding="utf-8") as f:
                json.dump(flat, f, ensure_ascii=False, indent=2)

            # 写入实验专属对比汇总 CSV (每次实验专门写到 eval_comparison.csv)
            comp_csv = os.path.join(root, "eval_comparison.csv")
            new_comp = not os.path.exists(comp_csv)
            try:
                with open(comp_csv, "a", newline="", encoding="utf-8") as f:
                    w_comp = csv.writer(f)
                    if new_comp:
                        w_comp.writerow(["step", "seen_orig_ssim", "seen_sc_ssim", "seen_delta",
                                         "strict_orig_ssim", "strict_sc_ssim", "strict_delta",
                                         "elapsed_s"])
                    w_comp.writerow([
                        step,
                        seen_orig["ssim"] if seen_orig else "",
                        seen_sc["ssim"] if seen_sc else "",
                        flat.get("delta_seen_ssim", ""),
                        strict_orig["ssim"] if strict_orig else "",
                        strict_sc["ssim"] if strict_sc else "",
                        flat.get("delta_strict_ssim", ""),
                        flat["elapsed_s"],
                    ])
            except Exception as e:
                log(f"write eval_comparison.csv failed: {e}")

            # 自动刷新统一登记处 (best-effort)
            try:
                import subprocess as _sp
                _sp.run([sys.executable, "tools/registry.py", "--build"],
                        cwd=BASE, capture_output=True, timeout=300)
            except Exception:
                pass
            log(f"step {step}: DONE origin_seen={flat.get('ssim')} sc_seen={seen_sc.get('ssim') if seen_sc else None} "
                f"delta={flat.get('delta_seen_ssim')}")
            # poster
            try:
                subprocess.run([sys.executable, "-u", "src/eval/posters.py", "--run-dir", seg,
                                "--step", str(step), "--sets", "seen,strict",
                                "--agg-sets", "seen,strict", "--tag", os.path.basename(seg)],
                               capture_output=True, text=True, timeout=900)
            except Exception as e:
                log(f"poster failed: {e!r}")
        if args.run:
            break
        time.sleep(args.poll)
    log("loop exited")


if __name__ == "__main__":
    main()
