# -*- coding: utf-8 -*-
"""gpu_eval_loop.py — GPU in-mem 评测循环 (替代 CPU daemon).

特性:
  - 与训练**并发** (seen: dit16/vae8 ≈3s/2.2G; strict: dit8/vae4 ≈76s/1.7G)
  - in-mem 指标 (无 PNG 必需; --save-samples 落盘供 poster)
  - 自动写 eval_auto_<step>.json + 生成 poster
  - --device cuda|cpu 开关 (cpu 走原 CPU 路径, 慢)

用法:
  python tools/eval/gpu_eval_loop.py --results-dir 5script/results/<exp> \
      [--device cuda] [--strict-every 10000] [--poll 30]
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

BASE = "/root/Workspace/xy/DiT"
sys.path.insert(0, BASE)
os.chdir(BASE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] [gpu-eval] {m}", flush=True)


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
    ap.add_argument("--seen-csv", default="5script/eval_seen_v10.csv")
    ap.add_argument("--strict-csv", default="5script/eval_fame3_strict_clean_v9.csv")
    ap.add_argument("--run", action="store_true", help="跑一次遍历后退出 (debug)")
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
            # 并发安全 batch: seen dit16/vae8; strict dit8/vae4
            batches = ["--dit-batch", "16", "--vae-batch", "8"] if len(sets) == 1 else \
                      ["--dit-batch", "8", "--vae-batch", "4"]
            cmd = [sys.executable, "-u", "tools/eval/eval_stdskel_batch.py",
                   "--results-dir", root, "--ckpt-override", ck, "--device", args.device,
                   "--save-samples", "--sets", *sets] + batches
            log(f"step {step}: eval {sets} on {args.device} ...")
            t0 = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                log(f"step {step}: eval FAILED rc={r.returncode} :: {r.stderr.strip()[-300:]}")
                break
            tail = [l for l in r.stdout.splitlines() if "step" in l or "peak" in l][-4:]
            for l in tail:
                log("  " + l)
            # 汇总 eval_auto json (seen ssim + strict)
            flat = {"step": step, "elapsed_s": round(time.time() - t0, 1),
                    "engine": f"gpu_inmem_{args.device}"}
            if os.path.exists(sum_path):
                for row in csv.DictReader(open(sum_path, encoding="utf-8")):
                    if int(row["step"]) != step:
                        continue
                    if row["set"] == "seen":
                        flat.update({"ssim": float(row["ssim_mean"]), "mse": float(row["mse_mean"]),
                                     "lpips": float(row["lpips_mean"]) if row.get("lpips_mean") else None})
                    elif row["set"] == "strict":
                        flat["strict"] = {"n": int(row["n"]), "ssim_mean": float(row["ssim_mean"]),
                                          "mse_mean": float(row["mse_mean"])}
            with open(os.path.join(ckpt_dir, f"eval_auto_{step}.json"), "w", encoding="utf-8") as f:
                json.dump(flat, f, ensure_ascii=False)
            log(f"step {step}: DONE seen={flat.get('ssim')} strict="
                f"{(flat.get('strict') or {}).get('ssim_mean')}")
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
