# -*- coding: utf-8 -*-
"""watch_pretrain_launch_ctrl.py — 等 s21 预训练早停 → 自动拉起 fame ControlNet + watchdog 早停."""
import os
import sys
import json
import time
import glob
import subprocess

import numpy as np

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")
PY = "/opt/conda/bin/python"
LOG = open("/tmp/fame_ctrl_pipeline.log", "a", encoding="utf-8")


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    LOG.write(line + "\n")
    LOG.flush()
    print(line, flush=True)


def procs(pat):
    out = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
    return [int(x) for x in out.stdout.split()]


def main():
    # ---- 1) 等预训练结束 (early-stop / Done) ----
    log("watching s21 pretrain ...")
    while True:
        logp = "/tmp/s21_pretrain.log"
        txt = open(logp, encoding="utf-8", errors="ignore").read()
        if "Done!" in txt or "[early-stop]" in txt and "Done!" in txt:
            break
        if not procs("src[.]train[.]train"):
            log("pretrain process gone")
            break
        time.sleep(300)
    best_log = sorted(set(re.findall(r"NEW BEST \(best_ssim=([0-9.]+)", txt))) if False else None
    log("pretrain finished")

    # ---- 2) best ckpt ----
    runs = sorted(glob.glob("5script/results/s21_fame_flow_v2/2026*"))
    run_dir = runs[-1]
    hist = {}
    for p in glob.glob(os.path.join(run_dir, "checkpoints", "eval_auto_*.json")):
        try:
            d = json.load(open(p, encoding="utf-8"))
            hist[int(d["step"])] = float(d["ssim"])
        except Exception:
            pass
    have = {int(os.path.basename(p).split(".")[0])
            for p in glob.glob(os.path.join(run_dir, "checkpoints", "*.pt"))}
    scored = sorted(((s, v) for s, v in hist.items() if s in have),
                    key=lambda t: t[1])
    step = scored[-1][0] if scored else max(have)
    main_ckpt = os.path.join(run_dir, "checkpoints", f"{step:07d}.pt")
    log(f"best ckpt: {main_ckpt} (ssim={scored[-1][1] if scored else 'n/a'})")

    # ---- 2.5) 等 skel latents 就绪 (20 shards) ----
    log("waiting for final_skel_latents_fame shards ...")
    t0 = time.time()
    while len(glob.glob("final_skel_latents_fame/shard_*.npz")) < 20:
        if time.time() - t0 > 6 * 3600:
            log("TIMEOUT waiting skel latents")
            return
        time.sleep(120)
    log("skel latents ready")

    # ---- 3) 配置 + 拉起 ctrl ----
    cfgp = "src/train/configs/ctrl_fame_v2.json"
    cfg = json.load(open(cfgp, encoding="utf-8"))
    cfg["main_ckpt"] = main_ckpt
    json.dump(cfg, open(cfgp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    # 若 ctrl 已在跑则不重复拉起; 否则支持从上次 ckpt resume
    if procs("train_controlnet"):
        log("ctrl already running -> skip launch")
    else:
        prev = sorted(glob.glob("5script/results/ctrl_fame_v2/*/checkpoints/*.pt"))
        cmd = [PY, "-m", "src.train.train_controlnet", "--config", cfgp,
               "--attn-impl", "eager"]
        if prev:
            cmd += ["--resume", prev[-1]]
            log(f"ctrl resume from {os.path.basename(prev[-1])}")
        lf = open("/tmp/fame_ctrl.log", "a")
        subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
        log("fame ctrl launched (eval cfg=0.7, watchdog patience 5)")
    time.sleep(60)  # 等 ctrl 完成初始化, 避免首个 watchdog eval 撞车

    # ---- 4) ctrl watchdog: 轮询 daemon 写的 eval_auto_*.json (无 GPU 争用) ----
    PATIENCE, best, stale = 5, -1.0, 0
    seen, history = set(), []
    t0 = time.time()
    while time.time() - t0 < 30 * 3600:
        time.sleep(300)
        if not procs("train_controlnet"):
            log("ctrl exited")
            break
        hist = {}
        for p in glob.glob("5script/results/ctrl_fame_v2/*/checkpoints/eval_auto_*.json"):
            try:
                d = json.load(open(p, encoding="utf-8"))
                hist[int(d["step"])] = float(d["ssim"])
            except Exception:
                pass
        scored = sorted(hist.items())
        if not scored:
            continue
        step, cur = scored[-1]
        if step in seen:
            continue
        seen.add(step)
        improved = cur > best + 0.002
        best = max(best, cur)
        stale = 0 if improved else stale + 1
        history.append({"step": step, "ssim": cur, "best": best, "stale": stale})
        log(f"[watchdog] step {step} ssim {cur:.4f} best {best:.4f} stale {stale}/{PATIENCE}")
        json.dump(history, open("5script/fame_ctrl_watchdog.json", "w"),
                  ensure_ascii=False, indent=1)
        if stale >= PATIENCE:
            log("converged -> stopping ctrl")
            for pid in procs("train_controlnet"):
                try:
                    os.kill(pid, 15)
                except Exception:
                    pass
            break
    log(f"PIPELINE DONE ctrl_best_ssim={best:.4f}")


if __name__ == "__main__":
    import re
    main()
