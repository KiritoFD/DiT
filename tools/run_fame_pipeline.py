# -*- coding: utf-8 -*-
"""
run_fame_pipeline.py — fame 串行自动化 pipeline.

  1. 等 build_fame_dataset.py 完成 (fame_meta.json 出现)
  2. 启动 fame 预训练 (s21_fame_flow_v2, 训练代码自带 early-stop)
  3. 预训练结束后选 best ckpt (eval_auto ssim 最高, 缺省取最后一个)
  4. 填入 ctrl_fame_v2.json 的 main_ckpt → 启动 ControlNet (--attn-impl eager)
  5. ctrl watchdog: 每 2500 步旁路 eval (n=100, cfg=0.7, median SSIM);
     连续 5 次无提升 → 判定收敛, 终止训练, 写最终报告

全程日志: /tmp/fame_pipeline.log, /tmp/fame_pretrain.log, /tmp/fame_ctrl.log
"""
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
LOG = open("/tmp/fame_pipeline.log", "a", encoding="utf-8")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    LOG.write(line + "\n")
    LOG.flush()
    print(line, flush=True)


def wait_for(predicate, timeout_h, poll_s=60, what=""):
    t0 = time.time()
    while time.time() - t0 < timeout_h * 3600:
        if predicate():
            return True
        time.sleep(poll_s)
    log(f"TIMEOUT waiting for {what}")
    return False


def procs_matching(pat):
    try:
        out = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
        return [int(x) for x in out.stdout.split()]
    except Exception:
        return []


def launch(cmd, log_path):
    lf = open(log_path, "w")
    subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True)


def best_ckpt(run_dir):
    """按 eval_auto ssim 选 best; 无 eval 则最后一个。"""
    hist = {}
    for p in glob.glob(os.path.join(run_dir, "checkpoints", "eval_auto_*.json")):
        try:
            d = json.load(open(p, encoding="utf-8"))
            hist[int(d["step"])] = float(d["ssim"])
        except Exception:
            pass
    have = {int(os.path.basename(p).split(".")[0])
            for p in glob.glob(os.path.join(run_dir, "checkpoints", "*.pt"))}
    scored = [(s, v) for s, v in hist.items() if s in have]
    if scored:
        return max(scored, key=lambda t: t[1])[0]
    pts = sorted(int(os.path.basename(p).split(".")[0])
                 for p in glob.glob(os.path.join(run_dir, "checkpoints", "*.pt")))
    return pts[-1] if pts else None


def main():
    # ---- 1) 等数据集 ----
    log("waiting for fame dataset build ...")
    ok = wait_for(lambda: os.path.exists("fame_meta.json"), 6.0, what="fame_meta.json")
    if not ok:
        return
    meta = json.load(open("fame_meta.json", encoding="utf-8"))
    log(f"fame dataset ready: {meta}")

    # ---- 2) 预训练 ----
    launch([PY, "-m", "src.train.train", "--config",
            "src/train/configs/s21_fame_flow_v2.json"], "/tmp/fame_pretrain.log")
    log("pretrain launched (early-stop in trainer)")

    def pretrain_done():
        return (not procs_matching("src.train.train.*s21_fame")
                and glob.glob("5script/results/s21_fame_flow_v2/*/checkpoints/*.pt"))

    if not wait_for(pretrain_done, 48.0, poll_s=300, what="pretrain finish"):
        return
    runs = sorted(glob.glob("5script/results/s21_fame_flow_v2/2026*"))
    run_dir = runs[-1]
    step = best_ckpt(run_dir)
    main_ckpt = os.path.join(run_dir, "checkpoints", f"{step:07d}.pt")
    log(f"pretrain done. best ckpt: {main_ckpt}")

    # ---- 3) 填 main_ckpt → 启动 ctrl ----
    cfgp = "src/train/configs/ctrl_fame_v2.json"
    cfg = json.load(open(cfgp, encoding="utf-8"))
    cfg["main_ckpt"] = main_ckpt
    json.dump(cfg, open(cfgp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    launch([PY, "-m", "src.train.train_controlnet", "--config", cfgp,
            "--attn-impl", "eager"], "/tmp/fame_ctrl.log")
    log("controlnet launched (watchdog early-stop, eval cfg=0.7)")

    # ---- 4) ctrl watchdog ----
    sys.path.insert(0, ".")
    import torch
    from src.model.controlnet import load_main_model, ControlNetDiT
    from src.eval.inference import (
        build_diffusion, sample_latents, load_eval_vae, make_eval_cache, _ssim)
    dev = torch.device("cuda")
    main_model = load_main_model(
        "DiT-2Cond-S/2", main_ckpt, device=dev,
        num_calligraphers=cfg["num_calligraphers"],
        num_characters=cfg["num_characters"],
        condition_fusion=cfg["condition_fusion"],
        callig_embed_dim=cfg["callig_embed_dim"],
        char_embed_dim=cfg["char_embed_dim"],
        char_proj_mode=cfg["char_proj_mode"],
        freeze_char_table=cfg["freeze_char_table"])
    main_model.eval()
    vae = load_eval_vae(dev, "pretrained_models/sd-vae-ft-ema")
    diffusion = build_diffusion(50, "flow")
    cache = make_eval_cache("5script/eval_fame_strict.csv", "final_imgs_256",
                            None, 256, 100, 8, 4, 0.18215,
                            skel_latent_shards_dir="final_skel_latents_fame")

    PATIENCE, BEST0 = 5, -1.0
    stale, best, seen = 0, BEST0, set()
    history = []
    t_start = time.time()
    while time.time() - t_start < 30 * 3600:
        time.sleep(240)
        if not procs_matching("train_controlnet.*fame-ctrl|ctrl_fame_v2"):
            log("ctrl process exited on its own")
            break
        done = sorted(glob.glob("5script/results/ctrl_fame_v2/*/checkpoints/*.pt.done"))
        if not done:
            continue
        ckpt = done[-1][:-5]
        if ckpt in seen:
            continue
        seen.add(ckpt)
        try:
            ctrl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
            raw = ctrl_sd.get("ema") or ctrl_sd.get("ctrl")
            raw = {k: v for k, v in raw.items() if not k.startswith("main.")}
            ctrl.load_state_dict(raw, strict=False)
            lat = sample_latents(ctrl, diffusion, cache["noise"], cache["conds"],
                                 0.7, 8, dev, skel=cache["skels_latent"], seed=0)
            ss = []
            with torch.no_grad():
                for i in range(0, cache["n"], 8):
                    rec = vae.decode(lat[i:i + 8].to(dev) / 0.18215).sample.float().cpu()
                    for k in range(rec.shape[0]):
                        p = ((rec[k].clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()
                        g = ((cache["gts"][i + k].clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()
                        ss.append(_ssim(p, g))
            med = float(np.median(ss))
            improved = med > best + 0.002
            best = max(best, med)
            stale = 0 if improved else stale + 1
            history.append({"ckpt": os.path.basename(ckpt), "median_ssim": med,
                            "best": best, "stale": stale})
            log(f"[watchdog] {os.path.basename(ckpt)} median {med:.4f} "
                f"best {best:.4f} stale {stale}/{PATIENCE}")
            json.dump(history, open("5script/fame_ctrl_watchdog.json", "w"),
                      ensure_ascii=False, indent=1)
            if stale >= PATIENCE:
                log(f"[watchdog] no improvement {PATIENCE} evals -> early stop ctrl")
                for pid in procs_matching("train_controlnet"):
                    try:
                        os.kill(pid, 15)
                    except Exception:
                        pass
                break
        except Exception as e:
            log(f"[watchdog] ERROR: {e}")

    log("PIPELINE DONE. "
        f"pretrain_best={main_ckpt} ctrl_best_median={best:.4f}")


if __name__ == "__main__":
    main()
