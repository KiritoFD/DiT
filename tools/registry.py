# -*- coding: utf-8 -*-
"""registry.py — 历史 config + 结果 JSON 统一登记处 (assets/registry/).

- `--build`  : 全量扫描 src/train/configs/*.json 与 assets/results/**(含 _archive),
               生成 configs/ 快照 + runs.json/runs.csv (状态/大小/last_step/best seen/strict)。
- `--update --run <results_dir>` : 只刷新单个 run (供训练/评测循环自动调用)。
"""
import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import time

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG = os.path.join(D, "assets", "registry")
RES = os.path.join(D, "assets", "results")


def _du(p):
    r = subprocess.run(["du", "-sb", p], capture_output=True, text=True)
    s = r.stdout.split()[0] if r.stdout else "0"
    return int(s) if s.isdigit() else 0


def _eval_metrics(rd):
    bs = bst = None
    sm = glob.glob(f"{rd}/eval_stdskel_summary.csv")
    if sm:
        for r in csv.DictReader(open(sm[0], encoding="utf-8")):
            try:
                v = float(r["ssim_mean"]); st = int(float(r["step"])); s = r["set"]
            except Exception:
                continue
            if s == "seen" and (bs is None or v > bs[0]):
                bs = (v, st)
            if s == "strict" and (bst is None or v > bst[0]):
                bst = (v, st)
    for f in glob.glob(f"{rd}/*/checkpoints/eval_auto_*.json"):
        try:
            d = json.load(open(f)); st = int(d["step"]); v = d.get("ssim")
        except Exception:
            continue
        if v is not None and (bs is None or v > bs[0]):
            bs = (v, st)
        sv = (d.get("strict") or {}).get("ssim_mean")
        if sv is not None and (bst is None or sv > bst[0]):
            bst = (sv, st)
    return bs, bst


def _config_of(rd):
    """从 run 的 resolved_config 或第一个 ckpt 的 args 取关键配置。"""
    for f in glob.glob(f"{rd}/*/resolved_config.json"):
        try:
            d = json.load(open(f, encoding="utf-8"))
            return {k: d.get(k) for k in (
                "model", "global_batch_size", "lr", "max_steps", "w_repa",
                "glyph_inject_mode", "glyph_inject_layers", "aux_latent_shards_dirs",
                "aux_loss_weights", "w_latent_skel", "w_latent_canny",
                "skel_as_glyph_cond", "skel_latent_shards_dir", "latent_shards_dir",
                "data_csv", "no_char_cond", "freeze_callig_table")}
        except Exception:
            pass
    return {}


def _run_info(rd, status):
    steps = [int(re.search(r"(\d+)\.pt$", c).group(1))
             for c in glob.glob(f"{rd}/*/checkpoints/[0-9]*.pt")
             if re.search(r"(\d+)\.pt$", c)]
    bs, bst = _eval_metrics(rd)
    return {
        "run": os.path.basename(rd).rstrip("/"),
        "status": status,
        "path": os.path.relpath(rd, D).replace("\\", "/"),
        "size_bytes": _du(rd),
        "n_ckpt": len(steps),
        "last_step": max(steps) if steps else -1,
        "best_seen": (f"{bs[0]:.4f}@{bs[1]}" if bs else None),
        "best_strict": (f"{bst[0]:.4f}@{bst[1]}" if bst else None),
        "config": _config_of(rd),
    }


def build(write=True):
    os.makedirs(os.path.join(REG, "configs"), exist_ok=True)
    # 1) config 快照
    n_cfg = 0
    for f in glob.glob(os.path.join(D, "src", "train", "configs", "*.json")):
        try:
            shutil.copy2(f, os.path.join(REG, "configs", os.path.basename(f)))
            n_cfg += 1
        except Exception:
            pass
    # 2) run 表
    runs = []
    for rd in sorted(glob.glob(f"{RES}/*")):
        if os.path.isdir(rd) and not os.path.basename(rd).startswith("_"):
            runs.append(_run_info(rd, "keep"))
    for rd in sorted(glob.glob(f"{RES}/_archive/*")):
        if os.path.isdir(rd) and not os.path.basename(rd).startswith("_"):
            runs.append(_run_info(rd, "archived"))
    out = {"updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "n_configs": n_cfg, "runs": runs}
    if write:
        with open(os.path.join(REG, "runs.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        with open(os.path.join(REG, "runs.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["run", "status", "n_ckpt", "last_step", "best_seen", "best_strict",
                        "size_GB", "model", "batch", "lr", "max_steps", "w_repa",
                        "inject", "aux", "aux_w"])
            for r in runs:
                c = r["config"]
                w.writerow([r["run"], r["status"], r["n_ckpt"], r["last_step"],
                            r["best_seen"], r["best_strict"], round(r["size_bytes"] / 1e9, 1),
                            c.get("model"), c.get("global_batch_size"), c.get("lr"),
                            c.get("max_steps"), c.get("w_repa"), c.get("glyph_inject_mode"),
                            c.get("aux_latent_shards_dirs"), c.get("aux_loss_weights")])
    print(f"[registry] configs={n_cfg} runs={len(runs)} -> {os.path.relpath(REG, D)}/")
    return out


def update_run(results_dir):
    """单个 run 刷新 (供训练/评测自动调用); 若 registry 不存在则全量 build。"""
    if not os.path.exists(os.path.join(REG, "runs.json")):
        return build()
    return build()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--run", default="")
    a = ap.parse_args()
    if a.update and a.run:
        update_run(a.run)
    else:
        build()
