"""Aggregate all eval points from all experiments into results.csv.

Sources:
- For each series dir under 5script/results/<series>/<run>/:
  - run/resolved_config.json (or run/checkpoints ckpt args / series config json)
  - run/checkpoints/eval_auto_*.json (each eval point)
  - run/checkpoints/cpu_eval_state.json (s6-style evals without json per point)
"""
import os, json, sys, glob, csv, re

ROOT = "/root/Workspace/xy/DiT"
RESULTS = os.path.join(ROOT, "5script/results")

# ---- helpers ----
def safe(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None and d[k] != "":
            return d[k]
    return default

def load_json(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def series_config_fallback(series):
    """Look for a config json at repo root / configs_* for this series."""
    candidates = []
    for root, _, files in os.walk(os.path.join(ROOT, "configs_s11")):
        for fn in files:
            candidates.append(os.path.join(root, fn))
    for fn in os.listdir(ROOT):
        if fn.endswith(".json") and (series in fn or fn.startswith("exp_")):
            candidates.append(os.path.join(ROOT, fn))
    return candidates

def extract_config(run_dir, series, ckpt_args=None):
    """Build a config dict for this run."""
    cfg = {}
    rj = load_json(os.path.join(run_dir, "resolved_config.json"))
    # resolved_config may be {} / None → treat as missing
    if rj and isinstance(rj, dict) and len(rj) > 2:
        cfg = dict(rj)
        cfg["_src"] = "resolved_config"
    elif ckpt_args:
        cfg = dict(vars(ckpt_args)) if hasattr(ckpt_args, "__dict__") else dict(ckpt_args)
        cfg["_src"] = "ckpt_args"
    if not cfg:
        # fallback: series-level config (configs_s11/s11_top6_p4.json, root exp json, etc.)
        for cand in series_config_fallback(series):
            c = load_json(cand)
            if not c or not isinstance(c, dict):
                continue
            cn = str(c.get("experiment_name", ""))
            if cn and cn in series:
                cfg = dict(c)
                cfg["_src"] = cand
                break
            if series in cand or series in str(c.get("config", "")):
                cfg = dict(c)
                cfg["_src"] = cand
                break
    return cfg

# ---- collect all runs ----
series_list = sorted([d for d in os.listdir(RESULTS)
                      if os.path.isdir(os.path.join(RESULTS, d))])

rows = []
for series in series_list:
    sdir = os.path.join(RESULTS, series)
    runs = sorted([d for d in os.listdir(sdir)
                   if os.path.isdir(os.path.join(sdir, d))])
    for run in runs:
        run_dir = os.path.join(sdir, run)
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        if not os.path.isdir(ckpt_dir):
            continue
        # config
        ckpt_args = None
        pts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
        for pt in pts[:1]:
            try:
                import torch
                ck = torch.load(pt, map_location="cpu", weights_only=False)
                ckpt_args = ck.get("args")
                if isinstance(ckpt_args, dict):
                    break
            except Exception:
                pass
        cfg = extract_config(run_dir, series, ckpt_args)
        if not cfg:
            print(f"WARN no config for {series}/{run}", file=sys.stderr)

        # eval points: eval_auto_*.json (优先)
        evals = sorted(glob.glob(os.path.join(ckpt_dir, "eval_auto_*.json")))
        eval_steps = set()
        for ev in evals:
            d = load_json(ev)
            if not d:
                continue
            step = d.get("step")
            if step is None:
                m = re.search(r"(\d+)", os.path.basename(ev))
                step = int(m.group(1)) if m else None
            if step is None:
                continue
            eval_steps.add(step)
            row = {
                "series": series, "run": run, "step": step,
                "mse": d.get("mse"), "ssim": d.get("ssim"),
                "ssim_old_uniform": d.get("ssim_old_uniform"),
                "skel_iou": d.get("skel_iou"), "lpips": d.get("lpips"),
                "cfg": cfg, "src": "eval_auto",
            }
            # ctrl-style: mse_base/ssim_base + mse_ctrl/ssim_ctrl
            if "ssim_base" in d or "ssim_ctrl" in d:
                row["mse"] = None
                row["ssim"] = None
                row["mse_base"] = d.get("mse_base")
                row["ssim_base"] = d.get("ssim_base")
                row["mse_ctrl"] = d.get("mse_ctrl")
                row["ssim_ctrl"] = d.get("ssim_ctrl")
                row["skel_iou_base"] = d.get("skel_iou_base")
                row["skel_iou_ctrl"] = d.get("skel_iou_ctrl")
                row["lpips_base"] = d.get("lpips_base")
                row["lpips_ctrl"] = d.get("lpips_ctrl")
                row["delta_ssim"] = d.get("delta_ssim")
                row["delta_mse"] = d.get("delta_mse")
                row["delta_skel_iou"] = d.get("delta_skel_iou")
                row["delta_lpips"] = d.get("delta_lpips")
            rows.append(row)
        # cpu_eval_state.json fallback: 只补 eval_auto 没有的 step
        state = load_json(os.path.join(ckpt_dir, "cpu_eval_state.json"))
        if state:
            for pt_name, v in state.items():
                if isinstance(v, dict) and v.get("ok"):
                    step = v.get("step") or int(re.sub(r"\D", "", pt_name) or 0)
                    if step in eval_steps:
                        continue
                    rows.append({
                        "series": series, "run": run, "step": step,
                        "mse": v.get("mse"), "ssim": v.get("ssim"),
                        "ssim_old_uniform": None, "skel_iou": v.get("skel_iou"),
                        "lpips": v.get("lpips"), "cfg": cfg,
                        "src": "cpu_eval_state",
                    })

# ---- flatten config into columns ----
def cfg_get(row, key):
    return safe(row["cfg"], key, default="")

def pick(*cands, default=""):
    for c in cands:
        if c not in (None, "", "None"):
            return c
    return default

out_rows = []
for r in rows:
    cfg = r["cfg"]
    c = r.copy()
    # identity
    c["series"] = r["series"]
    c["run"] = r["run"]
    c["step"] = r["step"]
    # model/data
    c["model"] = pick(cfg_get(r, "model"), "DiT-2Cond-S/2")
    c["vae"] = pick(cfg_get(r, "vae_path"), cfg_get(r, "vae"), "sd-vae-ft-ema")
    c["latent_channels"] = pick(cfg_get(r, "latent_channels"), 4)
    c["latent_spatial"] = pick(cfg_get(r, "latent_spatial"), 32)
    c["vae_scaling_factor"] = pick(cfg_get(r, "vae_scaling_factor"), 0.18215)
    c["data_csv"] = pick(cfg_get(r, "data_csv"), cfg_get(r, "csv"), "")
    c["img_root"] = pick(cfg_get(r, "img_root"), "")
    c["char_embed_dim"] = pick(cfg_get(r, "char_embed_dim"), "")
    c["callig_embed_dim"] = pick(cfg_get(r, "callig_embed_dim"), "")
    c["num_characters"] = pick(cfg_get(r, "num_characters"), "")
    c["dino"] = "dino" if str(cfg_get(r, "char_dino_embeddings")).lower().find("dino") >= 0 else "none"
    c["cond_mode"] = pick(cfg_get(r, "cond_mode"), "")
    c["condition_fusion"] = pick(cfg_get(r, "condition_fusion"), "")
    c["sampler"] = pick(cfg_get(r, "sampler"), "")
    c["use_skel"] = pick(cfg_get(r, "use_skel"), cfg_get(r, "w_skel"), 0)
    c["skel_root"] = pick(cfg_get(r, "skel_root"), "")
    c["w_skel"] = pick(cfg_get(r, "w_skel"), 0)
    c["use_canny"] = pick(cfg_get(r, "use_canny"), cfg_get(r, "w_canny"), 0)
    c["w_canny"] = pick(cfg_get(r, "w_canny"), 0)
    c["glyph_init_mix"] = pick(cfg_get(r, "glyph_init_mix"), 1.0)
    c["w_glyph_cond"] = pick(cfg_get(r, "w_glyph_cond"), False)
    c["balance_callig_alpha"] = pick(cfg_get(r, "balance_callig_alpha"), 0)
    c["balance_char_alpha"] = pick(cfg_get(r, "balance_char_alpha"), 0)
    # train
    c["lr"] = pick(cfg_get(r, "lr"), "")
    c["weight_decay"] = pick(cfg_get(r, "weight_decay"), "")
    c["warmup_steps"] = pick(cfg_get(r, "warmup_steps"), "")
    c["max_steps"] = pick(cfg_get(r, "max_steps"), cfg_get(r, "epochs"), "")
    c["batch_size"] = pick(cfg_get(r, "batch_size"), cfg_get(r, "global_batch_size"), "")
    c["ema_decay"] = pick(cfg_get(r, "ema_decay"), "")
    c["resume"] = pick(cfg_get(r, "resume_full"), cfg_get(r, "resume"), cfg_get(r, "pretrained"), "")
    c["train_ctrl_only"] = pick(cfg_get(r, "train_ctrl_only"), "")
    c["main_ckpt"] = pick(cfg_get(r, "main_ckpt"), "")
    c["skel_cond_channels"] = pick(cfg_get(r, "skel_cond_channels"), "")
    c["cond_drop_struct_prob"] = pick(cfg_get(r, "cond_drop_struct_prob"), "")
    c["warmup_steps"] = pick(cfg_get(r, "warmup_steps"), "")
    c["lr_schedule"] = pick(cfg_get(r, "lr_schedule"), "")
    c["min_lr_ratio"] = pick(cfg_get(r, "min_lr_ratio"), "")
    # eval
    c["eval_csv"] = pick(cfg_get(r, "eval_csv"), "")
    c["eval_n"] = pick(cfg_get(r, "eval_n"), "")
    c["eval_steps"] = pick(cfg_get(r, "eval_steps"), "")
    c["eval_cfg"] = pick(cfg_get(r, "eval_cfg"), "")
    c["eval_seed"] = pick(cfg_get(r, "eval_seed"), "")
    # metrics (gaussian SSIM 修正后)
    c["mse"] = r.get("mse")
    c["ssim"] = r.get("ssim")
    c["ssim_old_uniform"] = r.get("ssim_old_uniform")
    c["skel_iou"] = r.get("skel_iou")
    c["lpips"] = r.get("lpips")
    c["mse_base"] = r.get("mse_base")
    c["ssim_base"] = r.get("ssim_base")
    c["mse_ctrl"] = r.get("mse_ctrl")
    c["ssim_ctrl"] = r.get("ssim_ctrl")
    c["skel_iou_base"] = r.get("skel_iou_base")
    c["skel_iou_ctrl"] = r.get("skel_iou_ctrl")
    c["lpips_base"] = r.get("lpips_base")
    c["lpips_ctrl"] = r.get("lpips_ctrl")
    c["delta_ssim"] = r.get("delta_ssim")
    c["delta_mse"] = r.get("delta_mse")
    c["delta_skel_iou"] = r.get("delta_skel_iou")
    c["delta_lpips"] = r.get("delta_lpips")
    c["src"] = r["src"]
    out_rows.append(c)

# ---- write CSV ----
cols = [
    "series", "run", "step",
    "model", "vae", "latent_channels", "latent_spatial", "vae_scaling_factor",
    "data_csv", "img_root", "char_embed_dim", "callig_embed_dim", "num_characters", "dino",
    "cond_mode", "condition_fusion", "sampler",
    "use_skel", "skel_root", "w_skel", "use_canny", "w_canny",
    "glyph_init_mix", "w_glyph_cond", "balance_callig_alpha", "balance_char_alpha",
    "lr", "weight_decay", "warmup_steps", "max_steps", "batch_size", "ema_decay",
    "lr_schedule", "min_lr_ratio", "resume", "train_ctrl_only", "main_ckpt",
    "skel_cond_channels", "cond_drop_struct_prob",
    "eval_csv", "eval_n", "eval_steps", "eval_cfg", "eval_seed",
    "mse", "ssim", "ssim_old_uniform", "skel_iou", "lpips",
    "mse_base", "ssim_base", "mse_ctrl", "ssim_ctrl",
    "skel_iou_base", "skel_iou_ctrl", "lpips_base", "lpips_ctrl",
    "delta_ssim", "delta_mse", "delta_skel_iou", "delta_lpips",
    "src",
]

out = os.path.join(ROOT, "results.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in sorted(out_rows, key=lambda x: (x["series"], x["step"])):
        w.writerow(r)

print(f"Wrote {len(out_rows)} rows to {out}")
# summary
from collections import Counter
cnt = Counter((r["series"]) for r in out_rows)
for s, n in sorted(cnt.items()):
    print(f"  {s}: {n} points")
