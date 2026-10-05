import csv
import glob
import json
import os
import re

moyi_dir = "docs/106moyi"
out_csv = os.path.join(moyi_dir, "moyi_experiments_summary.csv")

print("=" * 80)
print(f"Scanning {moyi_dir} to build {out_csv}...")
print("=" * 80)

records = []

# 1. Scan metrics_40k.json and eval_metrics.json
for f in sorted(glob.glob(f"{moyi_dir}/**/*.json", recursive=True)):
    fn = os.path.basename(f)
    if "config" in f or "manifest" in f or "vocab" in f or "remap" in f:
        continue

    try:
        with open(f, "r", encoding="utf-8") as fp:
            d = json.load(fp)
    except Exception:
        continue

    rel_p = os.path.relpath(f, moyi_dir).replace("\\", "/")

    # Check if this is an evaluation metric JSON
    if "strict" in d or "ssim_mean" in d or "ssim" in d:
        rec = {
            "source_file": rel_p,
            "experiment": "",
            "model_type": "",
            "params": "",
            "step": "",
            "strict_ssim": "",
            "strict_ssim_med": "",
            "strict_lpips": "",
            "strict_skel_iou": "",
            "strict_mse": "",
            "seen_ssim": "",
            "seen_ssim_med": "",
            "seen_lpips": "",
            "seen_skel_iou": "",
            "gap_ssim": "",
        }

        # Determine experiment name from path
        if "exp1_rmsnorm" in rel_p:
            rec["experiment"] = "Phase_A_Exp1_RMSNorm"
            rec["model_type"] = "DiT-2Cond-S/2 (RMSNorm)"
            rec["params"] = "33.8M"
            rec["step"] = "10000"
        elif "tier2_sp" in rel_p:
            rec["experiment"] = "Capacity_Tier2_Sp"
            rec["model_type"] = "DiT-2Cond-Sp/2 (LayerNorm)"
            rec["params"] = "59.2M"
            rec["step"] = "10000" if "eval_10k" in rel_p else "20000"
        elif "tier3_b" in rel_p and "eval_5k" in rel_p:
            rec["experiment"] = "Capacity_Tier3_B"
            rec["model_type"] = "DiT-2Cond-B/2 (LayerNorm)"
            rec["params"] = "131.0M"
            rec["step"] = "5000"
        elif "moyi_top10_rf_4ch" in rel_p:
            rec["experiment"] = "Moyi_Top10_RF_4ch"
            rec["model_type"] = "Moyi-RF-4ch"
            rec["params"] = "SD-VAE 4ch"
            rec["step"] = "50000" if "step50000" in rel_p else "80000"
        elif "moyi_top10_rf" in rel_p:
            rec["experiment"] = "Moyi_Top10_RF_16ch"
            rec["model_type"] = "Moyi-RF-16ch (FLUX VAE)"
            rec["params"] = "FLUX 16ch"
            rec["step"] = "50000"

        # Parse metrics format A (metrics_40k.json)
        if "strict" in d and isinstance(d["strict"], dict):
            st = d["strict"]
            rec["strict_ssim"] = round(st.get("ssim_mean", 0), 4)
            rec["strict_ssim_med"] = round(st.get("ssim_med", 0), 4)
            rec["strict_lpips"] = round(st.get("lpips_mean", 0), 4)
            rec["strict_skel_iou"] = round(st.get("skel_iou_mean", 0), 4)
            rec["strict_mse"] = round(st.get("mse_mean", 0), 4)

            if "seen" in d and isinstance(d["seen"], dict):
                se = d["seen"]
                rec["seen_ssim"] = round(se.get("ssim_mean", 0), 4)
                rec["seen_ssim_med"] = round(se.get("ssim_med", 0), 4)
                rec["seen_lpips"] = round(se.get("lpips_mean", 0), 4)
                rec["seen_skel_iou"] = round(se.get("skel_iou_mean", 0), 4)

            if "gap" in d and isinstance(d["gap"], dict):
                rec["gap_ssim"] = round(d["gap"].get("ssim", 0), 4)

        # Parse metrics format B (eval_metrics.json / eval_auto.json)
        elif "ssim_mean" in d or "ssim" in d:
            rec["strict_ssim"] = round(d.get("ssim_mean", d.get("ssim", 0)), 4)
            rec["strict_ssim_med"] = round(d.get("ssim_med", 0), 4)
            rec["strict_lpips"] = round(
                d.get("lpips_mean", d.get("lpips", 0)), 4
            )
            rec["strict_mse"] = round(d.get("mse_mean", d.get("mse", 0)), 4)

        records.append(rec)

# Sort records logically
records.sort(
    key=lambda x: (x["experiment"], int(x["step"]) if x["step"] else 0)
)

fieldnames = [
    "experiment",
    "model_type",
    "params",
    "step",
    "strict_ssim",
    "strict_ssim_med",
    "strict_lpips",
    "strict_skel_iou",
    "strict_mse",
    "seen_ssim",
    "seen_ssim_med",
    "seen_lpips",
    "seen_skel_iou",
    "gap_ssim",
    "source_file",
]

with open(out_csv, "w", encoding="utf-8", newline="") as fp:
    w = csv.DictWriter(fp, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(records)

print(f"Generated {out_csv} with {len(records)} benchmark records:")
for r in records:
    print(
        f"  {r['experiment']:<24} | Step {r['step']:<5} | SSIM: {r['strict_ssim']:<6} | LPIPS: {r['strict_lpips']:<6} | Skel: {r['strict_skel_iou']:<6}"
    )
print("=" * 80)
