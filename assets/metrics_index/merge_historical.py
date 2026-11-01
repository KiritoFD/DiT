"""Merge all verified historical panels into the paired dataset."""
import csv
import json
from pathlib import Path

OUT = Path("assets/metrics_index/paired_eg_ocr_20261101.csv")
HIST = Path("assets/metrics_index/historical_eval.json")
BASE = Path("assets/metrics_index/paired_eg_ocr_remote.csv")

# directory -> (arm label, step)
NAME = {
    "assets/aligned_eval_results/Arm_2_50k_dist_aligned": ("Aligned-Arm2", 50000),
    "assets/aligned_eval_results/Base_55k_dist_aligned": ("U100-Base", 55000),
    "assets/aligned_eval_results/DiC_50k_dist_aligned": ("DiC", 50000),
    "assets/aligned_eval_results/DiC_80k_dist_aligned": ("DiC", 80000),
    "assets/aligned_eval_results/Exp_A_50k_dist_aligned": ("ExpA", 50000),
    "assets/aligned_eval_results/Exp_A_70k_dist_aligned": ("ExpA", 70000),
    "assets/recent_poster_20261017/png/l20_160": ("L-prior", 160000),
    "assets/recent_poster_20261017/png/l20_190": ("L-prior", 190000),
    "assets/recent_poster_20261017/png/l20_200": ("L-prior", 200000),
    "assets/recent_poster_20261017/png/l20_256": ("L-prior", 256000),
    "assets/recent_poster_20261017/png/std": ("std-input", 0),
    "assets/recent_poster_20261017/png/v84A": ("v84A", 200000),
    "assets/results/_diag_v83A_20261012_2018/step55000": ("v83A-diag", 55000),
    "assets/results/_local_v80_50k_ode/eval_samples_ctrl/step0050000/eval200fix": ("v80-ODE", 50000),
    "assets/results/_local_v80_50k_sde_g05/eval_samples_ctrl/step0050000/eval200fix": ("v80-SDE", 50000),
    "assets/results/v72_callivae_kl1e6_s12500_frozen_sp_c2ot/20261009-142151-v72-callivae-kl1e6-s12500-frozen-sp-c2ot/eval_samples_ctrl/step0130000/eval200fix": ("v72", 130000),
    "assets/results/v74_callivae_mode_sp_c2ot/20261010-025411-v74-callivae-mode-sp-c2ot/eval_samples_ctrl/step0115000/eval200fix": ("v74", 115000),
    "assets/results/v76_callivae_topband_sp_c2ot/20261010-150745-v76-callivae-topband-sp-c2ot/eval_samples_ctrl/step0040000/eval200fix": ("v76", 40000),
    "assets/results/v77_callivae_topband_norepa_sp_c2ot/20261010-183103-v77-callivae-topband-norepa-sp-c2ot/eval_samples_ctrl/step0040000/eval200fix": ("v77", 40000),
    "assets/results/v78_callivae_topband_scale1_sp_c2ot/20261010-222243-v78-callivae-topband-scale1-sp-c2ot/eval_samples_ctrl/step0045000/eval200fix": ("v78", 45000),
    "assets/results/v79_callivae_physics_sp_c2ot/20261011-133610-v79-callivae-physics-sp-c2ot/eval_samples_ctrl/step0075000/eval200fix": ("v79", 75000),
    "assets/results/v80_callivae_physics_noOT/20261011-195827-v80-callivae-physics-noOT/eval_samples_ctrl/step0190000/eval200fix": ("v80", 190000),
    "assets/results/v80_callivae_physics_noOT/20261011-195827-v80-callivae-physics-noOT/eval_samples_ctrl/step0200000/eval200fix": ("v80", 200000),
    "assets/results/v83A_55k_correct_eval_20261012/eval_samples_ctrl/eval200fix_input_g": ("X-input-g", 0),
    "assets/results/v83A_55k_correct_eval_20261012/eval_samples_ctrl/step0055000/eval200fix": ("v83A", 55000),
    "assets/results/v83A_stdsoft_zerofix/20261012-213952-v83A-stdsoft-zerofix/eval_samples_ctrl/step0105000/eval200fix": ("v83A-zf", 105000),
    "assets/results/v84A_std_cond/20261013-072215-v84A-std-cond/eval_samples_ctrl/step0200000/eval200fix": ("v84A", 200000),
    "assets/results/v85A_style_ada/20261014-032924-v85A-style-ada/eval_samples_ctrl/step0010000/eval200fix": ("v85A", 10000),
    "assets/results/v85A_style_ada/20261014-032924-v85A-style-ada/eval_samples_ctrl/step0015000/eval200fix": ("v85A", 15000),
    "assets/results/v85A_style_ada/20261014-032924-v85A-style-ada/eval_samples_ctrl/step0020000/eval200fix": ("v85A", 20000),
    "assets/results/v85A_style_ada/20261014-032924-v85A-style-ada/eval_samples_ctrl/step0030000/eval200fix": ("v85A", 30000),
    "assets/style_eval_src/cfg1.2": ("CFG1.2", 0),
    "assets/style_eval_src/cfg1.4": ("CFG1.4", 0),
    "assets/style_eval_src/cfg1.6": ("CFG1.6", 0),
    "assets/style_eval_src/moyi_top10_rf": ("MoYi-12ch", 50000),
    "assets/style_eval_src/moyi_top10_rf_4ch": ("MoYi-4ch", 80000),
    "assets/style_eval_src/step0020000": ("style-scan", 20000),
    "assets/style_eval_src/step0050000": ("style-scan", 50000),
    "assets/style_eval_src/step0080000": ("style-scan", 80000),
    "assets/style_eval_src/step0100000": ("style-scan", 100000),
    "assets/style_eval_src/step0115000": ("style-scan", 115000),
    "assets/style_eval_src/step0120000": ("style-scan", 120000),
    "assets/style_eval_src/step0200000": ("style-scan", 200000),
    "assets/style_eval_src/v85A_step0015000": ("v85A", 15000),
    "assets/style_eval_src/v85A_step0020000": ("v85A", 20000),
    "assets/style_eval_src/v85A_step0030000": ("v85A", 30000),
    "assets/style_eval_src/v85C_step0030000": ("v85C", 30000),
    "assets/style_eval_src/v85E_step0015000": ("v85E", 15000),
    "assets/style_eval_src/v85E_step0030000": ("v85E", 30000),
    "assets/style_eval_src/v88_step0005000": ("v88", 5000),
    "assets/style_eval_src/v88_step0050000": ("v88", 50000),
    "assets/v85_verified_control_10k": ("v85-ctrl", 10000),
    "assets/v85_verified_v84A_200k": ("v84A", 200000),
}

rows = list(csv.DictReader(BASE.open(encoding="utf-8-sig")))
hist = json.loads(HIST.read_text(encoding="utf-8"))
seen = {(r["arm"], str(r["step"])) for r in rows}
added = 0
for path, item in hist.items():
    if "error" in item:
        continue
    pooled = item.get("E-G", {}).get("pooled", {})
    ocr = item.get("ocr", {})
    if not pooled or ocr.get("top1") is None:
        continue
    key = NAME.get(path.replace("\\", "/"))
    if not key:
        continue
    arm, step = key
    if (arm, str(step)) in seen:
        continue
    seen.add((arm, str(step)))
    rows.append({"arm": arm, "step": step, "protocol": "eval200_fixed-187", "n": 187,
                 "eg_auc": round(100 * pooled["auc"], 4),
                 "eg_eer": round(100 * pooled["eer"], 4),
                 "ocr_top1": ocr["top1"], "ocr_n": ocr.get("n"),
                 "source_panel": path.replace("\\", "/"),
                 "source_csv": "exp-std/csv/eval200_fixed.csv"})
    added += 1

with OUT.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(sorted(rows, key=lambda r: (int(r["n"]), -float(r["eg_auc"]))))
print("rows", len(rows), "added", added)
