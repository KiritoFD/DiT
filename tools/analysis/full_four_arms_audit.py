import json, csv, os
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict
from src.eval.nearest_neighbor import samechar_nn_eval, _ssim

train_csv = "exp-std/csv/train_top10_aug_sym.csv"
eval_csv = "exp-std/csv/eval200_fixed.csv"

with open(eval_csv, encoding="utf-8") as f:
    ev_rows = list(csv.DictReader(f))[:187]

eval_dirs = {
    "Baseline (55k)": "/home/ds/Workspace/DiT/assets/results/unet_connectivity_100000/baseline_resume/20261018-030921-Unet-flow/eval_samples_ctrl/step0055000/eval200fix",
    "Arm1: DiC_residual_gate (75k)": "/home/ds/Workspace/DiT/assets/results/unet_dic/B_residual_gate/20261018-064328-Unet-flow/eval_samples_ctrl/step0075000/eval200fix",
    "Arm2: DiC_stage_adapter (75k)": "/home/ds/Workspace/DiT/assets/results/unet_dic/C_stage_cond_adapter/20261018-092206-Unet-flow/eval_samples_ctrl/step0075000/eval200fix",
    "Arm3: DiC_stage_indep (75k)": "/home/ds/Workspace/DiT/assets/results/unet_dic/C_stage_cond_independent/20261018-115725-Unet-flow/eval_samples_ctrl/step0075000/eval200fix",
    "Node4090: DiC_U100_b768 (50k)": "/home/ds/Workspace/DiT/assets/results/eval_4090_b768_50k/eval_samples_ctrl/step0050000/eval200fix",
}

# Pre-load candidate images by character from train_csv
by_char = defaultdict(list)
with open(train_csv, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        by_char[str(r.get("character", ""))].append(r)

results = {}
for name, d in eval_dirs.items():
    p = Path(d)
    preds = [np.asarray(Image.open(p / f"g{i}.png").convert("RGB"), dtype=np.float32) / 255.0 for i in range(187)]
    gts = [np.asarray(Image.open(p / f"gt{i}.png").convert("RGB"), dtype=np.float32) / 255.0 for i in range(187)]
    
    # Per-calligrapher stats
    cal_stats = defaultdict(lambda: {"own_ssim": [], "nn_ssim": [], "hit": [], "total_cand": []})
    for i in range(187):
        r = ev_rows[i]
        cal = r.get("calligrapher")
        ch = str(r.get("character", ""))
        own = float(_ssim(preds[i], gts[i]))
        cs = by_char.get(ch, [])
        if len(cs) > 12:
            st = len(cs) / 12
            cs = [cs[int(k * st)] for k in range(12)]
        cand_scores = []
        for c in cs:
            img_p = c.get("image_path", "")
            if not os.path.isabs(img_p):
                img_p = os.path.join(os.getcwd(), img_p)
            if not os.path.exists(img_p):
                continue
            with Image.open(img_p) as f_img:
                t = np.asarray(f_img.convert("RGB"), dtype=np.float32) / 255.0
            cand_scores.append((float(_ssim(preds[i], t)), c.get("calligrapher")))
        if not cand_scores:
            continue
        vals = [s for s, _ in cand_scores]
        best_cand = max(cand_scores, key=lambda x: x[0])
        hit = (best_cand[1] == cal)
        cal_stats[cal]["own_ssim"].append(own)
        cal_stats[cal]["nn_ssim"].append(best_cand[0])
        cal_stats[cal]["hit"].append(1 if hit else 0)
        cal_stats[cal]["total_cand"].append(len(cand_scores))

    # Overall style metrics
    overall_res = samechar_nn_eval(np.stack(preds), np.stack(gts), ev_rows, train_csv, max_cand=12, nw=16)
    results[name] = {"overall": overall_res, "per_cal": cal_stats}

# Output summary table
print("=== PER-CALLIGRAPHER STYLE HIT ACCURACY (%) ===")
calligraphers = ['赵孟頫', '何绍基', '文徵明', '欧阳询', '褚遂良', '王羲之', '苏轼', '柳公权', '颜真卿', '米芾']
header = f"{'Calligrapher':8s} | " + " | ".join(f"{name[:14]:14s}" for name in eval_dirs.keys())
print(header)
print("-" * len(header))
for cal in calligraphers:
    row = [f"{cal:8s}"]
    for name in eval_dirs.keys():
        stats = results[name]["per_cal"][cal]
        hit_rate = np.mean(stats["hit"]) * 100 if stats["hit"] else 0.0
        row.append(f"{hit_rate:13.1f}%")
    print(" | ".join(row))

print("\n=== OVERALL STYLE SUMMARY METRICS ===")
print(f"{'Experiment':30s} | {'SSIM':7s} | {'LPIPS':7s} | {'Frag':6s} | {'Ink-IoU':7s} | {'cal_hit':7s} | {'cal_enrich':10s} | {'tgt_spec':8s}")
print("-" * 105)
for name, d in eval_dirs.items():
    ov = results[name]["overall"]
    # read basic metrics from DONE.json or eval_stdskel_summary.csv if needed
    p = Path(d)
    done_file = p / "DONE.json"
    ssim = 0
    lpips = 0
    frag = 0
    iou = 0
    if done_file.is_file():
        done = json.loads(done_file.read_text(encoding="utf-8"))
        # from summary
    summary_csv = p.parent.parent / "eval_stdskel_summary.csv"
    if summary_csv.is_file():
        for line in summary_csv.read_text(encoding="utf-8").strip().splitlines()[1:]:
            parts = line.split(",")
            step = parts[1]
            if (step in str(p)):
                ssim = float(parts[9])
                lpips = float(parts[16])
                iou = float(parts[18])
                frag = float(parts[20])
                break
    print(f"{name:30s} | {ssim:7.4f} | {lpips:7.4f} | {frag:6.3f} | {iou:7.4f} | {ov['cal_hit']*100:6.1f}% | {ov['cal_enrich']:9.2f}x | {ov['tgt_spec']:+8.4f}")
