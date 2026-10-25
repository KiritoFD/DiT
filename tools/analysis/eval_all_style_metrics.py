import json, csv, os, glob
from pathlib import Path
from PIL import Image
import numpy as np
from src.eval.nearest_neighbor import samechar_nn_eval

train_csv = "exp-std/csv/train_top10_aug_sym.csv"
eval_csv = "exp-std/csv/eval200_fixed.csv"

with open(eval_csv, encoding="utf-8") as f:
    ev_rows = list(csv.DictReader(f))[:187]

eval_dirs = {
    "Baseline_55k": "/home/ds/Workspace/DiT/assets/results/unet_connectivity_100000/baseline_resume/20261018-030921-Unet-flow/eval_samples_ctrl/step0055000/eval200fix",
    "Arm1_residual_gate_65k": "/home/ds/Workspace/DiT/assets/results/unet_dic/B_residual_gate/20261018-064328-Unet-flow/eval_samples_ctrl/step0065000/eval200fix",
    "Arm1_residual_gate_75k": "/home/ds/Workspace/DiT/assets/results/unet_dic/B_residual_gate/20261018-064328-Unet-flow/eval_samples_ctrl/step0075000/eval200fix",
    "Arm2_stage_adapter_65k": "/home/ds/Workspace/DiT/assets/results/unet_dic/C_stage_cond_adapter/20261018-092206-Unet-flow/eval_samples_ctrl/step0065000/eval200fix",
    "Arm2_stage_adapter_75k": "/home/ds/Workspace/DiT/assets/results/unet_dic/C_stage_cond_adapter/20261018-092206-Unet-flow/eval_samples_ctrl/step0075000/eval200fix",
    "Arm3_stage_indep_65k": "/home/ds/Workspace/DiT/assets/results/unet_dic/C_stage_cond_independent/20261018-115725-Unet-flow/eval_samples_ctrl/step0065000/eval200fix",
    "Arm3_stage_indep_75k": "/home/ds/Workspace/DiT/assets/results/unet_dic/C_stage_cond_independent/20261018-115725-Unet-flow/eval_samples_ctrl/step0075000/eval200fix",
    "4090_DiC_U100_50k": "/home/ds/Workspace/DiT/assets/results/eval_4090_b768_50k/eval_samples_ctrl/step0050000/eval200fix",
}

print("=" * 95)
print(f"{'Model Experiment':26s} | {'nn_ssim':7s} | {'nn_mean':7s} | {'tgt_spec':8s} | {'cal_hit':7s} | {'cal_enrich':10s}")
print("=" * 95)

for name, d in eval_dirs.items():
    p = Path(d)
    if not p.is_dir():
        print(f"{name:26s} | NOT FOUND: {d}")
        continue
    preds = []
    gts = []
    for i in range(187):
        pred_p = p / f"g{i}.png"
        gt_p = p / f"gt{i}.png"
        if not (pred_p.is_file() and gt_p.is_file()):
            break
        preds.append(np.asarray(Image.open(pred_p).convert("RGB"), dtype=np.float32) / 255.0)
        gts.append(np.asarray(Image.open(gt_p).convert("RGB"), dtype=np.float32) / 255.0)
    if len(preds) < 187:
        print(f"{name:26s} | incomplete images ({len(preds)}/187)")
        continue
    res = samechar_nn_eval(np.stack(preds), np.stack(gts), ev_rows, train_csv, max_cand=12, nw=16)
    if res:
        nn_s = f"{res['nn_ssim']:.4f}"
        nn_m = f"{res['nn_mean']:.4f}"
        ts = f"{res['tgt_spec']:+.4f}"
        ch = f"{res['cal_hit']*100:.1f}%"
        ce = f"{res['cal_enrich']:.2f}x"
        print(f"{name:26s} | {nn_s:7s} | {nn_m:7s} | {ts:8s} | {ch:7s} | {ce:10s}")
    else:
        print(f"{name:26s} | eval returned None")
print("=" * 95)
