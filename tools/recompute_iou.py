# -*- coding: utf-8 -*-
"""recompute_iou.py — 从已落盘 samples 重算 IoU (dilate-3 skel IoU + mask IoU)."""
import csv
import multiprocessing as mp
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, generate_binary_structure

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ST = generate_binary_structure(2, 2)
RUNS = [
    ("S2_ref12ch", 90000), ("S2_v8_aux02", 120000), ("M432_sk3", 152500),
    ("M432_adaln4_sym", 42500), ("M432_sym_noise400k", 270000), ("Sp2_base", 17500),
]
DIR_PAT = {
    "S2_ref12ch": "v11_pretrain_S2_ref12ch",
    "S2_v8_aux02": "v11_pretrain_S2_v8_aux02",
    "M432_sk3": "v11_pretrain_M432_v8_sk3",
    "M432_adaln4_sym": "v11_pretrain_M432_adaln4_sym",
    "M432_sym_noise400k": "v11_pretrain_M432_adaln4_sym_noise400k",
    "Sp2_base": "v11_pretrain_Sp2_base",
}


def one(task):
    sub, step, i = task
    base = f"assets/results"
    # 找 run 目录 (按 RUNS 顺序在 main 里对齐; 这里传入完整路径)
    return task


def iou_one(args):
    g_path, gt_path = args
    try:
        g = np.asarray(Image.open(g_path).convert("L"))
        gt = np.asarray(Image.open(gt_path).convert("L"))
        b1, b2 = g < 128, gt < 128
        # mask IoU (笔画)
        miou = float((b1 & b2).sum()) / max(1.0, float((b1 | b2).sum()))
        # dilate-3 skel IoU
        try:
            from skimage.morphology import skeletonize
            s1 = skeletonize(b1)
            s2 = skeletonize(b2)
            s1 = binary_dilation(s1, ST, iterations=3)
            s2 = binary_dilation(s2, ST, iterations=3)
            siou = float((s1 & s2).sum()) / max(1.0, float((s1 | s2).sum()))
        except ImportError:
            siou = -1.0
        return miou, siou
    except Exception:
        return -1.0, -1.0


def main():
    out = []
    for run, step in RUNS:
        # 找 run 的 results_dir (eval_samples_ctrl/step{N})
        import glob
        dirs = glob.glob(f"assets/results/{DIR_PAT[run]}/eval_samples_ctrl/step{step:07d}")
        if not dirs:
            print(f"skip {run}@{step}: no samples")
            continue
        sd = dirs[0]
        for set_name, sub, n in (("seen", "g", 10), ("strict", "strict", 50)):
            g_dir = os.path.join(sd, sub)
            if not os.path.isdir(g_dir):
                # strict 的 sub 可能是 'strict'
                g_dir2 = os.path.join(sd, "g" if set_name == "seen" else "strict")
                if os.path.isdir(g_dir2):
                    g_dir = g_dir2
                else:
                    print(f"skip {run}@{step} {set_name}")
                    continue
            tasks = [(os.path.join(g_dir, f"g{i}.png"), os.path.join(g_dir, f"gt{i}.png"))
                     for i in range(n) if os.path.exists(os.path.join(g_dir, f"g{i}.png"))]
            if not tasks:
                print(f"skip {run}@{step} {set_name}: empty")
                continue
            with mp.Pool(16) as pool:
                res = pool.map(iou_one, tasks)
            mious = [m for m, _ in res if m >= 0]
            sious = [s for _, s in res if s >= 0]
            out.append({"run": run, "step": step, "set": set_name, "n": len(tasks),
                        "mask_iou_mean": round(float(np.mean(mious)), 4),
                        "mask_iou_med": round(float(np.median(mious)), 4),
                        "skel_iou_d3_mean": round(float(np.mean(sious)), 4),
                        "skel_iou_d3_med": round(float(np.median(sious)), 4)})
            print(f"{run}@{step} {set_name}: mask_iou={np.mean(mious):.4f} "
                  f"skel_iou_d3={np.mean(sious):.4f}", flush=True)
    with open("assets/eval_sweep_iou.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "step", "set", "n", "mask_iou_mean",
                                          "mask_iou_med", "skel_iou_d3_mean", "skel_iou_d3_med"])
        w.writeheader()
        w.writerows(out)
    print(f"written assets/eval_sweep_iou.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
