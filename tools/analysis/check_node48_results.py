import glob, os
from pathlib import Path

base_dir = Path("/home/ds/Workspace/DiT/assets/results/unet_dic")
for run in sorted(base_dir.glob("**/*-Unet-flow")):
    print("\n=========================================================================================")
    print("Arm: " + run.parent.name + " | Run: " + run.name)
    print("=========================================================================================")
    eval_csv = run / "eval_stdskel_summary.csv"
    if eval_csv.is_file():
        lines = eval_csv.read_text().strip().split("\n")
        header = lines[0].split(",")
        print("Step   | SSIM Mean | SSIM Med  | LPIPS   | Ink-IoU | Frag   | Hole   | SoftInk")
        print("-" * 75)
        for line in lines[1:]:
            parts = line.split(",")
            row = dict(zip(header, parts))
            step = row.get("step", "?")
            ssim_mean = f"{float(row.get('ssim_mean', 0)):.4f}"
            ssim_med = f"{float(row.get('ssim_med', 0)):.4f}"
            lpips = f"{float(row.get('lpips_mean', 0)):.4f}"
            iou = f"{float(row.get('ink_iou_mean', 0)):.4f}"
            frag = f"{float(row.get('frag_ratio', 0)):.3f}"
            hole = f"{float(row.get('hole_pred', 0)):.3f}"
            soft_ink = f"{float(row.get('soft_ink_ratio', 0)):.3f}"
            print(f"{step:6s} | {ssim_mean:9s} | {ssim_med:9s} | {lpips:7s} | {iou:7s} | {frag:6s} | {hole:6s} | {soft_ink:7s}")
