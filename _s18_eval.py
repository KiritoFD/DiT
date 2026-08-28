import json, glob, sys
sys.stdout.reconfigure(encoding="utf-8")
fs = sorted(glob.glob("5script/results/s18_s_flow_small/*/checkpoints/eval_auto_*.json"))
for f in fs:
    d = json.load(open(f))
    print("step", d.get("step"))
    for k in ["mse", "mse_std", "mse_q25", "mse_q50", "mse_q75", "ssim", "ssim_std", "ssim_min",
              "ssim_q25", "ssim_q50", "ssim_q75", "skel_iou", "skel_iou_std", "skel_iou_min",
              "skel_iou_q25", "skel_iou_q50", "skel_iou_q75", "lpips", "n"]:
        if k in d:
            v = d[k]
            print(f"   {k}: {round(v,4) if isinstance(v,float) else v}")
