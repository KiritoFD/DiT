import glob
import os

header = "exp,step,set,n,ssim_mean,ssim_p10,ssim_q1,ssim_med,ssim_q3,ssim_p90,mse_mean,lpips_mean,ink_ssim_mean,ink_iou_mean,skel_iou_mean,frag_ratio,hole_pred,hole_gt,nn_ssim,nn_mean,tgt_spec,cal_enrich\n"

# Search in assets/results/v21_skelnet_200k
target_dir = "assets/results/v21_skelnet_200k"
pattern = os.path.join(target_dir, "eval_stdskel_summary.csv*")

seen_steps = set()
rows = []
for f in sorted(glob.glob(pattern)):
    with open(f, "r", encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            if line.startswith("v21_skelnet_200k"):
                p = line.strip().split(",")
                try:
                    step = int(p[1])
                    sname = p[2]
                    key = (step, sname)
                    if key not in seen_steps:
                        seen_steps.add(key)
                        rows.append((step, sname, line.strip()))
                except:
                    pass

rows.sort()
out_csv = os.path.join(target_dir, "eval_stdskel_summary.csv")
with open(out_csv, "w", encoding="utf-8") as out:
    out.write(header)
    for r in rows:
        out.write(r[2] + "\n")

print(f"Consolidated {len(rows)} rows to {out_csv}")
