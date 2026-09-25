import glob
import os
import csv
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

batch_files = glob.glob(os.path.join(ROOT, "assets/results/**/eval_stdskel_batch.csv"), recursive=True)

header = "exp,step,set,n,ssim_mean,ssim_p10,ssim_q1,ssim_med,ssim_q3,ssim_p90,mse_mean,lpips_mean,ink_ssim_mean,ink_iou_mean,skel_iou_mean,frag_ratio,hole_pred,hole_gt,nn_ssim,nn_mean,tgt_spec,cal_enrich\n"

for bf in batch_files:
    sf = os.path.join(os.path.dirname(bf), "eval_stdskel_summary.csv")
    if os.path.exists(sf):
        continue
    exp_name = os.path.basename(os.path.dirname(bf))
    print(f"Aggregating batch CSV for: {exp_name}")
    
    with open(bf, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        rows_by_step_set = defaultdict(list)
        for r in reader:
            step = r.get("step")
            sname = r.get("set")
            if step and sname:
                try:
                    rows_by_step_set[(int(step), sname)].append(r)
                except:
                    pass
                    
    out_rows = []
    for (step, sname), rlist in sorted(rows_by_step_set.items()):
        n = len(rlist)
        ssims = [float(r["ssim"]) for r in rlist if r.get("ssim")]
        mses = [float(r["mse"]) for r in rlist if r.get("mse")]
        lpips = [float(r["lpips"]) for r in rlist if r.get("lpips")]
        frags = [float(r["frag_ratio"]) for r in rlist if r.get("frag_ratio")]
        
        ssim_mean = sum(ssims)/len(ssims) if ssims else 0.0
        ssim_med = sorted(ssims)[len(ssims)//2] if ssims else 0.0
        mse_mean = sum(mses)/len(mses) if mses else 0.0
        lpips_mean = sum(lpips)/len(lpips) if lpips else 0.0
        frag_mean = sum(frags)/len(frags) if frags else 0.0
        
        line = f"{exp_name},{step},{sname},{n},{ssim_mean:.4f},0,0,{ssim_med:.4f},0,0,{mse_mean:.5f},{lpips_mean:.5f},0,0,0,{frag_mean:.4f},0,0,0,0,0,0"
        out_rows.append(line)
        
    with open(sf, "w", encoding="utf-8") as out:
        out.write(header)
        for line in out_rows:
            out.write(line + "\n")
    print(f"  Wrote {len(out_rows)} summary rows to {sf}")
