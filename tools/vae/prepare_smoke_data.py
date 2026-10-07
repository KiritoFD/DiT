import os
import csv
import glob
import shutil

pngs = glob.glob("/home/ds/Workspace/moyi/results/moyi_top10_rf/eval_samples_step20000/*.png")
print("Found %d test pngs" % len(pngs))
out_dir = "/tmp/vae_smoke"
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(out_dir, "smoke.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["image_path"])
    for p in pngs:
        w.writerow([os.path.basename(p)])

for p in pngs:
    shutil.copy(p, os.path.join(out_dir, os.path.basename(p)))

print("Smoke dataset prepared successfully in %s" % out_dir)
