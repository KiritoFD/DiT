import csv
from pathlib import Path
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim

r5 = list(csv.DictReader(open("exp-std/csv/eval_top5_fixed.csv", encoding="utf-8-sig")))
r10 = list(csv.DictReader(open("exp-std/csv/eval200_fixed.csv", encoding="utf-8-sig")))

img_to_r10 = {r["image_path"]: (i, r) for i, r in enumerate(r10)}

failures = []

for i5, r in enumerate(r5):
    p_img = r["image_path"]
    if p_img not in img_to_r10:
        continue
    i10 = img_to_r10[p_img][0]
    
    p_t5 = Path(f"assets/results/top5_u100_base_50k/g{i5}.png")
    p_t10 = Path(f"artifacts/DiC_U100_review_20261020/images/originals/decA_70k/g{i10}.png")
    p_gt = Path(f"artifacts/DiC_U100_review_20261020/images/reference/gt/g{i10}.png")
    
    if not (p_t5.exists() and p_t10.exists() and p_gt.exists()):
        continue
        
    im_t5 = np.array(Image.open(p_t5).convert("L"))
    im_t10 = np.array(Image.open(p_t10).convert("L"))
    im_gt = np.array(Image.open(p_gt).convert("L"))
    
    s_t5 = float(ssim(im_t5, im_gt))
    s_t10 = float(ssim(im_t10, im_gt))
    
    failures.append({
        "i5": i5, "i10": i10,
        "callig": r["calligrapher"],
        "script": r["script"],
        "char": r["character"],
        "s_t5": s_t5,
        "s_t10": s_t10,
        "diff": s_t10 - s_t5,
        "path": p_img
    })

failures.sort(key=lambda x: x["diff"], reverse=True)

with open("scratch/top5_vs_top10_analysis.txt", "w", encoding="utf-8") as f:
    f.write("Rank | i5  | i10 | Calligrapher Script [Char] | SSIM Top5 | SSIM Top10 | Diff\n")
    f.write("-" * 75 + "\n")
    for rank, item in enumerate(failures[:25]):
        line = f"#{rank+1:02d}  | g{item['i5']:02d} | g{item['i10']:03d} | {item['callig']} {item['script']} 【{item['char']}】 | {item['s_t5']:.4f}    | {item['s_t10']:.4f}     | {item['diff']:+6.4f}\n"
        f.write(line)

print("Saved top failures to scratch/top5_vs_top10_analysis.txt")
