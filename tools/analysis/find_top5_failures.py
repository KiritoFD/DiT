import csv
from pathlib import Path
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Top 5 eval csv: exp-std/csv/eval_top5_fixed.csv (93 samples)
# Top 10 eval csv: exp-std/csv/eval200_fixed.csv (187 samples)
# Top 5 results: assets/results/top5_u100_base_50k/g{i5}.png
# Top 10 results: artifacts/DiC_U100_review_20261020/images/originals/decA_70k/g{i10}.png

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
    
    ink_t5 = int((im_t5 < 128).sum())
    ink_t10 = int((im_t10 < 128).sum())
    ink_gt = int((im_gt < 128).sum())
    
    # Calculate SSIM with GT
    s_t5 = float(ssim(im_t5, im_gt))
    s_t10 = float(ssim(im_t10, im_gt))
    
    diff_ssim = s_t10 - s_t5
    
    # Check if Top5 is collapsed/broken (e.g. ink is very low or SSIM is very low while Top10 is good)
    failures.append({
        "i5": i5, "i10": i10,
        "callig": r["calligrapher"],
        "script": r["script"],
        "char": r["character"],
        "ink_t5": ink_t5,
        "ink_t10": ink_t10,
        "ink_gt": ink_gt,
        "ssim_t5": s_t5,
        "ssim_t10": s_t10,
        "diff_ssim": diff_ssim,
        "path": p_img
    })

# Sort by diff_ssim (where Top 10 beat Top 5 by the most)
failures.sort(key=lambda x: x["diff_ssim"], reverse=True)

print(f"{'Rank':4s} | {'Sample':20s} | {'Char':4s} | {'SSIM Top5':9s} | {'SSIM Top10':10s} | {'Diff':6s} | {'Ink T5':7s} | {'Ink T10':7s} | {'Ink GT':7s}")
print("-" * 85)

for rank, f in enumerate(failures[:20]):
    info = f"i5={f['i5']:02d} i10={f['i10']:03d} {f['callig']} {f['script']}"
    print(f"#{rank+1:02d}  | {info:20s} | {f['char']:2s}   | {f['ssim_t5']:.4f}    | {f['ssim_t10']:.4f}     | {f['diff_ssim']:+6.4f} | {f['ink_t5']:6d}  | {f['ink_t10']:6d}  | {f['ink_gt']:6d}")
