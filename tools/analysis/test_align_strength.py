import json
import sys
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.insert(0, ".")
from tools.render_fixed_20row_poster import measure_p50_and_contrast, smooth_antialiased_sdf_align

with open("assets/stroke_width_trainset_distribution.json", encoding="utf-8") as f:
    td = json.load(f)

# Let's test on Top 10 g80 (王羲之楷书, target mean=8.8, p50=8.8)
# and Top 10 g111 (苏轼楷书, target mean=16.5, p50=15.2)
p80 = Path("artifacts/DiC_U100_review_20261020/images/originals/decA_70k/g80.png")
p111 = Path("artifacts/DiC_U100_review_20261020/images/originals/decA_70k/g111.png")

im80 = np.array(Image.open(p80).convert("L"), dtype=np.float32) / 255.0
im111 = np.array(Image.open(p111).convert("L"), dtype=np.float32) / 255.0

p50_80_orig, _ = measure_p50_and_contrast(im80)
p50_111_orig, _ = measure_p50_and_contrast(im111)

print(f"Original g80 (Wang): p50 = {p50_80_orig:.2f} px (Target: {td['王羲之_楷']['ref_p50']:.1f})")
print(f"Original g111 (Su):   p50 = {p50_111_orig:.2f} px (Target: {td['苏轼_楷']['ref_p50']:.1f})")

# Current smooth_antialiased_sdf_align
res80 = smooth_antialiased_sdf_align(im80, td['王羲之_楷'], alpha=1.0)
res111 = smooth_antialiased_sdf_align(im111, td['苏轼_楷'], alpha=1.0)

p50_80_cur, _ = measure_p50_and_contrast(res80)
p50_111_cur, _ = measure_p50_and_contrast(res111)

print(f"Current align g80: p50 = {p50_80_cur:.2f} px (Changed: {p50_80_cur - p50_80_orig:+.2f} px)")
print(f"Current align g111: p50 = {p50_111_cur:.2f} px (Changed: {p50_111_cur - p50_111_orig:+.2f} px)")
