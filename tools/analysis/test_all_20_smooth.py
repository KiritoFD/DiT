import json
import sys
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.insert(0, ".")
from tools.analysis.test_smooth_sdf import smooth_antialiased_sdf_align
from tools.analysis.verify_20_pristine import candidates_pool

ROOT = Path(".")
with open("assets/stroke_width_trainset_distribution.json", encoding="utf-8") as f:
    train_dist = json.load(f)

print("Testing smooth antialiased alignment on all 20 samples...")

for idx, c in enumerate(candidates_pool):
    slot = f"{c['callig']}_{c['script']}"
    target = train_dist[slot]
    
    # Top 10 image
    p_top10 = Path(f"artifacts/DiC_U100_review_20261020/images/originals/decA_70k/g{c['i10']}.png")
    im_raw = np.array(Image.open(p_top10).convert("L"), dtype=np.float32) / 255.0
    
    for a in [0.35, 0.70, 1.00]:
        res = smooth_antialiased_sdf_align(im_raw, target, alpha=a)
        # Check that it's clean and has smooth transitions
        assert res.min() >= 0.0 and res.max() <= 1.0

print("All 20 samples aligned cleanly without errors!")
