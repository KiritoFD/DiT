import sys
sys.path.insert(0, ".")
from PIL import Image
import numpy as np
from tools.render_fixed_20row_poster import SAMPLE_MAPPING

for r_idx, s in enumerate(SAMPLE_MAPPING):
    i2 = s["i2"]
    ch = s["char"]
    p30 = f"assets/results/top2_u100_base_30k/g{i2}.png"
    p50 = f"assets/results/top2_u100_base_50k/g{i2}.png"
    im30 = np.array(Image.open(p30).convert("L"))
    im50 = np.array(Image.open(p50).convert("L"))
    ink30 = int((im30 < 128).sum())
    ink50 = int((im50 < 128).sum())
    min30 = int(im30.min())
    min50 = int(im50.min())
    print(f"Row {r_idx:02d} [{ch}] i2={i2:02d}: 30k ink={ink30:5d}, min={min30:3d} | 50k ink={ink50:5d}, min={min50:3d}")
