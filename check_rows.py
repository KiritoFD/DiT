import os, sys
import numpy as np
from PIL import Image

for r in range(4):
    p = f"poster_sample0_row{r}.png"
    im = Image.open(p)
    arr = np.array(im)
    print(f"Row {r} ({p}): size={im.size}, mode={im.mode}, min={arr.min()}, max={arr.max()}, mean={arr.mean():.2f}")
