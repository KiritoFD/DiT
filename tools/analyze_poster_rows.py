from PIL import Image
import numpy as np

im = Image.open("exp/purestd_posters/eval200fix_poster.png")
w, h = im.size
print(f"eval200fix_poster.png: {w}x{h}")

# In eval200fix_poster.png:
# Let's see: each cell is likely 256x256 or 270x270
# Let's find vertical row cuts:
arr = np.array(im.convert("L"))
# Invert if white background
# Let's check background color:
corner_color = arr[0, 0]
print(f"Corner pixel value: {corner_color}")

# Row cuts:
# Horizontal projection:
row_proj = arr.mean(axis=1)
# Find boundaries where row_proj changes significantly
for y in range(0, h, 20):
    print(f"y={y:4d}: mean={row_proj[y]:.1f}")
