from PIL import Image
import numpy as np

im = Image.open("exp/purestd_posters/eval200fix_poster.png")
w, h = im.size
print(f"eval200fix_poster.png: {w}x{h}")

# In eval200fix_poster.png:
# How many rows are there?
# Let's inspect vertical projection
arr = np.array(im.convert("L"))
# Look at average brightness across each horizontal row
row_means = arr.mean(axis=1)

print("Row brightness min/max/mean:", row_means.min(), row_means.max(), row_means.mean())
# Find where the character rows are located vertically
# Let's sample a vertical slice at x=500
slice_vert = arr[:, 500]
print("Vertical slice at x=500, every 100 pixels:")
for y in range(0, h, 100):
    print(f"y={y:4d}: val={arr[y, 500]:.1f}, row_mean={row_means[y]:.1f}")
