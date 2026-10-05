from PIL import Image
import numpy as np

img = Image.open("exp/v40_joint_render_frozen/posters/joint_eval_step_0002000.png")
w, h = img.size
print(f"Poster dimensions: width={w}, height={h}")

# The poster has 4 rows (each height is 256):
# Row 0: 0 ~ 256
# Row 1: 256 ~ 512
# Row 2: 512 ~ 768
# Row 3: 768 ~ 1024
# Let's crop sample 0 (x: 0~256) for each row and measure mean pixel intensity and inspect
for row_idx in range(4):
    crop = img.crop((0, row_idx * 256, 256, (row_idx + 1) * 256))
    crop.save(f"poster_sample0_row{row_idx}.png")
    arr = np.array(crop).astype(float)
    # black pixels count (< 128)
    black_ratio = np.mean(arr < 128)
    print(f"Row {row_idx}: mean={arr.mean():.2f}, black_pixel_ratio={black_ratio*100:.2f}%")
