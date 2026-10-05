from PIL import Image
import numpy as np

img = Image.open("exp/v39_render_mix50/posters/render_eval_step_0007000.png")
w, h = img.size
print(f"Render poster dimensions: width={w}, height={h}")

# Render poster has 3 rows:
# Row 0: Skeleton condition
# Row 1: Rendered image
# Row 2: GT image
for row_idx in range(3):
    crop = img.crop((0, row_idx * 256, 256, (row_idx + 1) * 256))
    crop.save(f"render_sample0_row{row_idx}.png")
    arr = np.array(crop).astype(float)
    black_ratio = np.mean(arr < 128)
    print(f"Render Poster Row {row_idx}: mean={arr.mean():.2f}, black_pixel_ratio={black_ratio*100:.2f}%")
