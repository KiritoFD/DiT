from PIL import Image
import numpy as np

img = Image.open("exp/v34_stage2_mix25/render_base_eval_real200_poster.png")
w, h = img.size
print(f"Base Render Poster dimensions: width={w}, height={h}")

# 3 rows:
# Row 0: Skeleton condition (0~256)
# Row 1: v34 30k Render output (256~512)
# Row 2: GT image (512~768)
for row_idx in range(3):
    crop = img.crop((0, row_idx * 256, 256, (row_idx + 1) * 256))
    crop.save(f"v34_sample0_row{row_idx}.png")
    arr = np.array(crop).astype(float)
    black_ratio = np.mean(arr < 128)
    print(f"v34 Row {row_idx}: mean={arr.mean():.2f}, black_pixel_ratio={black_ratio*100:.2f}%")
