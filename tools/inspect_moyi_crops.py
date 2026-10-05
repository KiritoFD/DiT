from PIL import Image
import numpy as np
import os

im = Image.open("exp/moyi_top10_rf/posters/eval_step_0022000.png")
w, h = im.size
print(f"Overall poster size: {w}x{h}")

# The grid was saved with nrow=5, so 2 rows, 5 columns
# padding was 4
# Let's crop the first 4 cells
os.makedirs("exp/moyi_top10_rf/crops", exist_ok=True)

# Grid dimensions:
# Each cell is 256x256
# With padding=4:
# x0 = 4, x1 = 4 + 256 = 260
# cell 0: (4, 4, 260, 260)
# cell 1: (264, 4, 520, 260)
prompts = [
    "01_wangxizhi_kai_yong",
    "02_wangxizhi_xing_he",
    "03_wangxizhi_xing_qing",
    "04_yanzhenqing_kai_qing",
    "05_yanzhenqing_kai_du",
    "06_mifu_xing_tian",
    "07_mifu_xing_feng",
    "08_zhaomengfu_xing_yun",
    "09_zhaomengfu_li_min",
    "10_liugongquan_kai_xin"
]

cell_size = 256
pad = 4

for idx, name in enumerate(prompts):
    row = idx // 5
    col = idx % 5
    x0 = pad + col * (cell_size + pad)
    y0 = pad + row * (cell_size + pad)
    x1 = x0 + cell_size
    y1 = y0 + cell_size
    cropped = im.crop((x0, y0, x1, y1))
    crop_path = f"exp/moyi_top10_rf/crops/{name}.png"
    cropped.save(crop_path)
    arr = np.array(cropped)
    print(f"{name}: mean={arr.mean():.1f}, min={arr.min()}, max={arr.max()}, std={arr.std():.1f}")
