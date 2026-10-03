import sys, os
ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch as th
from PIL import Image

cache = th.load("data/top10_style23/eval_real200_cache.pt", map_location="cpu")
print("Cache keys:", cache.keys())
print("std_pngs shape:", cache["std_pngs"].shape, "mean:", cache["std_pngs"].mean().item())
print("gt_pngs shape:", cache["gt_pngs"].shape, "mean:", cache["gt_pngs"].mean().item())

# Save first sample of each to inspect
Image.fromarray((cache["std_pngs"][0].permute(1, 2, 0).numpy() * 255).astype("uint8")).save("test_cache_std.png")
Image.fromarray((cache["gt_pngs"][0].permute(1, 2, 0).numpy() * 255).astype("uint8")).save("test_cache_gt.png")

if "rows" in cache:
    r0 = cache["rows"][0]
    print("row 0 image_path:", r0["image_path"])
    img = Image.open(r0["image_path"])
    print("real image size/mode:", img.size, img.mode)
    img.save("test_real_img0.png")
