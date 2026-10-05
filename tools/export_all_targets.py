import torch
from torchvision.utils import save_image
import os

cache_path = "/home/ds/Workspace/moyi/data/top10_style23/eval_real200_cache.pt"
print(f"Loading {cache_path}...")
cache = torch.load(cache_path, map_location="cpu")

out_dir = "/home/ds/Workspace/moyi/results/moyi_top10_rf/extracted_eval_gt"
os.makedirs(out_dir, exist_ok=True)

rows = cache["rows"]
gt_pngs = cache["gt_pngs"]
std_pngs = cache["std_pngs"]

target_ids = {
    17513: ("g10", "王羲之", "楷", "旨"),
    17687: ("g11", "王羲之", "行", "好"),
    20672: ("g21", "颜真卿", "楷", "其"),
    4219:  ("g22", "颜真卿", "行", "憫"),
    18038: ("g13", "米芾",   "行", "墟"),
    14290: ("g19", "赵孟頫", "行", "匠"),
    37234: ("g20", "赵孟頫", "隶", "盤"),
    10042: ("g6",  "柳公权", "楷", "連"),
}

for idx, r in enumerate(rows):
    iid = int(r.get("img_id", -1))
    if iid in target_ids:
        tag, callig, script, ch = target_ids[iid]
        print(f"Found {tag}: img_id={iid} at cache index {idx} ({callig} {script} {ch})")
        save_image(gt_pngs[idx], os.path.join(out_dir, f"gt_{tag}.png"))
        save_image(std_pngs[idx], os.path.join(out_dir, f"std_{tag}.png"))

print("All GT and Std images exported!")
