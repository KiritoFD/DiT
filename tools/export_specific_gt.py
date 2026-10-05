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

# Find index for img_id 20672 (颜真卿 楷 其)
for idx, r in enumerate(rows):
    iid = int(r.get("img_id", -1))
    if iid == 20672:
        print(f"Found img_id 20672 at cache index {idx} ({r.get('calligrapher')} {r.get('script')} {r.get('character')})")
        save_image(gt_pngs[idx], os.path.join(out_dir, "gt_20672_yanzhenqing_kai_qi.png"))
        save_image(std_pngs[idx], os.path.join(out_dir, "std_20672_yanzhenqing_kai_qi.png"))

print("Export completed!")
