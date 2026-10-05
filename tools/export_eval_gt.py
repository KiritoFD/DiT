import torch
from torchvision.utils import save_image
import os

cache_path = "/home/ds/Workspace/moyi/data/top10_style23/eval_real200_cache.pt"
print(f"Loading {cache_path}...")
cache = torch.load(cache_path, map_location="cpu")

out_dir = "/home/ds/Workspace/moyi/results/moyi_top10_rf/extracted_eval_gt"
os.makedirs(out_dir, exist_ok=True)

indices = [10, 11, 21, 22, 13, 19, 20, 6]
std_pngs = cache["std_pngs"]
gt_pngs = cache["gt_pngs"]
rows = cache["rows"]

for idx in indices:
    r = rows[idx]
    c = r.get("calligrapher", "")
    sc = r.get("script", "")
    ch = r.get("character", "")
    
    # Save standard font
    std_img = std_pngs[idx] # (3, 256, 256)
    std_path = os.path.join(out_dir, f"std_g{idx}_{c}_{sc}_{ch}.png")
    save_image(std_img, std_path)
    
    # Save ground truth
    gt_img = gt_pngs[idx] # (3, 256, 256)
    gt_path = os.path.join(out_dir, f"gt_g{idx}_{c}_{sc}_{ch}.png")
    save_image(gt_img, gt_path)
    print(f"✓ Saved idx={idx} ({c} {sc} {ch}): std and gt")

print("All GT and Std images extracted!")
