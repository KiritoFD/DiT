import glob
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

targets = {
    "Base 55k": ROOT / "assets/results/unet_ablation/u100_base_55k",
    "DiC 50k": ROOT / "assets/results/unet_connectivity/DiC_combined",
    "Arm2 50k": ROOT / "assets/results/unet_ablation/arm2_writer_film_dec_b768",
    "ExpA 50k": ROOT / "assets/results/unet_decoupled_114m/armA_writer_decoupled_50k",
    "ExpA 70k": ROOT / "assets/results/unet_decoupled_114m/armA_writer_decoupled_70k",
}

for name, path in targets.items():
    if not path.exists():
        print(f"{name:12s}: Directory does NOT exist: {path}")
        continue
    # find where g0.png or g*.png is located
    matches = list(path.glob("**/g*.png"))
    dirs_with_g = sorted(list(set(p.parent for p in matches)))
    print(f"{name:12s}: Total {len(matches)} images in {len(dirs_with_g)} dirs:")
    for d in dirs_with_g:
        count = len(list(d.glob("g*.png")))
        print(f"    {d.relative_to(ROOT)} -> {count} pngs")
