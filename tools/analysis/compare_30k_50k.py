from PIL import Image
import numpy as np
from pathlib import Path

p30 = Path("assets/results/top2_u100_base_30k")
p50 = Path("assets/results/top2_u100_base_50k")

pngs = sorted([f for f in p50.glob("g*.png") if f.stem[1:].isdigit()], key=lambda x: int(x.stem[1:]))

print("File   | 30k Ink  | 30k Min  | 50k Ink  | 50k Min  | Status")
print("-" * 65)

blank_30k = 0
blank_50k = 0

for f in pngs:
    arr50 = np.array(Image.open(f).convert("L"))
    ink50 = int((arr50 < 128).sum())
    min50 = int(arr50.min())
    
    f30 = p30 / f.name
    if f30.exists():
        arr30 = np.array(Image.open(f30).convert("L"))
        ink30 = int((arr30 < 128).sum())
        min30 = int(arr30.min())
    else:
        ink30 = -1
        min30 = -1
        
    is_blank_50 = (ink50 < 800 or min50 > 180)
    is_blank_30 = (ink30 < 800 or min30 > 180)
    
    if is_blank_50: blank_50k += 1
    if is_blank_30: blank_30k += 1
    
    if is_blank_50 or is_blank_30:
        if not is_blank_30 and is_blank_50:
            status = "30k OK -> 50k WHITE"
        elif is_blank_30 and is_blank_50:
            status = "BOTH WHITE"
        else:
            status = "30k WHITE -> 50k OK"
        print(f"{f.name:6s} | {ink30:8d} | {min30:8d} | {ink50:8d} | {min50:8d} | {status}")

print("-" * 65)
print(f"Total evaluated: {len(pngs)}")
print(f"Total Blank in 30k: {blank_30k}")
print(f"Total Blank in 50k: {blank_50k}")
