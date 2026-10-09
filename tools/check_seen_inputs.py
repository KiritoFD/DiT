import os
import glob
import pandas as pd
from PIL import Image

os.chdir("/root/Workspace/xy/DiT")

# Check if seen_input_g has images
p_in = "assets/results/std_callig_aug/eval_samples_ctrl/seen_input_g"
print("Input images exist in seen_input_g:", os.path.exists(p_in))
if os.path.exists(p_in):
    print("Files in seen_input_g:", sorted(os.listdir(p_in))[:12])

# Check GT images in v13 or v68
p_gt = "assets/results/v13_base_50k/20260917-211905-v13-base-50k/eval_samples_ctrl/step0155000/g"
print("GT images exist in v13:", os.path.exists(p_gt))
if os.path.exists(p_gt):
    print("Files in v13 g:", sorted([f for f in os.listdir(p_gt) if "gt" in f])[:12])
