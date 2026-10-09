import os
import json
import pandas as pd
from PIL import Image

os.chdir("/root/Workspace/xy/DiT")

# Check config of v10b_cos_e
cfg_p = "assets/results/v10b_stdskel_fame3_c41x_cos_e/20260909-193018-v10b-stdskel-fame3-c41x-cos-e/resolved_config.json"
if os.path.exists(cfg_p):
    with open(cfg_p, encoding="utf-8") as f:
        cfg = json.load(f)
        print("v10b eval settings:", {k: v for k, v in cfg.items() if "eval" in k or "csv" in k})

# Check eval_v13_strict_fixed vs eval_fame3_strict_clean_v9
df_v13 = pd.read_csv("assets/eval_v13_strict_fixed.csv")
df_fame3 = pd.read_csv("assets/eval_fame3_strict_clean_v9.csv")
print("v13 strict first 5 chars:", df_v13[["character", "calligrapher", "script"]].head(5).to_dict(orient="records"))
print("fame3 strict first 5 chars:", df_fame3[["character", "calligrapher", "script"]].head(5).to_dict(orient="records"))

# Check if gt0 in v10b strict50 matches gt0 in df_fame3 or df_v13
# We can check by checking image path or computing pixel difference
p_gt0_v10b = "assets/results/v10b_stdskel_fame3_c41x_cos_e/20260909-193018-v10b-stdskel-fame3-c41x-cos-e/eval_samples_ctrl/step0390000/strict50/gt0.png"
p_gt0_v13 = "assets/results/v13_base_50k/eval_samples_ctrl/step0075000/strict/gt0.png"
if os.path.exists(p_gt0_v10b) and os.path.exists(p_gt0_v13):
    im_v10b = Image.open(p_gt0_v10b)
    im_v13 = Image.open(p_gt0_v13)
    print("v10b gt0 size:", im_v10b.size, "v13 gt0 size:", im_v13.size)
