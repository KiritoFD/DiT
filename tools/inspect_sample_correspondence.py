import os
import glob
import pandas as pd
from PIL import Image

os.chdir("/root/Workspace/xy/DiT")

df_v13 = pd.read_csv("assets/eval_v13_strict_fixed.csv")
print(f"eval_v13_strict_fixed.csv has {len(df_v13)} rows")

# Let's inspect the files in v13, v21, v23, std_aug, v10b
p_v13 = "assets/results/v13_base_50k/20260917-211905-v13-base-50k/eval_samples_ctrl/step0155000/strict50"
p_v13_75k = "assets/results/v13_base_50k/eval_samples_ctrl/step0075000/strict"
p_v21_75k = "archive_experiments/results_archive_failed/failed_skelnet_runs/v21_skelnet_200k/eval_samples_ctrl/step0075000/strict"
p_v23_75k = "archive_experiments/_archive_historical/20261003_twostage/assets_results/v23_splitnorm/eval_samples_ctrl/step0075000/strict"
p_stdaug_75k = "assets/results/std_callig_aug/eval_samples_ctrl/step0075000/strict"
p_v10b_390k = "assets/results/v10b_stdskel_fame3_c41x_cos_e/20260909-193018-v10b-stdskel-fame3-c41x-cos-e/eval_samples_ctrl/step0390000/strict50"
p_v10b_seen = "assets/results/v10b_stdskel_fame3_c41x_cos_e/20260909-193018-v10b-stdskel-fame3-c41x-cos-e/eval_samples_ctrl/step0390000/g"

for label, p in [
    ("v13_155k_strict50", p_v13),
    ("v13_75k_strict", p_v13_75k),
    ("v21_75k_strict", p_v21_75k),
    ("v23_75k_strict", p_v23_75k),
    ("stdaug_75k_strict", p_stdaug_75k),
    ("v10b_390k_strict50", p_v10b_390k),
    ("v10b_390k_g", p_v10b_seen),
]:
    full_p = os.path.join("/root/Workspace/xy/DiT", p)
    if os.path.exists(full_p):
        files = sorted(os.listdir(full_p))
        print(f"[{label}] exists: {len(files)} files, e.g., {files[:5]}")
    else:
        print(f"[{label}] NOT FOUND: {full_p}")

matches = glob.glob("/root/Workspace/xy/DiT/assets/results/v13_base_50k/**/step0075000*", recursive=True)
print(f"v13 step0075000 matches: {matches}")
matches_v13_any = glob.glob("/root/Workspace/xy/DiT/assets/results/v13_base_50k/**/strict*", recursive=True)
print(f"v13 strict matches: {matches_v13_any}")

# Also check v66 and v68
for v_name, v_path in [
    ("v66_150k", "assets/results/v66_tables_condroute2456/20261006-024102-v66-tables-condroute2456-adaLN/eval_samples_ctrl/step0150000/eval200fix"),
    ("v68_200k", "assets/results/v68_aug_sp_c2ot/20261006-192021-v68-aug-sp-c2ot/eval_samples_ctrl/step0200000/eval200fix"),
]:
    full_p = os.path.join("/root/Workspace/xy/DiT", v_path)
    if os.path.exists(full_p):
        print(f"[{v_name}] exists: {len(os.listdir(full_p))} files")
    else:
        print(f"[{v_name}] NOT FOUND")
