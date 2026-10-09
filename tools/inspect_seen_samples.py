import os
import pandas as pd
from PIL import Image

os.chdir("/root/Workspace/xy/DiT")

df_seen = pd.read_csv("assets/eval_seen_v10.csv")
print("=== eval_seen_v10.csv (all 10 rows) ===")
for idx, r in df_seen.iterrows():
    print(f"[{idx}] {r['character']} | {r['calligrapher']} | {r['script']} | {r['image_path']}")

models_g = {
    "v10b (390k)": "assets/results/v10b_stdskel_fame3_c41x_cos_e/20260909-193018-v10b-stdskel-fame3-c41x-cos-e/eval_samples_ctrl/step0390000/g",
    "v13 (155k)": "assets/results/v13_base_50k/20260917-211905-v13-base-50k/eval_samples_ctrl/step0155000/g",
    "v21_skelnet (75k)": "archive_experiments/results_archive_failed/failed_skelnet_runs/v21_skelnet_200k/eval_samples_ctrl/step0075000/g",
    "v23_splitnorm (75k)": "archive_experiments/_archive_historical/20261003_twostage/assets_results/v23_splitnorm/eval_samples_ctrl/step0075000/g",
    "std_aug (75k)": "assets/results/std_callig_aug/eval_samples_ctrl/step0075000/g",
    "v66 (150k)": "assets/results/v66_tables_condroute2456/20261006-024102-v66-tables-condroute2456-adaLN/eval_samples_ctrl/step0150000/g",
    "v68_sp (200k)": "assets/results/v68_aug_sp_c2ot/20261006-192021-v68-aug-sp-c2ot/eval_samples_ctrl/step0200000/g"
}

print("\n=== Availability of g0..g9 in each model ===")
for m_name, m_dir in models_g.items():
    if not os.path.exists(m_dir):
        print(f"{m_name}: DIRECTORY NOT FOUND: {m_dir}")
        continue
    available = []
    for i in range(10):
        gp = os.path.join(m_dir, f"g{i}.png")
        if os.path.exists(gp):
            available.append(i)
    print(f"{m_name}: {len(available)} samples available: {available}")
