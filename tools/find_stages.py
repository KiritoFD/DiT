import os
import glob

dirs_to_check = {
    "v10b": [
        "/root/Workspace/xy/DiT/assets/results/v10b_stdskel_fame3_c41x_cos_e/*/eval_samples_ctrl",
        "/root/Workspace/xy/DiT/assets/results/v10b_stdskel_fame3_c41x/*/eval_samples_ctrl",
        "/root/Workspace/xy/DiT/assets/results/v10b_stdskel_fame3_c41x_cos/*/eval_samples_ctrl",
    ],
    "v13": [
        "/root/Workspace/xy/DiT/assets/results/v13_base_50k/*/eval_samples_ctrl",
        "/root/Workspace/xy/DiT/assets/results/v13_12ch_post/*/eval_samples_ctrl",
        "/root/Workspace/xy/DiT/assets/results/v13_styletok/*/eval_samples_ctrl",
    ],
    "v20_v21_v23": [
        "/root/Workspace/xy/DiT/archive_experiments/results_archive_failed/failed_skelnet_runs/v20_deform_skel_100k/*/eval_samples_ctrl",
        "/root/Workspace/xy/DiT/archive_experiments/results_archive_failed/failed_skelnet_runs/v21_skelnet_200k/*/eval_samples_ctrl",
        "/root/Workspace/xy/DiT/archive_experiments/_archive_historical/20261003_twostage/assets_results/v23_splitnorm/*/eval_samples_ctrl",
        "/root/Workspace/xy/DiT/assets/results/std_callig_aug/eval_samples_ctrl",
    ],
    "v64_v66_v68": [
        "/root/Workspace/xy/DiT/assets/results/v64_skel_tables_4ch/*/eval_samples_ctrl",
        "/root/Workspace/xy/DiT/assets/results/v65_tables_condinj2468/*/eval_samples_ctrl",
        "/root/Workspace/xy/DiT/assets/results/v66_tables_condroute2456/*/eval_samples_ctrl",
        "/root/Workspace/xy/DiT/assets/results/v68_aug_sp_c2ot/*/eval_samples_ctrl",
    ]
}

for group, patterns in dirs_to_check.items():
    print(f"=== Group: {group} ===")
    for pat in patterns:
        matches = glob.glob(pat)
        for m in matches:
            if os.path.exists(m):
                steps = sorted(os.listdir(m))
                print(f"  {m} ({len(steps)} steps)")
                if steps:
                    print(f"    sample steps: {steps[:5]} ... {steps[-2:]}")
                    # check subdirs in latest step
                    last_step = os.path.join(m, steps[-1])
                    subdirs = os.listdir(last_step) if os.path.isdir(last_step) else []
                    print(f"    subdirs in {steps[-1]}: {subdirs}")
