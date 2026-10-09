import os
import glob

dirs = {
    "v10b_cos_e": "/root/Workspace/xy/DiT/assets/results/v10b_stdskel_fame3_c41x_cos_e/20260909-193018-v10b-stdskel-fame3-c41x-cos-e/eval_samples_ctrl",
    "v10b_c41x": "/root/Workspace/xy/DiT/assets/results/v10b_stdskel_fame3_c41x/20260908-232551-v10b-stdskel-fame3-c41x/eval_samples_ctrl",
    "v13_base": "/root/Workspace/xy/DiT/assets/results/v13_base_50k/20260917-211905-v13-base-50k/eval_samples_ctrl",
    "v13_12ch": "/root/Workspace/xy/DiT/assets/results/v13_12ch_post/20260918-082201-v13-12ch-post/eval_samples_ctrl",
    "v20": "/root/Workspace/xy/DiT/archive_experiments/results_archive_failed/failed_skelnet_runs/v20_deform_skel_100k/20260925-194549-v20-deform-skel-100k/eval_samples_ctrl",
    "v21": "/root/Workspace/xy/DiT/archive_experiments/results_archive_failed/failed_skelnet_runs/v21_skelnet_200k/20260926-005749-v21-skelnet-200k/eval_samples_ctrl",
    "std_callig_aug": "/root/Workspace/xy/DiT/assets/results/std_callig_aug/eval_samples_ctrl",
    "v23": "/root/Workspace/xy/DiT/archive_experiments/_archive_historical/20261003_twostage/assets_results/v23_splitnorm/20260927-124752-v23-splitnorm/eval_samples_ctrl",
    "v66": "/root/Workspace/xy/DiT/assets/results/v66_tables_condroute2456/20261006-024102-v66-tables-condroute2456-adaLN/eval_samples_ctrl",
    "v68": "/root/Workspace/xy/DiT/assets/results/v68_aug_sp_c2ot/20261006-192021-v68-aug-sp-c2ot/eval_samples_ctrl"
}

for name, path in dirs.items():
    if not os.path.exists(path):
        print(f"[{name}] NOT FOUND: {path}")
        continue
    steps = sorted(os.listdir(path))
    print(f"[{name}] {len(steps)} steps:")
    for s in steps[:3] + steps[-2:]:
        sp = os.path.join(path, s)
        if os.path.isdir(sp):
            subdirs = sorted(os.listdir(sp))
            print(f"   {s} -> {subdirs[:5]}")
            # If there is a subfolder, check file count inside
            for sub in subdirs[:3]:
                subp = os.path.join(sp, sub)
                if os.path.isdir(subp):
                    files = sorted(os.listdir(subp))
                    print(f"      {sub} has {len(files)} files: {files[:4]}")
                elif os.path.isfile(subp):
                    pass
