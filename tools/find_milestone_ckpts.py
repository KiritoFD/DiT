import os
import glob
import json

stages = {
    "v10b": [
        "/root/Workspace/xy/DiT/assets/results/v10b_stdskel_fame3_c41x_cos_e/**/checkpoints/*.pt",
        "/root/Workspace/xy/DiT/assets/results/v10b_stdskel_fame3_c41x/**/checkpoints/*.pt",
        "/root/Workspace/xy/DiT/assets/results/v10b*/**/*.pt"
    ],
    "v13": [
        "/root/Workspace/xy/DiT/assets/results/v13_base_50k/**/checkpoints/*.pt",
        "/root/Workspace/xy/DiT/assets/results/v13_12ch_post/**/checkpoints/*.pt",
        "/root/Workspace/xy/DiT/assets/results/v13*/**/*.pt"
    ],
    "v21": [
        "/root/Workspace/xy/DiT/archive_experiments/results_archive_failed/failed_skelnet_runs/v21_skelnet_200k/**/checkpoints/*.pt",
        "/root/Workspace/xy/DiT/assets/*v21*.pt"
    ],
    "v23": [
        "/root/Workspace/xy/DiT/archive_experiments/_archive_historical/20261003_twostage/assets_results/v23_splitnorm/**/checkpoints/*.pt",
        "/root/Workspace/xy/DiT/assets/*v23*.pt"
    ],
    "v66": [
        "/root/Workspace/xy/DiT/assets/results/v66_tables_condroute2456/**/checkpoints/*.pt"
    ],
    "v68": [
        "/root/Workspace/xy/DiT/assets/results/v68_aug_sp_c2ot/**/checkpoints/*.pt"
    ],
    "v70": [
        "/root/Workspace/xy/DiT/assets/results/v70_aug_sp_stdskel_c2ot/**/checkpoints/*.pt"
    ]
}

print("=== Scanning Milestone Checkpoints on 4090 ===")
found_ckpts = {}
for name, patterns in stages.items():
    all_ckpts = []
    for pat in patterns:
        all_ckpts.extend(glob.glob(pat, recursive=True))
    all_ckpts = sorted(set([c for c in all_ckpts if os.path.isfile(c) and not c.endswith(".tmp")]))
    found_ckpts[name] = all_ckpts
    print(f"\n[{name}] Found {len(all_ckpts)} checkpoints:")
    for c in all_ckpts[-4:]:
        sz_mb = os.path.getsize(c) / (1024 * 1024)
        print(f"  {c} ({sz_mb:.1f} MB)")

# Also find resolved_config.json for each
print("\n=== Scanning Configs ===")
for name in stages:
    ckpts = found_ckpts.get(name, [])
    if not ckpts:
        continue
    # look for resolved_config.json in parent directories
    c = ckpts[-1]
    cur = os.path.dirname(c)
    cfg_found = None
    for _ in range(3):
        cand = os.path.join(cur, "resolved_config.json")
        if os.path.exists(cand):
            cfg_found = cand
            break
        cand_cfg = glob.glob(os.path.join(cur, "*.json"))
        if cand_cfg:
            cfg_found = cand_cfg[0]
            break
        cur = os.path.dirname(cur)
    print(f"[{name}] config: {cfg_found}")
