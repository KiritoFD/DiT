import os
import json

base = "/root/Workspace/xy/DiT"
cfg_files = [
    os.path.join(base, "src/train/configs/v13_base_50k.json"),
    os.path.join(base, "assets/results/v13_wd01/20260918-210256-v13-base-50k/resolved_config.json")
]

for cp in cfg_files:
    if os.path.exists(cp):
        print(f"=== Config: {cp} ===")
        with open(cp, "r", encoding="utf-8") as f:
            c = json.load(f)
        for k in [
            "experiment_name", "model", "skel_as_glyph_cond", "skel_latent_shards_dir",
            "eval_skel_latent_shards_dir", "aux_latent_shards_dirs", "w_aux_skel",
            "train_csv", "eval_csv", "no_char_cond", "dataset_path", "_comment"
        ]:
            if k in c:
                print(f"  {k}: {c[k]}")

# 检查 data/50k/shards_std 的来源
print("\n=== Checking data/50k Shard Metadata ===")
shard_std = os.path.join(base, "data/50k/shards_std")
if os.path.exists(shard_std):
    files = os.listdir(shard_std)[:5]
    print(f"shards_std found: {len(os.listdir(shard_std))} files, e.g. {files}")
else:
    print(f"shards_std NOT found at {shard_std}")

shard_aux = os.path.join(base, "data/50k/shards_aux_skel3")
if os.path.exists(shard_aux):
    print(f"shards_aux_skel3 found: {len(os.listdir(shard_aux))} files")
else:
    print(f"shards_aux_skel3 NOT found")
