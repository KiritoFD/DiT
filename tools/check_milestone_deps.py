import os
import glob
import json

base_dir = "/root/Workspace/xy/DiT"

# 1. 检查三表 remap json
triple_dir = os.path.join(base_dir, "assets/triple_tables_best_minimal")
print("triple_tables_best_minimal exists:", os.path.exists(triple_dir))
if os.path.exists(triple_dir):
    for fn in os.listdir(triple_dir):
        fp = os.path.join(triple_dir, fn)
        sz = os.path.getsize(fp)
        print(f"  {fn} ({sz} bytes)")

# 2. 检查 v10b 评测时使用的 callig_id_map
v10_cfg = os.path.join(base_dir, "src/train/configs/v10b_stdskel_fame3_c41x_cos_e.json")
if os.path.exists(v10_cfg):
    with open(v10_cfg, encoding="utf-8") as f:
        c = json.load(f)
    print("v10b callig_id_map:", c.get("callig_id_map"))
    print("v10b callig_script_map:", c.get("callig_script_map"))

# 3. 检查 v13 评测时使用的 callig_id_map
v13_cfg = os.path.join(base_dir, "src/train/configs/v13_base_50k.json")
if os.path.exists(v13_cfg):
    with open(v13_cfg, encoding="utf-8") as f:
        c = json.load(f)
    print("v13 callig_id_map:", c.get("callig_id_map"))
    print("v13 callig_script_map:", c.get("callig_script_map"))

# 4. 检查骨架 latent shards 目录
shards = glob.glob(os.path.join(base_dir, "assets/*shard*")) + glob.glob(os.path.join(base_dir, "data/*shard*")) + glob.glob(os.path.join(base_dir, "exp-std/*shard*"))
print("Found shard dirs:", shards)

# 5. 检查 eval200 是否有预先提取好的 skel latents
std_dir = os.path.join(base_dir, "exp-std/data/std_fixed_eval200")
print("std_fixed_eval200 exists:", os.path.exists(std_dir), "count:", len(os.listdir(std_dir)) if os.path.exists(std_dir) else 0)
