import os
import json
import csv

base = "/root/Workspace/xy/DiT"
manifest_path = os.path.join(base, "exp_milestones/milestones_manifest.json")
with open(manifest_path, "r", encoding="utf-8") as f:
    milestones = json.load(f)

print("=== 查看各阶段训练时配置的 eval_csv / eval 规格 ===")
for m in milestones:
    sid = m["stage_id"]
    cfg_p = m["resolved_cfg"] if os.path.exists(m["resolved_cfg"]) else m["config"]
    if os.path.exists(cfg_p):
        with open(cfg_p, "r", encoding="utf-8") as f:
            c = json.load(f)
        eval_csv = c.get("eval_csv")
        eval_csvs = c.get("eval_csvs")
        skel_shards = c.get("eval_skel_latent_shards_dir") or c.get("skel_latent_shards_dir")
        print(f"[{sid}] {m['name']}")
        print(f"   eval_csv: {eval_csv}")
        print(f"   eval_csvs: {eval_csvs}")
        print(f"   skel_shards: {skel_shards}")
    else:
        print(f"[{sid}] 未找到 config: {cfg_p}")
