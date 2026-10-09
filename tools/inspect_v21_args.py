import os
import torch
import json

p = "/root/Workspace/xy/DiT/archive_experiments/results_archive_failed/failed_skelnet_runs/v21_skelnet_200k/20260926-005749-v21-skelnet-200k/checkpoints/0155000.pt"
ckpt = torch.load(p, map_location="cpu")
a = vars(ckpt.get("args", {}))
for k in ["num_calligraphers", "callig_script_map", "callig_id_map", "model", "use_script_cond", "callig_embed_dim"]:
    print(k, ":", a.get(k))

for k, v in (ckpt.get("ema") or ckpt.get("model")).items():
    if "callig" in k:
        print("sd key:", k, v.shape)
