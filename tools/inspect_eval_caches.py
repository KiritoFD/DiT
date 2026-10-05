import os, sys, json, csv
ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.eval import inference
from types import SimpleNamespace

args = SimpleNamespace(
    callig_script_map="assets/callig_script_id_map_top10.json",
    callig_id_map=None,
    eval_skel_latent_shards_dir="data/top10_style23/predskel_std84_e2e",
    shards_std="data/top10_style23/shards_std",
)
csmap = json.load(open(args.callig_script_map, encoding="utf-8"))

c_seen = inference.make_eval_cache("assets/eval_top10_seen_20.csv", None, None, 256, 20, 8, 4, 0.18215,
                                  skel_latent_shards_dir=args.eval_skel_latent_shards_dir,
                                  callig_script_map=csmap)

c_strict = inference.make_eval_cache("assets/eval_v13_strict84_aligned.csv", None, None, 256, 84, 8, 4, 0.18215,
                                    skel_latent_shards_dir=args.eval_skel_latent_shards_dir,
                                    callig_script_map=csmap)

print("Seen20 conds[:3]:", c_seen["conds"][:3])
print("Strict84 conds[:3]:", c_strict["conds"][:3])

print("Seen20 skels_latent shape:", c_seen["skels_latent"].shape, "mean:", c_seen["skels_latent"].mean().item())
print("Strict84 skels_latent shape:", c_strict["skels_latent"].shape, "mean:", c_strict["skels_latent"].mean().item())
