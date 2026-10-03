import os, sys, csv
import numpy as np
import torch as th
from PIL import Image

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")

eval_csv = "assets/eval_top10_real_200.csv"
callig_map_path = "assets/callig_script_id_map_top10.json"
shards_gt_dir = "data/top10_style23/shards_gtskel_w7"
shards_std_dir = "data/top10_style23/shards_std_w7"
out_cache_path = "data/top10_style23/eval_real200_cache.pt"

from src.utils.callig_script_map import load_callig_script_map
from src.eval.in_mem_eval import _get_vae

dev = th.device("cuda" if th.cuda.is_available() else "cpu")
vae = _get_vae(dev, "data/pretrained/pretrained_models/sd-vae-ft-ema").eval()
csmap = load_callig_script_map(callig_map_path)

with open(eval_csv, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"Building cache for {len(rows)} samples in {eval_csv}...")

# 1. Load shard indices
def index_shards(sdir):
    import glob
    shards = sorted(glob.glob(os.path.join(sdir, "shard_*.npz")))
    id_map = {}
    for sp in shards:
        d = np.load(sp)
        key = "img_ids" if "img_ids" in d else "ids"
        ids = [int(x) for x in d[key].tolist()]
        for j, iid in enumerate(ids):
            id_map[iid] = (sp, j)
    return id_map

gt_id_map = index_shards(shards_gt_dir)
std_id_map = index_shards(shards_std_dir)

# 2. Extract latents for each eval sample
gt_lats = []
std_lats = []
conds = []
img_ids = []

for r in rows:
    iid = int(r["img_id"])
    gid = int(r.get("glyph_id", 0))
    if "pair_id" in r and r["pair_id"] != "":
        cid = int(r["pair_id"])
    else:
        slot = r["slot_name"]
        pair_key = f"{r['calligrapher_id']}:{r['script_id']}"
        cid = csmap["pair_map"][pair_key] if csmap and "pair_map" in csmap else int(r["calligrapher_id"])
    conds.append((cid, gid))
    img_ids.append(iid)

    sp_gt, j_gt = gt_id_map[iid]
    d_gt = np.load(sp_gt)
    gt_lats.append(d_gt["latents"][j_gt])

    sp_std, j_std = std_id_map[iid]
    d_std = np.load(sp_std)
    std_lats.append(d_std["latents"][j_std])

gt_lats = th.from_numpy(np.stack(gt_lats)).float()
std_lats = th.from_numpy(np.stack(std_lats)).float()

# 3. Fixed deterministic evaluation noise
g_gen = th.Generator().manual_seed(42)
noise = th.randn(len(rows), 4, 32, 32, generator=g_gen)

# 4. Pre-decode GT pngs and STD pngs (batched in 28)
with th.no_grad():
    gt_pngs = []
    for s in range(0, len(rows), 28):
        _dec = (vae.decode(gt_lats[s:s+28].to(dev) / 0.18215).sample.clamp(-1, 1) + 1) / 2
        gt_pngs.append(_dec.cpu())
    gt_pngs = th.cat(gt_pngs, dim=0)

    std_pngs = []
    for s in range(0, len(rows), 28):
        _dec = (vae.decode(std_lats[s:s+28].to(dev) / 0.18215).sample.clamp(-1, 1) + 1) / 2
        std_pngs.append(_dec.cpu())
    std_pngs = th.cat(std_pngs, dim=0)

cache = {
    "rows": rows,
    "conds": conds,
    "img_ids": img_ids,
    "noise": noise,
    "std_lats": std_lats,
    "gt_lats": gt_lats,
    "std_pngs": std_pngs,
    "gt_pngs": gt_pngs,
}

th.save(cache, out_cache_path)
print(f"Precomputed eval cache saved to {out_cache_path} ({os.path.getsize(out_cache_path)/1e6:.1f} MB)")
print("Verification complete: 200 real eval items ready for ultra-fast GPU evaluation!")
