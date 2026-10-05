import os, sys, json, csv, re
import numpy as np
import torch as th
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

dev = th.device("cuda" if th.cuda.is_available() else "cpu")

from src.eval import model_io
from src.eval.in_mem_eval import _get_vae
from src.eval.inference import sample_latents, build_diffusion
from src.eval.metrics import frag_ratio
from src.utils.callig_script_map import map_callig_script
import torchvision.transforms as T

# Load Step 10,000 ckpt
ckpt_path = "exp/v35_union/20261002-003600-v35-union-1step/checkpoints/0010000.pt"
ck = th.load(ckpt_path, map_location="cpu")
gen, _ = model_io.load_model_from_ckpt(ck["gen_ckpt"], device=dev, use_ema=True)
gen.load_state_dict({(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in ck["gen_ema"].items()})
gen.eval()

# Load v31 baseline gen (before union)
gen_v31, _ = model_io.load_model_from_ckpt(ck["gen_ckpt"], device=dev, use_ema=True)
gen_v31.eval()

# Load 50k_v2 shards_std
shards_dir = "data/50k_v2_glyph15k/shards_std"
import glob
id_to_shard = {}
for p in sorted(glob.glob(os.path.join(shards_dir, "shard_*.npz"))):
    with np.load(p) as z:
        for j, iid in enumerate(z["img_ids"]):
            id_to_shard[int(iid)] = (p, j)

rows = list(csv.DictReader(open("assets/eval_top10_strict_subset84.csv", encoding="utf-8")))[:10]
csmap = json.load(open("assets/callig_script_id_map_top10.json", encoding="utf-8"))

g_stds, conds = [], []
for r in rows:
    iid = int(re.search(r"(\d+)\.png", r["image_path"]).group(1))
    old_id = int(r.get("old_50k_id", iid))
    p, j = id_to_shard[old_id if old_id in id_to_shard else iid]
    with np.load(p) as z:
        g_stds.append(z["latents"][j])
    cid = map_callig_script(int(r["calligrapher_id"]), int(r["script_id"]), csmap)
    conds.append((cid, int(r.get("glyph_id", 0))))

g_stds = th.from_numpy(np.stack(g_stds)).float().to(dev)

diff = build_diffusion(25, "flow")
g_gen = th.Generator(device=dev).manual_seed(0)
noise = th.randn(len(rows), 4, 32, 32, generator=g_gen, device=dev)

with th.no_grad():
    g_pred_v31 = sample_latents(gen_v31, diff, noise.clone(), conds, 1.0, 10, dev, skel=g_stds)
    g_pred_union = sample_latents(gen, diff, noise.clone(), conds, 1.0, 10, dev, skel=g_stds)

vae = _get_vae(dev, "data/pretrained/pretrained_models/sd-vae-ft-ema")
with th.no_grad():
    dec_std = (vae.decode(g_stds.to(dev) / 0.18215).sample.clamp(-1, 1) + 1) / 2
    dec_v31 = (vae.decode(g_pred_v31.to(dev) / 0.18215).sample.clamp(-1, 1) + 1) / 2
    dec_union = (vae.decode(g_pred_union.to(dev) / 0.18215).sample.clamp(-1, 1) + 1) / 2

# Measure fragmentation ratio on skeletons
# frag_ratio: connected components of pred / connected components of std
std_np = dec_std.permute(0, 2, 3, 1).cpu().numpy()
v31_np = dec_v31.permute(0, 2, 3, 1).cpu().numpy()
union_np = dec_union.permute(0, 2, 3, 1).cpu().numpy()

frags_v31 = frag_ratio(v31_np, std_np)
frags_union = frag_ratio(union_np, std_np)

print("=== 骨架破碎度 (frag_ratio) 现场诊断对比 (n=10) ===")
print(f"标准字骨架 基准破碎度 : 1.000")
print(f"v31 原版 Stage 1 骨架破碎度 : {np.mean(frags_v31):.3f}")
print(f"v35 联训后 Stage 1 骨架破碎度: {np.mean(frags_union):.3f}")

from torchvision.utils import save_image
# 拼图: 每行 1 个字，3 列: [标准骨架 | v31 原版预测骨架 | 联训后预测骨架]
montage = th.stack([dec_std.cpu(), dec_v31.cpu(), dec_union.cpu()], dim=1).reshape(-1, 3, 256, 256)
out_png = "exp/v35_union/diag_skel_fragmentation.png"
os.makedirs("exp/v35_union", exist_ok=True)
save_image(montage, out_png, nrow=3, padding=2, normalize=False)
print(f"✓ 诊断对比拼图已保存: {out_png}")
