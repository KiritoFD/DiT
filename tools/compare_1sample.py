import os, sys, json, csv, re, time
import torch as th
import numpy as np
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

th.set_num_threads(32)
dev = th.device("cpu")

from src.eval import model_io
from src.model.joint_skel2img import JointSkel2Img
from src.eval.inference import sample_latents, build_diffusion
from src.eval.in_mem_eval import _get_vae
from src.eval.metrics import ssim_torch
from src.utils.callig_script_map import map_callig_script
import torchvision.transforms as T

# 1. 载入模型 (Step 6,000 ckpt)
ckpt_path = "exp/v35_union/20261002-003600-v35-union-1step/checkpoints/0006000.pt"
print(f"[1] Loading ckpt: {ckpt_path}")
ck = th.load(ckpt_path, map_location="cpu")
gen, _ = model_io.load_model_from_ckpt(ck["gen_ckpt"], device=dev, use_ema=True)
bak, ba = model_io.load_model_from_ckpt(ck["bak_ckpt"], device=dev, use_ema=True)
gen.load_state_dict({(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in ck["gen_ema"].items()})
gen.eval()
bak.eval()

wrapper = JointSkel2Img(gen, bak, gen_steps=25).to(dev).eval()
diff = build_diffusion(50, "flow")

# 2. 载入分片索引
import glob
def build_index(shards_dir):
    idx = {}
    for p in sorted(glob.glob(os.path.join(shards_dir, "shard_*.npz"))):
        with np.load(p) as z:
            for j, iid in enumerate(z["img_ids"]):
                idx[int(iid)] = (p, j)
    return idx

idx_seen = build_index("data/top10_style23/shards_std")
idx_strict = build_index("data/50k_v2_glyph15k/shards_std")

csmap = json.load(open("assets/callig_script_id_map_top10.json", encoding="utf-8"))
tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])

# Sample 0 of Seen
r_seen = list(csv.DictReader(open("assets/eval_top10_seen_20.csv", encoding="utf-8")))[0]
iid_seen = int(re.search(r"(\d+)\.png", r_seen["image_path"]).group(1))
p, j = idx_seen[iid_seen]
with np.load(p) as z:
    g_seen = th.from_numpy(z["latents"][j:j+1]).float().to(dev)
cid_seen = map_callig_script(int(r_seen["calligrapher_id"]), int(r_seen["script_id"]), csmap)
cond_seen = [(cid_seen, int(r_seen.get("glyph_id", 0)))]
gt_seen = tf(Image.open(r_seen["image_path"]).convert("RGB")).unsqueeze(0).to(dev)

# Sample 0 of Strict
r_strict = list(csv.DictReader(open("assets/eval_top10_strict_subset84.csv", encoding="utf-8")))[0]
iid_strict = int(re.search(r"(\d+)\.png", r_strict["image_path"]).group(1))
old_id = int(r_strict.get("old_50k_id", iid_strict))
target_id = old_id if old_id in idx_strict else iid_strict
p, j = idx_strict[target_id]
with np.load(p) as z:
    g_strict = th.from_numpy(z["latents"][j:j+1]).float().to(dev)
cid_strict = map_callig_script(int(r_strict["calligrapher_id"]), int(r_strict["script_id"]), csmap)
cond_strict = [(cid_strict, int(r_strict.get("glyph_id", 0)))]
gt_strict = tf(Image.open(r_strict["image_path"]).convert("RGB")).unsqueeze(0).to(dev)

print("\n[2] Sample Info:")
print(f"  Seen:   {r_seen['calligrapher']} · {r_seen['character']} ({r_seen['script']}), cid={cid_seen}, g_mean={g_seen.mean():.4f}")
print(f"  Strict: {r_strict['calligrapher']} · {r_strict['character']} ({r_strict['script']}), cid={cid_strict}, g_mean={g_strict.mean():.4f}")

noise = th.randn(1, 4, 32, 32, generator=th.Generator().manual_seed(0))

print("\n[3] 运行 Seen 样本采样 (JointSkel2Img)...")
t0 = time.time()
with th.no_grad():
    x_seen = sample_latents(wrapper, diff, noise.clone(), cond_seen, 1.0, 1, dev, skel=g_seen)
print(f"  Seen 采样完成! 耗时 {time.time()-t0:.1f}s, output mean={x_seen.mean():.4f}")

print("\n[4] 运行 Strict 样本采样 (JointSkel2Img)...")
t0 = time.time()
with th.no_grad():
    x_strict = sample_latents(wrapper, diff, noise.clone(), cond_strict, 1.0, 1, dev, skel=g_strict)
print(f"  Strict 采样完成! 耗时 {time.time()-t0:.1f}s, output mean={x_strict.mean():.4f}")

vae = _get_vae(dev, "data/pretrained/pretrained_models/sd-vae-ft-ema")
with th.no_grad():
    dec_seen = (vae.decode(x_seen / 0.18215).sample.clamp(-1, 1) + 1) / 2
    dec_strict = (vae.decode(x_strict / 0.18215).sample.clamp(-1, 1) + 1) / 2
    gt_s_norm = (gt_seen + 1) / 2
    gt_st_norm = (gt_strict + 1) / 2

ssim_s = ssim_torch(dec_seen, gt_s_norm).item()
ssim_st = ssim_torch(dec_strict, gt_st_norm).item()

print(f"\n" + "="*50)
print(f"【实测对比结果】:")
print(f"  Seen 样本 ({r_seen['calligrapher']} · {r_seen['character']}):   SSIM = {ssim_s:.4f}")
print(f"  Strict 样本 ({r_strict['calligrapher']} · {r_strict['character']}): SSIM = {ssim_st:.4f}")
print(f"="*50)
