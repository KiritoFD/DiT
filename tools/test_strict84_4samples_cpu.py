import os, sys, json, csv, re, time
import numpy as np
import torch as th
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

th.set_num_threads(32)
dev = th.device("cpu")

from src.eval import model_io
from src.eval.in_mem_eval import _get_vae
from src.eval.inference import sample_latents, build_diffusion
from src.model.joint_skel2img import JointSkel2Img
import torchvision.transforms as T

ckpt_path = "exp/v35_union/20261002-003600-v35-union-1step/checkpoints/0006000.pt"
print(f"[1] 载入 Checkpoint Step 6,000: {ckpt_path}")
ck = th.load(ckpt_path, map_location="cpu")
gen_sd = ck["gen_ema"]
gen_ckpt = ck["gen_ckpt"]
bak_ckpt = ck["bak_ckpt"]

gen, ga = model_io.load_model_from_ckpt(gen_ckpt, device=dev, use_ema=True)
bak, ba = model_io.load_model_from_ckpt(bak_ckpt, device=dev, use_ema=True)

gen.load_state_dict({(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in gen_sd.items()})
gen.eval()
bak.eval()

# 统一模型: Stage 1 生成 25 步
wrapper = JointSkel2Img(gen, bak, gen_steps=25).to(dev).eval()

# 官方 50k_v2 标准字骨架分片
shards_dir = "data/50k_v2_glyph15k/shards_std"
shard_cache = {}
id_to_shard = {}
import glob
for p in sorted(glob.glob(os.path.join(shards_dir, "shard_*.npz"))):
    with np.load(p) as z:
        for j, iid in enumerate(z["img_ids"]):
            id_to_shard[int(iid)] = (p, j)

def get_std_lat(iid):
    p, j = id_to_shard[iid]
    if p not in shard_cache:
        with np.load(p) as z:
            shard_cache[p] = np.array(z["latents"], copy=True)
    return shard_cache[p][j]

csv_path = "assets/eval_top10_strict_subset84.csv"
rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))[:4]
print(f"[2] 选取前 4 条 Strict84 样本测试:")
for i, r in enumerate(rows):
    print(f"  Sample {i}: 书家={r['calligrapher']} 书体={r['script']} 字={r['character']} img_path={r['image_path']}")

csmap = json.load(open("assets/callig_script_id_map_top10.json", encoding="utf-8"))
from src.utils.callig_script_map import map_callig_script

g_stds, conds, gts = [], [], []
tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])

for r in rows:
    iid = int(re.search(r"(\d+)\.png", r["image_path"]).group(1))
    old_id = int(r.get("old_50k_id", iid))
    lat = get_std_lat(old_id if old_id in id_to_shard else iid)
    g_stds.append(lat)
    cid = map_callig_script(int(r["calligrapher_id"]), int(r["script_id"]), csmap)
    conds.append((cid, int(r.get("glyph_id", 0))))
    gts.append(tf(Image.open(r["image_path"]).convert("RGB")))

g_stds = th.from_numpy(np.stack(g_stds)).float().to(dev)
gts = th.stack(gts).to(dev)
noise = th.randn(4, 4, 32, 32, generator=th.Generator().manual_seed(0))

diff = build_diffusion(50, "flow")

print("\n[3] 运行 JointSkel2Img 采样 (4 samples, 25 gen + 50 bak steps)...")
t0 = time.time()
with th.no_grad():
    x_pred = sample_latents(wrapper, diff, noise, conds, cfg_scale=1.0, batch=4, device=dev, skel=g_stds)
dt = time.time() - t0
print(f"  采样完成! 耗时: {dt:.1f}s, x_pred shape: {x_pred.shape}")

print("[4] VAE 解码与 SSIM 计算...")
vae = _get_vae(dev, "data/pretrained/pretrained_models/sd-vae-ft-ema")
with th.no_grad():
    dec = (vae.decode(x_pred / 0.18215).sample.clamp(-1, 1) + 1) / 2
    gts_norm = (gts + 1) / 2

from src.eval.in_mem_eval import _ssim_tensor
ssim_vals = _ssim_tensor(dec, gts_norm)
print(f"\n🎉 4 样本真实 Strict84 E2E SSIM 结果:")
for i, (r, ssim) in enumerate(zip(rows, ssim_vals)):
    print(f"  [{i}] {r['calligrapher']} · {r['character']} ({r['script']}): SSIM = {ssim:.4f}")
print(f"  ==> 均值 SSIM: {float(np.mean(ssim_vals)):.4f}")
