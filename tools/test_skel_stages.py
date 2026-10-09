import os
import sys

BASE_DIR = "/root/Workspace/xy/DiT"
if os.path.exists(BASE_DIR):
    os.chdir(BASE_DIR)

import torch
import json
import csv
from PIL import Image
import torchvision.transforms as T
from diffusers.models import AutoencoderKL
from src.eval.model_io import build_model_from_args
from src.loss import create_diffusion_or_flow

print("=== 测试骨架输入类模型 (v10b, v13, v70) 单步采样流程 ===")
torch.set_num_threads(16)
device = torch.device("cpu")

# 1. 载入 VAE
vae_path = "/root/Workspace/xy/DiT/pretrained_models/sd-vae-ft-ema"
vae = AutoencoderKL.from_pretrained(vae_path).to(device).eval()
for p in vae.parameters():
    p.requires_grad_(False)
sf = 0.18215

# 2. 读取 eval200 的第 0 个样本
eval_csv = "exp-std/csv/eval200_fixed.csv"
rows = list(csv.DictReader(open(eval_csv, encoding="utf-8")))
r0 = rows[0]
print(f"样本 #0: {r0['character']} | {r0['calligrapher']} | {r0['script']} | std: {r0['std_path']}")

# 3. 准备骨架 latent
std_img = Image.open(r0["std_path"]).convert("RGB").resize((256, 256))
tf = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
x_skel = tf(std_img).unsqueeze(0).to(device)
with torch.no_grad():
    skel_latent = vae.encode(x_skel).latent_dist.sample() * sf
print(f"骨架 latent: {skel_latent.shape}")

# 4. 固定噪声
g = torch.Generator(device="cpu").manual_seed(42)
z_init = torch.randn(1, 4, 32, 32, generator=g, device=device)

# 测试目标阶段
test_stages = [
    {
        "id": "v10b",
        "ckpt": "exp_milestones/01_v10b/checkpoint.pt",
        "cfg": "exp_milestones/01_v10b/resolved_config.json",
        "diff_type": "ddpm",
        "callig_map": "assets/callig_id_map.json"
    },
    {
        "id": "v13",
        "ckpt": "exp_milestones/02_v13/checkpoint.pt",
        "cfg": "exp_milestones/02_v13/resolved_config.json",
        "diff_type": "flow",
        "callig_map": "assets/callig_id_map_50k.json"
    },
    {
        "id": "v70",
        "ckpt": "exp_milestones/07_v70/checkpoint.pt",
        "cfg": "exp_milestones/07_v70/resolved_config.json",
        "diff_type": "flow",
        "callig_remap": "assets/triple_tables_best_minimal/callig_remap.json"
    }
]

for s in test_stages:
    sid = s["id"]
    print(f"\n--- 测试阶段: {sid} ---")
    ckpt = torch.load(s["ckpt"], map_location="cpu")
    cfg = json.load(open(s["cfg"], encoding="utf-8"))
    for k, v in vars(ckpt.get("args", {})).items():
        if v is not None:
            cfg[k] = v
    model = build_model_from_args(cfg, device="cpu").eval()
    sd = ckpt.get("ema") or ckpt.get("model") or ckpt
    clean_sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean_sd, strict=False)

    # 构造条件
    callig_name = r0["calligrapher"]
    cid_raw = int(r0["calligrapher_id"])
    
    if "callig_remap" in s:
        # v70 走 top10 remap
        remap = {int(k): int(v) for k, v in json.load(open(s["callig_remap"])).items()}
        y_callig = torch.tensor([remap.get(cid_raw, 0)], device=device)
    elif "callig_map" in s:
        # 查历史映射表
        m_data = json.load(open(s["callig_map"]))
        # 处理可能的不同嵌套结构
        id_map = m_data.get("id_map", m_data)
        # 支持以名字或 raw_id 查询
        mapped_id = id_map.get(str(cid_raw), id_map.get(callig_name, 0))
        y_callig = torch.tensor([int(mapped_id)], device=device)
    else:
        y_callig = torch.tensor([0], device=device)

    # 纯骨架模型无字表，y_char 传入 dummy
    y_char = torch.tensor([0], device=device)

    # 采样 5 步
    learn_sigma = bool(cfg.get("learn_sigma", False))
    diff = create_diffusion_or_flow("5", diffusion_type=s["diff_type"], learn_sigma=learn_sigma)
    model_kwargs = {
        "y_callig": y_callig,
        "y_char": y_char,
        "g": skel_latent
    }
    with torch.no_grad():
        out_latent = diff.ddim_sample_loop(
            model, z_init.shape, z_init,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            device=device
        )
    print(f"[{sid}] 采样成功: shape={out_latent.shape}, mean={out_latent.mean().item():.3f}")
