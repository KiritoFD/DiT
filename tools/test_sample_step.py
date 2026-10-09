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

print("=== 测试单图推理适配流程 ===")
torch.set_num_threads(16)
device = torch.device("cpu")

# 1. 载入 VAE
vae_path = "/root/Workspace/xy/DiT/pretrained_models/sd-vae-ft-ema"
if not os.path.exists(vae_path):
    vae_path = "stabilityai/sd-vae-ft-ema"
print(f"载入 VAE: {vae_path}")
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
std_img_path = r0["std_path"]
std_img = Image.open(std_img_path).convert("RGB").resize((256, 256))
tf = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
x_skel = tf(std_img).unsqueeze(0).to(device)
with torch.no_grad():
    skel_latent = vae.encode(x_skel).latent_dist.sample() * sf
print(f"骨架 latent 生成成功: {skel_latent.shape}, mean={skel_latent.mean().item():.3f}")

# 4. 固定噪声
g = torch.Generator(device="cpu").manual_seed(42)
z_init = torch.randn(1, 4, 32, 32, generator=g, device=device)

# 5. 测试加载 v66 并前向一步
v66_ckpt = "exp_milestones/05_v66/checkpoint.pt"
ckpt = torch.load(v66_ckpt, map_location="cpu")
cfg = json.load(open("exp_milestones/05_v66/resolved_config.json", encoding="utf-8"))
for k, v in vars(ckpt.get("args", {})).items():
    if v is not None:
        cfg[k] = v
model_v66 = build_model_from_args(cfg, device="cpu").eval()
sd = ckpt.get("ema") or ckpt.get("model") or ckpt
clean_sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}
model_v66.load_state_dict(clean_sd, strict=False)

# 读取三表 remap
callig_remap = {int(k): int(v) for k, v in json.load(open("assets/triple_tables_best_minimal/callig_remap.json")).items()}
char_remap = {int(k): int(v) for k, v in json.load(open("assets/triple_tables_best_minimal/char_remap.json")).items()}
font_remap = {int(k): int(v) for k, v in json.load(open("assets/triple_tables_best_minimal/font_remap.json")).items()}

y_callig = torch.tensor([callig_remap[int(r0["calligrapher_id"])]], device=device)
y_char = torch.tensor([char_remap[int(r0["character_id"])]], device=device)
y_font = torch.tensor([font_remap[int(r0["script_id"])]], device=device)

print(f"v66 条件: callig={y_callig.item()}, char={y_char.item()}, font={y_font.item()}")

# 采样 5 步快速验证
diff_v66 = create_diffusion_or_flow("5", diffusion_type="flow")
model_kwargs = {
    "y_callig": y_callig,
    "y_char": y_char,
    "y_script": y_font,
    "g": None
}
print("正在执行 v66 采样 (5 steps)...")
with torch.no_grad():
    sample_latent = diff_v66.ddim_sample_loop(
        model_v66, z_init.shape, z_init,
        model_kwargs=model_kwargs,
        clip_denoised=False,
        device=device
    )
print(f"采样成功: {sample_latent.shape}")

# 解码图像
with torch.no_grad():
    dec = vae.decode(sample_latent / sf).sample
img_out = dec[0].permute(1, 2, 0).clamp(-1, 1).add(1).mul(127.5).byte().cpu().numpy()
Image.fromarray(img_out).save("/tmp/test_v66_out.png")
print("🎉 v66 测试图生成成功: /tmp/test_v66_out.png")
