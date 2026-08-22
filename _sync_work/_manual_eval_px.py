# -*- coding: utf-8 -*-
"""手动 GPU eval pixel 50000 ckpt + 存可视化 PNG."""
import subprocess, time
HOST="root@10.176.54.17"; PORT="36430"

EVAL_SCRIPT = r'''
import os, sys, re, json, time
os.environ["XFORMERS_DISABLED"] = "1"
sys.path.insert(0, "/root/Workspace/xy/DiT/src")
import torch, numpy as np
from PIL import Image
from models import DiT_2Cond
from diffusion import create_diffusion

device = "cuda"
CKPT = "/root/Workspace/xy/DiT/5script/results/px_s_scratch/20260822-145511-px-s-scratch-diff/checkpoints/0055000.pt"
EVAL_CSV = "/root/Workspace/xy/DiT/5script/eval100_top6.csv"
IMG_ROOT = "/root/Workspace/xy/DiT/final_imgs_256"
OUT = "/tmp/px_eval_55k"

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
print("step:", ck.get("train_steps"), flush=True)

model = DiT_2Cond(input_size=256, patch_size=8, in_channels=3,
    hidden_size=384, depth=12, num_heads=6,
    num_calligraphers=1011, num_characters=35130,
    condition_fusion="factorized_add", callig_embed_dim=128, char_embed_dim=256,
    cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
    use_checkpoint=False, learn_sigma=True)
sd = ck.get("ema") or ck.get("delta")
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"load: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
model = model.to(device).eval()
diffusion = create_diffusion(timestep_respacing="")

import csv
rows = list(csv.DictReader(open(EVAL_CSV, encoding="utf-8")))[:20]
gts = []
for s in rows:
    m = re.search(r"(\d+)\.png", s["image_path"])
    img_id = int(m.group(1))
    gt = np.asarray(Image.open(os.path.join(IMG_ROOT, f"{img_id}.png")).convert("RGB"), dtype=np.float32)/255.0
    gts.append(gt)
gts = np.stack(gts)

os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)
t0 = time.time()
with torch.no_grad():
    z = torch.randn(20, 3, 256, 256, device=device)
    yc = torch.tensor([int(s["calligrapher_id"]) for s in rows], device=device)
    ych = torch.tensor([int(s.get("glyph_id", s["character_id"])) for s in rows], device=device)
    mk = dict(y_callig=yc, y_char=ych, cfg_scale=4.0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = diffusion.p_sample_loop(model, z.shape, z, model_kwargs=mk, progress=False, device=device)
    out = out.float().clamp(-1,1)
    px = (out.permute(0,2,3,1).cpu().numpy()+1)/2.0

print(f"sample done in {time.time()-t0:.1f}s", flush=True)

# 拼图: 上 GT 下 pred, 4 列
canvas = np.zeros((2*256, 4*256, 3), dtype=np.uint8)
for i in range(4):
    canvas[:256, i*256:(i+1)*256] = (gts[i]*255).astype(np.uint8)
    canvas[256:, i*256:(i+1)*256] = (px[i]*255).clip(0,255).astype(np.uint8)
Image.fromarray(canvas).save(os.path.join(OUT, "eval_grid.png"))
# 存前 8 张单独 pred
for i in range(min(8, 20)):
    Image.fromarray((px[i]*255).clip(0,255).astype(np.uint8)).save(os.path.join(OUT, f"pred_{i}.png"))
    Image.fromarray((gts[i]*255).astype(np.uint8)).save(os.path.join(OUT, f"gt_{i}.png"))

# MSE/SSIM
mse_sum = ssim_sum = 0.0
for i in range(20):
    p, g = px[i], gts[i]
    mse_sum += float(np.mean((p-g)**2))
    gp, gg = p.mean(), g.mean()
    vp, vg = p.var(), g.var()
    cov = np.mean((p-gp)*(g-gg))
    c1, c2 = 0.01**2, 0.03**2
    ssim_sum += float(((2*gp*gg+c1)*(2*cov+c2))/((gp**2+gg**2+c1)*(vp+vg+c2)+1e-8))
print(f"MSE={mse_sum/20:.4f} SSIM={ssim_sum/20:.4f}", flush=True)
print("OUT:", OUT, flush=True)
'''

b64 = subprocess.run(["python","-c",
    f"import base64; print(base64.b64encode(open(r'{__file__}',encoding='utf-8').read().split('EVAL_SCRIPT = r',1)[1].split(chr(39)+chr(39),1)[0].encode()).decode())"],
    capture_output=True, text=True)
# 简单方法: 写临时文件
import tempfile
with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8") as f:
    f.write(EVAL_SCRIPT)
    local_script = f.name

# scp + run
r = subprocess.run(["scp","-o","ConnectTimeout=25","-P",PORT,
    local_script, "root@10.176.54.17:/tmp/_px_eval.py"],
    capture_output=True, timeout=60)
print("scp:", r.returncode==0)

# 运行 eval
r2 = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,
    "cd /root/Workspace/xy/DiT && /opt/conda/bin/python /tmp/_px_eval.py 2>&1 | tail -20"],
    capture_output=True, text=True, timeout=300, encoding="utf-8", errors="replace")
print("eval:", r2.stdout if r2.returncode==0 else f"rc={r2.returncode}\n{r2.stderr[:300]}")