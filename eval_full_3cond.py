import os, sys, torch
os.environ["XFORMERS_DISABLED"] = "1"
import torch.nn as nn
from torchvision.utils import save_image
from models import DiT_3Cond_models
from lora import inject_lora
from dataset import MCCDDataset
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
from diffusion import create_diffusion

def main(ckpt_path, n=8, t=150, out="eval_full_3cond.png", start=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] device={device}, ckpt={ckpt_path}, n={n}, t={t}, start={start}")

    # Load a few GT samples from full train set (3cond: callig+script+char)
    ds = MCCDDataset("train.csv", "dataset", image_size=256)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    xs, ys, canny_s, skel_s = [], [], [], []
    for i, b in enumerate(loader):
        if i < start:
            continue
        xs.append(b['image']); canny_s.append(b['canny']); skel_s.append(b['skeleton'])
        ys.append((b['y_callig'], b['y_script'], b['y_char']))
        if len(xs) >= n:
            break
    x = torch.cat(xs).to(device)
    y_callig = torch.cat([y[0] for y in ys]).to(device)
    y_script = torch.cat([y[1] for y in ys]).to(device)
    y_char = torch.cat([y[2] for y in ys]).to(device)

    # Model: 3Cond-S, LoRA r=32
    model = DiT_3Cond_models["DiT-3Cond-S/2"](
        input_size=32, num_calligraphers=2021, num_scripts=12, num_characters=7765,
        use_checkpoint=False
    ).to(device)
    model = inject_lora(model, r=32, lora_alpha=32)

    # Load ckpt (model key = extract_lora_and_new_embedders output)
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[eval] loaded ckpt. missing={len(missing)} (body weights from random init), unexpected={len(unexpected)}")
    model.eval()

    vae = AutoencoderKL.from_pretrained("pretrained_models/sd-vae-ft-ema").to(device)
    vae.eval()
    with torch.no_grad():
        x_latent = vae.encode(x).latent_dist.sample().mul_(0.18215)
    diffusion = create_diffusion(timestep_respacing="")

    model_kwargs = dict(y_callig=y_callig, y_script=y_script, y_char=y_char)
    fixed_noise = torch.randn_like(x_latent)
    with torch.no_grad():
        ld = diffusion.training_losses(model, x_latent, torch.tensor([t]*n, device=device, dtype=torch.long),
                                       model_kwargs, noise=fixed_noise)
        pred = ld["pred_xstart"]
        decoded = vae.decode(pred / 0.18215).sample

    # grid rows: [GT, Canny, Skel, Pred] repeated
    grids = []
    for i in range(n):
        c3 = canny_s[i].repeat(1,3,1,1).to(device) * 2 - 1
        s3 = skel_s[i].repeat(1,3,1,1).to(device) * 2 - 1
        grids.append(torch.cat([x[i:i+1], c3, s3, decoded[i:i+1]], dim=0))
    grid = torch.cat(grids, dim=0)
    save_image(grid, out, nrow=4, normalize=True, value_range=(-1, 1))
    print(f"[eval] saved {out}  (grid {n}x4)")

if __name__ == "__main__":
    main(ckpt_path=sys.argv[1], out=sys.argv[2] if len(sys.argv) > 2 else "eval_full_3cond.png",
         n=int(sys.argv[3]) if len(sys.argv) > 3 else 8,
         t=int(sys.argv[4]) if len(sys.argv) > 4 else 150,
         start=int(sys.argv[5]) if len(sys.argv) > 5 else 0)
