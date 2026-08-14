import os, sys, csv, math
sys.stdout.reconfigure(encoding='utf-8')
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse

from models import DiT_3Cond_XL_2
from lora import inject_lora
from diffusers.models import AutoencoderKL
from diffusion import create_diffusion

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- SSIM ----------
def gaussian_window(window_size=11, sigma=1.5):
    g = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g

def ssim(x, y, data_range=1.0, window_size=11):
    # x,y: (1,1,H,W) or (1,3,H,W) tensors in [0,1]
    if x.shape[1] == 3:
        return float(np.mean([ssim(x[:, i:i+1], y[:, i:i+1], data_range, window_size) for i in range(3)]))
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    win = gaussian_window(window_size, 1.5).to(x.device).reshape(1, 1, window_size, 1) @ \
          gaussian_window(window_size, 1.5).to(x.device).reshape(1, 1, 1, window_size)
    win = win.repeat(1, 1, 1, 1)

    mu_x = F.conv2d(x, win, padding=window_size // 2)
    mu_y = F.conv2d(y, win, padding=window_size // 2)
    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, win, padding=window_size // 2) - mu_x2
    sigma_y2 = F.conv2d(y * y, win, padding=window_size // 2) - mu_y2
    sigma_xy = F.conv2d(x * y, win, padding=window_size // 2) - mu_xy
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return float(ssim_map.mean().item())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default="lora_only_0027500.pt")
    ap.add_argument("--csv", default="test_1k/test_1k.csv")
    ap.add_argument("--timestep", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default="eval_quant_result.csv")
    args = ap.parse_args()

    # autoencoder
    ae = AutoencoderKL.from_pretrained("pretrained_models/sd-vae-ft-ema").to(DEVICE).eval()
    diffusion = create_diffusion(timestep_respacing="")
    vae_scale = 0.18215

    # model
    model = DiT_3Cond_XL_2(num_calligraphers=2021, num_scripts=12, num_characters=7765)
    inject_lora(model, r=32, lora_alpha=32)
    ckpt = torch.load(args.lora, map_location="cpu", weights_only=False)
    lora_sd = ckpt["lora"]
    missing, unexpected = model.load_state_dict(lora_sd, strict=False)
    print(f"loaded lora: {len(lora_sd)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
    model = model.to(DEVICE).eval()

    # read csv
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    print(f"eval samples: {len(rows)}")

    to_t = transforms.ToTensor()
    results = []
    mse_sum = 0.0
    ssim_sum = 0.0
    n = 0

    # batch process
    for i in range(0, len(rows), args.batch):
        batch = rows[i:i + args.batch]
        imgs = []
        yc, ys, yh = [], [], []
        for r in batch:
            img = Image.open(r["image_path"]).convert("RGB").resize((256, 256))
            imgs.append(to_t(img))
            yc.append(int(r["calligrapher_id"]))
            ys.append(int(r["script_id"]))
            yh.append(int(r["character_id"]))
        x = torch.stack(imgs, 0).to(DEVICE) * 2 - 1  # [-1,1]

        with torch.no_grad():
            lat = ae.encode(x).latent_dist.sample() * vae_scale
            t = torch.full((x.shape[0],), args.timestep, device=DEVICE, dtype=torch.long)
            model_kwargs = {"y_callig": torch.tensor(yc, device=DEVICE),
                            "y_script": torch.tensor(ys, device=DEVICE),
                            "y_char": torch.tensor(yh, device=DEVICE)}
            out = diffusion.training_losses(model, lat, t, model_kwargs)
            x_start = out["pred_xstart"]
            pred = ae.decode(x_start / vae_scale).sample  # [-1,1]

        # GT in [-1,1] for mse; [0,1] for ssim
        gt01 = (x + 1) / 2
        pred01 = (pred + 1) / 2
        pred01 = pred01.clamp(0, 1)

        for j in range(x.shape[0]):
            mse = F.mse_loss(pred01[j:j+1], gt01[j:j+1]).item()
            ss = ssim(pred01[j:j+1], gt01[j:j+1], data_range=1.0)
            mse_sum += mse
            ssim_sum += ss
            n += 1
            results.append((batch[j]["image_path"], mse, ss))

    print(f"=== RESULTS (timestep={args.timestep}) ===")
    print(f"n={n}  MSE={mse_sum/n:.6f}  SSIM={ssim_sum/n:.4f}")

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "mse", "ssim"])
        w.writerows(results)
    print(f"per-sample saved -> {args.out}")

if __name__ == "__main__":
    main()
