"""
2Cond overfit 推理对比：加载 overfit_500 的 LoRA checkpoint，对 overfit_500.csv 前 N 张
做固定-noise 单步 pred_xstart 解码，拼 [GT | 生成] 输出 PNG。

用法（远程，GPU）:
  /opt/conda/bin/python sample_overfit_2cond.py \
      --ckpt results/overfit_500/checkpoints/0010000.pt \
      --csv  overfit_500.csv --data_dir dataset \
      --out  overfit_2cond_comparison.png --n 8 --t 150
"""
import argparse
import torch
import torch.nn as nn
from torchvision.utils import save_image
from models import DiT_2Cond_models
from lora import inject_lora
from dataset import MCCDDataset
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
from diffusion import create_diffusion

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", default="overfit_500.csv")
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--pretrained", default="pretrained_models/DiT-XL-2-256x256.pt")
    ap.add_argument("--out", default="overfit_2cond_comparison.png")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--t", type=int, default=150)
    ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # 1. data
    ds = MCCDDataset(args.csv, args.data_dir, image_size=args.image_size,
                     load_maps=False, is_train=False)
    loader = DataLoader(ds, batch_size=args.n, shuffle=False)
    batch = next(iter(loader))
    x = batch['image'].to(device)                 # GT, [-1,1]
    y_callig = batch['y_callig'].to(device)
    y_char = batch['y_char'].to(device)
    n = x.shape[0]

    # 2. model
    model = DiT_2Cond_models["DiT-2Cond-XL/2"](
        input_size=args.image_size // 8,
        num_calligraphers=2021, num_characters=7765).to(device)
    model = inject_lora(model, r=args.r, lora_alpha=args.r)
    # load pretrained body
    from download import find_model
    sd = find_model(args.pretrained)
    sd = {k: v for k, v in sd.items()
          if not k.startswith(("y_embedder", "y_callig", "y_script", "y_char", "cond_fusion"))}
    model.load_state_dict(sd, strict=False)
    # load lora ckpt
    ckpt = torch.load(args.ckpt, map_location="cpu")
    lora_sd = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(lora_sd, strict=False)
    print(f"LoRA ckpt loaded. missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    # 3. vae
    vae = AutoencoderKL.from_pretrained("pretrained_models/sd-vae-ft-ema").to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    with torch.no_grad():
        x_latent = vae.encode(x).latent_dist.sample().mul_(0.18215)

    diffusion = create_diffusion(timestep_respacing="")
    t = torch.tensor([args.t] * n, device=device, dtype=torch.long)
    fixed_noise = torch.randn_like(x_latent)

    with torch.no_grad():
        loss_dict = diffusion.training_losses(model, x_latent, t,
                                              dict(y_callig=y_callig, y_char=y_char),
                                              noise=fixed_noise)
        pred_latent = loss_dict["pred_xstart"]
        decoded = vae.decode(pred_latent / 0.18215).sample  # [-1,1]

    # 4. 拼 [GT | 生成]
    grid = torch.cat([x, decoded], dim=0)
    save_image(grid, args.out, nrow=n, normalize=True, value_range=(-1, 1))
    print(f"Saved {args.out}  (top=GT, bottom=pred, n={n})")

if __name__ == "__main__":
    main()
