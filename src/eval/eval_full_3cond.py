import os, sys, torch
os.environ["XFORMERS_DISABLED"] = "1"
import torch.nn as nn
from torchvision.utils import save_image
from src.model import DiT_3Cond_models
from src.model import inject_lora
from src.utils import MCCDDataset
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
from src.loss import create_diffusion

def main(ckpt_path, n=8, t=150, out="eval_full_3cond.png", start=0, csv_path="test.csv"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] device={device}, ckpt={ckpt_path}, n={n}, t={t}, start={start}, csv={csv_path}")

    # Load GT samples from the EVAL split (test.csv by default, NOT train).
    # csv paths are "images/...", "canny/...", "skeleton/..." (no prefix),
    # so root_dir must be "dataset" (remote layout: dataset/images, dataset/canny, dataset/skeleton).
    ds = MCCDDataset(csv_path, "dataset", image_size=256)
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

    # Model: 3Cond-XL/2 (matches the full-training config), LoRA r=32
    model = DiT_3Cond_models["DiT-3Cond-XL/2"](
        input_size=32, num_calligraphers=2021, num_scripts=12, num_characters=7765,
        use_checkpoint=False
    ).to(device)
    model = inject_lora(model, r=32, lora_alpha=32)

    # Load ckpt. Prefer "ema" (full body + 3Cond heads, already LoRA-applied),
    # fall back to "model" (LoRA-only delta + new embedders).
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("ema")
    src = "ema"
    if sd is None:
        sd = ckpt.get("model", ckpt)
        src = "model"
    missing, unexpected = model.load_state_dict(sd, strict=False)
    finite = all(torch.isfinite(v).all() for v in model.state_dict().values())
    print(f"[eval] loaded ckpt from '{src}'. missing={len(missing)} unexpected={len(unexpected)} all_finite={finite}")
    if not finite:
        print("[eval] WARNING: checkpoint contains non-finite (NaN/Inf) weights -> output will be noise.")
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
