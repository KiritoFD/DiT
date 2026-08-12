import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
from models import DiT_3Cond_models
from lora import inject_lora
from dataset import MCCDDataset
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
from diffusion import create_diffusion
from losses import SobelCannyLoss, SkeletonLoss

def run_overfit_and_sample():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load single sample
    ds = MCCDDataset(r"example_100\example.csv", r"g:\GitHub\DiT\example_100", image_size=256)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    batch = next(iter(loader))
    
    x = batch['image'].to(device)
    canny_gt = batch['canny'].to(device)
    skel_gt = batch['skeleton'].to(device)
    y_callig = batch['y_callig'].to(device)
    y_script = batch['y_script'].to(device)
    y_char = batch['y_char'].to(device)
    
    # 2. Init model
    model = DiT_3Cond_models["DiT-3Cond-S/2"](input_size=32, num_calligraphers=2243, num_scripts=12, num_characters=7765).to(device)
    
    # Non-zero init adaLN & final layer
    for block in model.blocks:
        nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.02)
        nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.02)
    nn.init.normal_(model.final_layer.linear.weight, std=0.02)
    
    model = inject_lora(model, r=32, lora_alpha=32)
    
    # Trainable parameters
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if 'lora_' in name or 'y_' in name or 'cond_fusion' in name or 'adaLN' in name:
            p.requires_grad = True
            
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=3e-3)
    
    # 3. VAE
    vae_path = r"pretrained_models/sd-vae-ft-ema"
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
        
    with torch.no_grad():
        x_latent = vae.encode(x).latent_dist.sample().mul_(0.18215)
        
    canny_loss_fn = SobelCannyLoss().to(device)
    skel_loss_fn = SkeletonLoss(lambda_bg=1.0).to(device)
    diffusion = create_diffusion(timestep_respacing="")
    
    # Fix noise & t for fast overfit visualization
    t = torch.tensor([150], device=device, dtype=torch.long)
    fixed_noise = torch.randn_like(x_latent)
    model_kwargs = dict(y_callig=y_callig, y_script=y_script, y_char=y_char)
    
    print("Training overfit for 150 steps...")
    model.train()
    for step in range(1, 151):
        loss_dict = diffusion.training_losses(model, x_latent, t, model_kwargs, noise=fixed_noise)
        loss_diff = loss_dict["loss"].mean()
        
        pred_xstart_latent = loss_dict.get("pred_xstart", None)
        loss_canny = torch.tensor(0.0, device=device)
        loss_skel = torch.tensor(0.0, device=device)
        
        if pred_xstart_latent is not None:
            x0_pred = vae.decode(pred_xstart_latent / 0.18215).sample
            loss_canny = canny_loss_fn(x0_pred, canny_gt)
            loss_skel = skel_loss_fn(x0_pred, skel_gt)
            
        loss = loss_diff + 0.1 * loss_canny + 0.1 * loss_skel
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"Step {step}: Total Loss = {loss.item():.4f}, Diff = {loss_diff.item():.4f}")

    # Decode and save comparison
    model.eval()
    with torch.no_grad():
        loss_dict = diffusion.training_losses(model, x_latent, t, model_kwargs, noise=fixed_noise)
        pred_latent = loss_dict["pred_xstart"]
        decoded_x0 = vae.decode(pred_latent / 0.18215).sample
        
    # Format comparison grid: [GT Image, GT Canny, GT Skeleton, Decoded Prediction]
    canny_3c = canny_gt.repeat(1, 3, 1, 1) * 2 - 1
    skel_3c = skel_gt.repeat(1, 3, 1, 1) * 2 - 1
    
    grid = torch.cat([x, canny_3c, skel_3c, decoded_x0], dim=0)
    save_path = "overfit_comparison.png"
    save_image(grid, save_path, nrow=4, normalize=True, value_range=(-1, 1))
    print(f"Saved comparison image to {save_path}")

if __name__ == "__main__":
    run_overfit_and_sample()
