# -*- coding: utf-8 -*-
"""FontDiffuser-Latent training on top10_style23.

Whole model in the project latent space (sd-vae-ft-ema f8, 4x32x32, x0.18215):
  - target/style/content latents come from precomputed shards (no pixel IO);
  - DDPM (official betas) over latents; UNet in/out = 4ch @ 32x32;
  - losses: noise MSE + x0-latent MSE (replaces the pixel VGG perceptual term,
    same coefficient 0.01 — no pixel tensors allowed) + DCN offset norm (0.5);
  - classifier-free drop: content/style latents replaced by the blank-white
    latent (latent analogue of torch.ones_like images), drop_prob 0.1.

Run from baseline/FontDiffuser:
  accelerate launch --num_processes=1 ../adapter/fontdiffuser/train_latent.py \
    --train_batch_size 8 --max_train_steps 150000 ...
"""
import logging
import os
import sys

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from configs.fontdiffuser import get_parser  # noqa: E402
from build_latent import (FontDiffuserModelLatent, build_content_encoder_latent,  # noqa: E402
                          build_style_encoder_latent, build_unet_latent)
from latent_common import blank_latent  # noqa: E402
from latent_dataset import CollateFN, FontLatentDataset  # noqa: E402

logger = get_logger(__name__)


def get_args():
    parser = get_parser()
    parser.add_argument("--num_refs", type=int, default=1)
    parser.add_argument("--compile", action="store_true", default=False,
                        help="torch.compile the unet+encoders (memory/speed win)")
    args = parser.parse_args()
    args.resolution = 32                     # latent grid
    args.style_image_size = (32, 32)
    args.content_image_size = (32, 32)
    return args


def main():
    args = get_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=f"{args.output_dir}/logs")
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(filename=f"{args.output_dir}/fontdiffuser_latent.log",
                        level=logging.INFO)
    if args.seed is not None:
        set_seed(args.seed)

    unet = build_unet_latent(args)
    style_encoder = build_style_encoder_latent(args)
    content_encoder = build_content_encoder_latent(args)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
        beta_schedule=args.beta_scheduler, trained_betas=None,
        variance_type="fixed_small", clip_sample=True)

    model = FontDiffuserModelLatent(unet=unet, style_encoder=style_encoder,
                                    content_encoder=content_encoder)
    if args.compile:
        unet_c = torch.compile(unet)
        se_c = torch.compile(style_encoder)
        ce_c = torch.compile(content_encoder)
        model = FontDiffuserModelLatent(unet=unet_c, style_encoder=se_c,
                                        content_encoder=ce_c)

    dataset = FontLatentDataset(device="cpu")
    loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=args.train_batch_size,
        collate_fn=CollateFN(), num_workers=8, pin_memory=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  betas=(args.adam_beta1, args.adam_beta2),
                                  weight_decay=args.adam_weight_decay,
                                  eps=args.adam_epsilon)
    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps)

    model, optimizer, loader, lr_scheduler = accelerator.prepare(
        model, optimizer, loader, lr_scheduler)

    drop_latent = blank_latent(str(accelerator.device))       # (1,4,32,32)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.experience_name)

    global_step = 0
    progress = tqdm(range(args.max_train_steps),
                    disable=not accelerator.is_local_main_process)
    progress.set_description("Steps")
    cycl = iter(loader)

    while global_step < args.max_train_steps:
        try:
            samples = next(cycl)
        except StopIteration:
            cycl = iter(loader)
            samples = next(cycl)

        with accelerator.accumulate(model):
            target = samples["target_latent"].to(accelerator.device, non_blocking=True)
            content = samples["content_latent"].to(accelerator.device, non_blocking=True)
            style = samples["style_latent"].to(accelerator.device, non_blocking=True)

            noise = torch.randn_like(target)
            bsz = target.shape[0]
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps,
                                      (bsz,), device=target.device).long()
            x_t = noise_scheduler.add_noise(target, noise, timesteps)

            # classifier-free drop (latent-space blank)
            mask = torch.bernoulli(torch.zeros(bsz, device=target.device) + args.drop_prob).bool()
            content = torch.where(mask[:, None, None, None], drop_latent, content)
            style = torch.where(mask[:, None, None, None], drop_latent, style)

            noise_pred, offset_out_sum = model(
                x_t=x_t, timesteps=timesteps, style_images=style,
                content_images=content,
                content_encoder_downsample_size=args.content_encoder_downsample_size)

            diff_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            ac = noise_scheduler.alphas_cumprod.to(accelerator.device)
            alpha_prod = ac[timesteps].view(-1, 1, 1, 1)
            beta_prod = 1 - alpha_prod
            x0_pred = (x_t - beta_prod.sqrt() * noise_pred) / alpha_prod.sqrt()
            x0_loss = F.mse_loss(x0_pred.float().clamp(-6, 6), target.float())
            offset_loss = offset_out_sum / 2

            loss = diff_loss + args.perceptual_coefficient * x0_loss + \
                args.offset_coefficient * offset_loss

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress.update(1)
                global_step += 1
                accelerator.log({"train_loss": loss.item()}, step=global_step)
                if global_step % args.log_interval == 0:
                    lr = lr_scheduler.get_last_lr()[0]
                    progress.set_postfix(loss=loss.item(), diff=diff_loss.item(),
                                         x0=x0_loss.item(), lr=lr)
                if accelerator.is_main_process and \
                        global_step % args.ckpt_interval == 0:
                    sd = f"{args.output_dir}/global_step_{global_step}"
                    os.makedirs(sd, exist_ok=True)
                    unwrapped = accelerator.unwrap_model(model)
                    # torch.compile wraps modules in OptimizedModule — strip it
                    strip = lambda m: getattr(m, "_orig_mod", m)  # noqa: E731
                    torch.save(strip(unwrapped.unet).state_dict(), f"{sd}/unet.pth")
                    torch.save(strip(unwrapped.style_encoder).state_dict(),
                               f"{sd}/style_encoder.pth")
                    torch.save(strip(unwrapped.content_encoder).state_dict(),
                               f"{sd}/content_encoder.pth")

    if accelerator.is_main_process:
        sd = f"{args.output_dir}/final"
        os.makedirs(sd, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        strip = lambda m: getattr(m, "_orig_mod", m)  # noqa: E731
        torch.save(strip(unwrapped.unet).state_dict(), f"{sd}/unet.pth")
        torch.save(strip(unwrapped.style_encoder).state_dict(), f"{sd}/style_encoder.pth")
        torch.save(strip(unwrapped.content_encoder).state_dict(), f"{sd}/content_encoder.pth")
    print("TRAINING_DONE")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
