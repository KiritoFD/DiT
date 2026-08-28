"""
Sampling & Inference Script for 3-Condition Guided DiT (DiT_3Cond)
Generates 256x256 calligraphy images given Calligrapher, Script, and Character conditions.
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import torch
from PIL import Image

from models import DiT_3Cond_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def sample(
    model_name="DiT-3Cond-S/2",
    ckpt_path=None,
    calligrapher="王羲之",
    script="行",
    character="永",
    cfg_scale=4.0,
    num_sampling_steps=50,
    output_path="sample_output.png"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sampling Calligraphy Image on device: {device}")
    print(f"Condition Input -> Calligrapher: '{calligrapher}', Script: '{script}', Character: '{character}'")

    # Load Vocabularies
    with open("labels/calligrapher_to_id.json", "r", encoding="utf-8") as f:
        callig_dict = json.load(f)
    with open("labels/script_to_id.json", "r", encoding="utf-8") as f:
        script_dict = json.load(f)
    with open("labels/character_to_id.json", "r", encoding="utf-8") as f:
        char_dict = json.load(f)

    # Resolve Condition IDs
    y_callig_id = callig_dict.get(calligrapher, 0)
    y_script_id = script_dict.get(script, 0)
    y_char_id = char_dict.get(character, 0)

    print(f"Resolved IDs -> Callig ID: {y_callig_id}, Script ID: {y_script_id}, Char ID: {y_char_id}")

    # Instantiate Model
    model = DiT_3Cond_models[model_name](
        num_calligraphers=len(callig_dict),
        num_scripts=len(script_dict),
        num_characters=len(char_dict)
    ).to(device)

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading checkpoint from: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        ckpt_model = state_dict["model"] if "model" in state_dict else state_dict
        # train.py checkpoints store only trainable params (LoRA + condition embedders).
        # If LoRA params are present, inject LoRA layers first so the keys can be applied.
        if any("lora_" in k for k in ckpt_model.keys()):
            from lora import inject_lora
            model = inject_lora(model)
        missing, unexpected = model.load_state_dict(ckpt_model, strict=False)
        if missing:
            print(f"Warning: {len(missing)} missing keys (using default init): {list(missing)[:5]}...")
        if unexpected:
            print(f"Info: {len(unexpected)} unexpected keys ignored: {list(unexpected)[:5]}...")
    else:
        print("Warning: No checkpoint path specified or found. Using initialized weights for sampling test.")

    model.eval()

    # Load VAE & Diffusion
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    vae.eval()
    diffusion = create_diffusion(str(num_sampling_steps))

    # Condition Tensors (Batch size = 1)
    y_callig_t = torch.tensor([y_callig_id], device=device)
    y_script_t = torch.tensor([y_script_id], device=device)
    y_char_t = torch.tensor([y_char_id], device=device)

    # Initial random noise (1, 4, 32, 32)
    z_t = torch.randn(1, 4, 32, 32, device=device)

    print(f"Starting DDIM Sampling ({num_sampling_steps} steps, CFG Scale={cfg_scale})...")

    # Model kwargs for CFG sampling
    model_kwargs = dict(
        y_callig=y_callig_t,
        y_script=y_script_t,
        y_char=y_char_t,
        cfg_scale=cfg_scale
    )

    # Sampling Loop
    samples = diffusion.ddim_sample_loop(
        model.forward_with_cfg,
        z_t.shape,
        z_t,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        device=device
    )

    # Decode Latents to Image
    with torch.no_grad():
        samples_x0 = vae.decode(samples / 0.18215).sample # (1, 3, 256, 256) in [-1, 1]

    # Convert to numpy uint8 image
    img_np = ((samples_x0[0].permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Save output image
    cv2.imencode('.png', img_bgr)[1].tofile(output_path)
    print(f"Sample generated and saved to: '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample 3-Condition Calligraphy DiT")
    parser.add_argument("--model", type=str, default="DiT-3Cond-S/2")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--calligrapher", type=str, default="王羲之")
    parser.add_argument("--script", type=str, default="行")
    parser.add_argument("--character", type=str, default="永")
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output", type=str, default="sample_output.png")
    args = parser.parse_args()

    sample(
        model_name=args.model,
        ckpt_path=args.ckpt,
        calligrapher=args.calligrapher,
        script=args.script,
        character=args.character,
        cfg_scale=args.cfg,
        num_sampling_steps=args.steps,
        output_path=args.output
    )
