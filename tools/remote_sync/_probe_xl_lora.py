# -*- coding: utf-8 -*-
"""DiT-2Cond-XL/2 + LoRA + 结构 loss batch 显存 probe。

与 train.py 完全一致：XL body + LoRA + factorized_add 条件，diff loss +
全量 pixel canny + skel loss（VAE decode pred_xstart）。输出各 batch 峰值显存
直到 OOM，用于决定「XL+LoRA+结构loss」能开多大 batch。
"""
import os, sys, json, glob
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["XFORMERS_DISABLED"] = "1"
import torch
import torch.nn.functional as F
import numpy as np
import gc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from models import DiT_2Cond_models
from lora import inject_lora
from download import find_model
from diffusers.models import AutoencoderKL
from losses import SobelCannyLoss, SkeletonLoss
from diffusion import create_diffusion

BATCHES = [4, 8, 12, 16]
RANK = 32


def main():
    torch.manual_seed(0)
    device = "cuda"
    model = DiT_2Cond_models["DiT-2Cond-XL/2"](
        input_size=32, num_calligraphers=1011, num_characters=35130,
        use_checkpoint=False, condition_fusion="factorized_add",
        callig_embed_dim=128, char_embed_dim=192,
        cond_drop_all_prob=0.10, cond_drop_one_prob=0.30)
    total = sum(p.numel() for p in model.parameters())
    print(f"[probe-xl-struct] DiT-2Cond-XL/2 total={total/1e6:.1f}M")

    pre = find_model("pretrained_models/DiT-XL-2-256x256.pt")
    pre = {k: v for k, v in pre.items()
           if not k.startswith(("y_embedder", "cond_fusion", "callig_proj",
                                "char_proj", "y_callig", "y_char"))}
    missing, unexpected = model.load_state_dict(pre, strict=False)
    inject_lora(model, r=RANK, lora_alpha=RANK, target="all")
    model = model.to(device).train()

    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[probe-xl-struct] trainable={tr/1e6:.1f}M")

    # VAE + 结构 loss + diffusion
    vae = AutoencoderKL.from_pretrained("pretrained_models/sd-vae-ft-ema").to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False
    canny_loss = SobelCannyLoss().to(device)
    skel_loss = SkeletonLoss(lambda_bg=1.0).to(device)
    diffusion = create_diffusion("")

    # mock GT 图（256x256）、canny、skel
    results = {}
    for bs in BATCHES:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        x_lat = torch.randn(bs, 4, 32, 32, device=device)
        gt_img = torch.rand(bs, 3, 256, 256, device=device) * 2 - 1
        canny_gt = (torch.rand(bs, 1, 256, 256, device=device) > 0.5).float()
        skel_gt = (torch.rand(bs, 1, 256, 256, device=device) > 0.5).float()
        t = torch.randint(0, 1000, (bs,), device=device)
        yc = torch.randint(0, 1011, (bs,), device=device)
        yg = torch.randint(0, 35130, (bs,), device=device)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss_dict = diffusion.training_losses(
                    model, x_lat, t, dict(y_callig=yc, y_char=yg))
                loss_diff = loss_dict["loss"].mean()
                pred_xstart = loss_dict["pred_xstart"]
            # 结构 loss 需要 decode pred_xstart（fp32 + grad checkpoint 同 train.py）
            def _decode(z):
                return vae.decode(z).sample
            with torch.autocast("cuda", dtype=torch.float32):
                x0_pred = torch.utils.checkpoint.checkpoint(
                    _decode, pred_xstart.float() / 0.18215, use_reentrant=False)
            lc = canny_loss(x0_pred, canny_gt)
            ls = skel_loss(x0_pred, skel_gt)
            loss = loss_diff + 1.0 * lc + 1.0 * ls
            loss.backward()
            peak = torch.cuda.max_memory_allocated() / 1024**3
            results[f"b{bs}"] = round(peak, 2)
            print(f"[probe-xl-struct] batch {bs}: peak {peak:.2f}G OK")
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    p.grad = None
        except torch.cuda.OutOfMemoryError:
            results[f"b{bs}"] = "OOM"
            print(f"[probe-xl-struct] batch {bs}: OOM -> 上限是上一个 OK")
            break
    print("[probe-xl-struct] done")


if __name__ == "__main__":
    main()
