# -*- coding: utf-8 -*-
"""10k test 终评：加载指定 ckpt，在 final_test.csv (10000) 上算 MSE/SSIM + 落盘前 20 张对比图。

用法:
  python eval_test.py --ckpt <path.pt> --model DiT-3Cond-S/2 --use-lora 0 \
      --pretrained null --num-calligraphers 1873 --out test_eval_S
  或 XL-LoRA:
  python eval_test.py --ckpt <path.pt> --model DiT-3Cond-XL/2 --use-lora 1 \
      --lora-r 32 --lora-target all --pretrained pretrained_models/DiT-XL-2-256x256.pt \
      --num-calligraphers 1873 --out test_eval_XL_r32
"""
import os, sys, json, argparse, datetime
sys.stdout.reconfigure(encoding="utf-8")
import torch

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)

from models import DiT_3Cond_models
from lora import inject_lora
from dataset import MCCDDataset
from diffusers.models import AutoencoderKL
from diffusion import create_diffusion
from eval_auto import prepare_eval_cache, eval_in_memory


def _str2bool(v):
    return str(v).lower() in ("1", "true", "yes", "on")


def build_model(args, device):
    model = DiT_3Cond_models[args.model](
        input_size=32,
        num_calligraphers=args.num_calligraphers,
        num_scripts=args.num_scripts,
        num_characters=args.num_characters,
        use_checkpoint=False,
    )
    # pretrained body (filter conditioning head)
    if args.pretrained and args.pretrained.lower() != "none" and os.path.exists(args.pretrained):
        pre = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        if "model" in pre:
            pre = pre["model"]
        pre = {k: v for k, v in pre.items()
               if not k.startswith(("y_embedder", "y_callig", "y_script", "y_char", "cond_fusion"))}
        missing, _ = model.load_state_dict(pre, strict=False)
        print(f"[eval_test] pretrained body loaded, missing(cond head only)={len(missing)}")
    if args.use_lora:
        inject_lora(model, r=args.lora_r, lora_alpha=args.lora_alpha or args.lora_r,
                    target=args.lora_target)
    # load delta / full state dict
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    delta = ckpt.get("delta", ckpt.get("model", ckpt))
    missing, unexpected = model.load_state_dict(delta, strict=False)
    print(f"[eval_test] loaded ckpt delta: missing={len(missing)} unexpected={len(unexpected)}")
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="DiT-3Cond-S/2")
    ap.add_argument("--use-lora", type=_str2bool, default=False)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=None)
    ap.add_argument("--lora-target", default="all")
    ap.add_argument("--pretrained", default="null")
    ap.add_argument("--num-calligraphers", type=int, default=1873)
    ap.add_argument("--num-scripts", type=int, default=12)
    ap.add_argument("--num-characters", type=int, default=7765)
    ap.add_argument("--csv", default="final_test.csv")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="test_eval")
    ap.add_argument("--vae-path", default="pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--vis-n", type=int, default=20)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args, device)

    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False
    diffusion = create_diffusion(timestep_respacing="")

    ds = MCCDDataset(args.csv, "", image_size=256, load_canny=False, load_skel=False)
    print(f"[eval_test] preparing cache on {args.csv} (n={args.n}) ...")
    cache = prepare_eval_cache(vae, ds, device, n=args.n)

    vis_dir = f"{args.out}_imgs"
    mse, ssim = eval_in_memory(model, vae, diffusion, device, cache, n=args.n,
                               batch_size=args.batch, vis_out=vis_dir, vis_n=args.vis_n)
    result = {
        "ckpt": args.ckpt,
        "model": args.model,
        "csv": args.csv,
        "n": args.n,
        "mse": mse,
        "ssim": ssim,
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(f"{args.out}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[eval_test] DONE n={args.n} MSE={mse:.5f} SSIM={ssim:.4f} -> {args.out}.json + {vis_dir}/")


if __name__ == "__main__":
    main()
