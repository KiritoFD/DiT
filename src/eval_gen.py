# -*- coding: utf-8 -*-
"""自由采样 eval：与推理完全一致（纯噪声 -> DDIM 去噪链），与 GT 图比 MSE/SSIM。

用法:
  /opt/conda/bin/python eval_gen.py --ckpt <ckpt.pt> --model DiT-3Cond-S/2 \
      --use-lora 0 --pretrained null --csv final_eval.csv --n 50 --out gen_eval_S
"""
import os, sys, json, argparse, datetime
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("XFORMERS_DISABLED", "1")
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from models import DiT_2Cond_models, DiT_3Cond_models
from lora import inject_lora
from dataset import MCCDDataset
from diffusers.models import AutoencoderKL
from diffusion import create_diffusion
from eval_auto import _gaussian_window, _ssim
from latent_condition_probe import LatentConditionProbe


def _s2b(v):
    return str(v).lower() in ("1", "true", "yes", "on")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--use-lora", type=_s2b, default=False)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-target", default="all")
    ap.add_argument("--pretrained", default="null")
    ap.add_argument("--num-calligraphers", type=int, default=None)
    ap.add_argument("--num-scripts", type=int, default=None)
    ap.add_argument("--num-characters", type=int, default=None)
    ap.add_argument("--condition-fusion", choices=["legacy", "factorized_add", "xl_highdim"], default=None)
    ap.add_argument("--callig-embed-dim", type=int, default=None)
    ap.add_argument("--script-embed-dim", type=int, default=None)
    ap.add_argument("--char-embed-dim", type=int, default=None)
    ap.add_argument("--cond-mode", choices=["2cond", "3cond"], default=None)
    ap.add_argument("--use-ema", type=_s2b, default=True,
                    help="Use checkpoint EMA weights when available.")
    ap.add_argument("--csv", default="final_eval.csv")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16, help="DDIM 采样 batch（并行采样加速）")
    ap.add_argument("--steps", type=int, default=50, help="DDIM 采样步数")
    ap.add_argument("--cfg", type=float, default=4.0, help="CFG 强度（1.0=无 CFG）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="gen_eval")
    ap.add_argument("--vae-path", default="pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--vis-n", type=int, default=20)
    ap.add_argument("--condition-probe", default=None,
                    help="Optional LatentConditionProbe checkpoint for condition-adherence metrics.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args")

    def saved(name, fallback):
        if saved_args is None:
            return fallback
        if isinstance(saved_args, dict):
            return saved_args.get(name, fallback)
        return getattr(saved_args, name, fallback)

    model_name = args.model or saved("model", "DiT-3Cond-S/2")
    num_calligraphers = args.num_calligraphers or saved("num_calligraphers", 1873)
    num_scripts = args.num_scripts or saved("num_scripts", 12)
    num_characters = args.num_characters or saved("num_characters", 7765)
    condition_fusion = args.condition_fusion or saved("condition_fusion", "legacy")
    callig_embed_dim = args.callig_embed_dim or saved("callig_embed_dim", None)
    script_embed_dim = args.script_embed_dim or saved("script_embed_dim", None)
    char_embed_dim = args.char_embed_dim or saved("char_embed_dim", None)
    cond_mode = args.cond_mode or saved("cond_mode", "3cond")
    print(f"[eval_gen] free-sampling eval: model={model_name} cond_mode={cond_mode} "
          f"fusion={condition_fusion} cfg={args.cfg} steps={args.steps} n={args.n} csv={args.csv}")

    # ---- model（与训练/推理完全一致的加载）----
    if cond_mode == "2cond":
        model = DiT_2Cond_models[model_name](
            input_size=32, num_calligraphers=num_calligraphers,
            num_characters=num_characters,
            use_checkpoint=False, condition_fusion=condition_fusion,
            callig_embed_dim=callig_embed_dim, char_embed_dim=char_embed_dim,
            cond_drop_all_prob=saved("cond_drop_all_prob", 0.05),
            cond_drop_one_prob=saved("cond_drop_one_prob", 0.0))
    else:
        model = DiT_3Cond_models[model_name](
            input_size=32, num_calligraphers=num_calligraphers,
            num_scripts=num_scripts, num_characters=num_characters,
            use_checkpoint=False, condition_fusion=condition_fusion,
            callig_embed_dim=callig_embed_dim, script_embed_dim=script_embed_dim,
            char_embed_dim=char_embed_dim,
            cond_drop_all_prob=saved("cond_drop_all_prob", 0.05),
            cond_drop_one_prob=saved("cond_drop_one_prob", 0.0))
    if args.pretrained and args.pretrained.lower() != "none" and os.path.exists(args.pretrained):
        pre = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        if "model" in pre:
            pre = pre["model"]
        pre = {k: v for k, v in pre.items()
               if not k.startswith(("y_embedder", "y_callig", "y_script", "y_char", "cond_fusion"))}
        model.load_state_dict(pre, strict=False)
        print(f"[eval_gen] pretrained body: {args.pretrained}")
    if args.use_lora:
        model = inject_lora(model, r=args.lora_r, lora_alpha=args.lora_r, target=args.lora_target)
    delta = ckpt.get("ema") if args.use_ema else None
    source = "ema"
    if delta is None:
        delta = ckpt.get("delta", ckpt.get("model", ckpt))
        source = "delta"
    missing, unexpected = model.load_state_dict(delta, strict=False)
    print(f"[eval_gen] {source} loaded: missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    condition_probe = None
    if args.condition_probe:
        probe_ckpt = torch.load(args.condition_probe, map_location="cpu", weights_only=False)
        probe_args = probe_ckpt.get("args", {})
        condition_probe = LatentConditionProbe(
            num_characters=int(probe_args.get("num_characters", num_characters)),
            num_calligraphers=int(probe_args.get("num_calligraphers", num_calligraphers)),
            num_scripts=int(probe_args.get("num_scripts", num_scripts)),
            width=int(probe_args.get("width", 32)))
        condition_probe.load_state_dict(probe_ckpt["model"], strict=True)
        condition_probe = condition_probe.to(device).eval()
        condition_probe.requires_grad_(False)
        print(f"[eval_gen] condition probe loaded: {args.condition_probe}; "
              f"validation={probe_ckpt.get('metrics', {})}")

    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device).eval()
    diffusion = create_diffusion(timestep_respacing="")
    sampler = create_diffusion(str(args.steps))  # DDIM respace

    # ---- 数据（GT 条件 + GT 图）----
    ds = MCCDDataset(args.csv, "", image_size=256, load_canny=False, load_skel=False)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    conds, gts = [], []
    for i, b in enumerate(loader):
        if len(conds) >= args.n:
            break
        conds.append((b["y_callig"].item(), b["y_script"].item(), b["y_char"].item()))
        gts.append(b["image"])  # (1,3,256,256)
    gts = torch.cat(gts).to(device)
    print(f"[eval_gen] {len(conds)} samples")

    # ---- 自由采样（与推理一致：纯噪声 -> DDIM，batch 并行）----
    win = _gaussian_window(11, 1.5, device)
    mse_sum = ssim_sum = 0.0
    condition_correct = dict(char_top1=0, char_top5=0, callig_top1=0,
                             callig_top5=0, script_top1=0)
    os.makedirs(f"{args.out}_imgs", exist_ok=True)
    torch.manual_seed(args.seed)
    B = args.batch
    with torch.no_grad():
        for i in range(0, len(conds), B):
            j = min(i + B, len(conds))
            z = torch.randn(j - i, 4, 32, 32, device=device)
            yc = torch.tensor([c[0] for c in conds[i:j]], device=device)
            yh = torch.tensor([c[2] for c in conds[i:j]], device=device)
            if cond_mode == "2cond":
                mk = dict(y_callig=yc, y_char=yh, cfg_scale=args.cfg)
            else:
                ys = torch.tensor([c[1] for c in conds[i:j]], device=device)
                mk = dict(y_callig=yc, y_script=ys, y_char=yh, cfg_scale=args.cfg)
            samples = sampler.ddim_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False,
                model_kwargs=mk, progress=False, device=device)
            if condition_probe is not None:
                char_logits, callig_logits, script_logits = condition_probe(samples.float())
                condition_correct["char_top1"] += int(
                    char_logits.argmax(1).eq(yh).sum())
                condition_correct["char_top5"] += int(
                    char_logits.topk(5, dim=1).indices.eq(yh[:, None]).any(1).sum())
                condition_correct["callig_top1"] += int(
                    callig_logits.argmax(1).eq(yc).sum())
                condition_correct["callig_top5"] += int(
                    callig_logits.topk(5, dim=1).indices.eq(yc[:, None]).any(1).sum())
                if cond_mode == "3cond":
                    condition_correct["script_top1"] += int(
                        script_logits.argmax(1).eq(ys).sum())
            dec = vae.decode(samples / 0.18215).sample  # (b,3,256,256)
            gt = gts[i:j]
            mse_sum += F.mse_loss(dec, gt).item() * (j - i)
            for _k in range(dec.shape[0]):
                ssim_sum += _ssim((dec[_k:_k+1] + 1) / 2, (gt[_k:_k+1] + 1) / 2, 1.0, 11, win)
            if i < args.vis_n:
                def _to_pil(t):
                    a = ((t.clamp(-1, 1) + 1) / 2).squeeze(0).permute(1, 2, 0).cpu().float().numpy()
                    return Image.fromarray((a * 255).clip(0, 255).astype(np.uint8))
                for _k in range(dec.shape[0]):
                    if i + _k >= args.vis_n:
                        break
                    canvas = Image.new("RGB", (512, 256), (20, 20, 20))
                    canvas.paste(_to_pil(dec[_k:_k+1]), (0, 0))
                    canvas.paste(_to_pil(gt[_k:_k+1]), (256, 0))
                    canvas.save(f"{args.out}_imgs/{i+_k:04d}_gen_gt.png")
            if (i + B) % 100 == 0 or j == len(conds):
                print(f"  [{j}/{len(conds)}] mse={mse_sum/j:.5f} ssim={ssim_sum/j:.4f}")
    n = len(conds)
    result = dict(ckpt=args.ckpt, model=model_name, condition_fusion=condition_fusion,
                  weights=source, csv=args.csv, n=n,
                  steps=args.steps, cfg=args.cfg, seed=args.seed,
                  mse=mse_sum / n, ssim=ssim_sum / n,
                  time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if condition_probe is not None:
        result["condition_probe"] = args.condition_probe
        result["condition_accuracy"] = {
            key: value / n for key, value in condition_correct.items()}
    with open(f"{args.out}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[eval_gen] DONE 自由采样 n={n} MSE={result['mse']:.5f} SSIM={result['ssim']:.4f}")


if __name__ == "__main__":
    main()
