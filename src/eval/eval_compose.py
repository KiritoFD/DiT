# -*- coding: utf-8 -*-
"""V3-A 二因子 pair-composed 评估（正确的 score 组合）。

对每个样本，在 DDIM 的**每一步**上做 4 路 forward：
  full : eps(x, t, callig, glyph)
  A    : eps(x, t, callig, drop-glyph)   -> style marginal score
  G    : eps(x, t, drop-callig, glyph)   -> content marginal score
  0    : eps(x, t, drop, drop)           -> unconditional base

然后按 Möbius / product-of-experts 组合（CFG 公式）：
  eps_c = eps_0 + cfg * ( wI*(eps_full - eps_0) + (1-wI)*((eps_A - eps_0) + (eps_G - eps_0)) )
其中 wI=1 -> 纯 full CFG；wI=0 -> 纯 pair-composed（s_A + s_G - s_0）。

用组合后的 eps 反推 x0，再走一步 DDIM（eta=0 确定性），循环到 t=0。
输出各 wI 与 full-CFG 的 MSE/SSIM（vs GT），并落盘可视化 (full | pair | GT)。

用法:
  /opt/conda/bin/python eval_compose.py --ckpt <ckpt.pt> \
      --csv 5script/eval_strata/clean_unseen_triple_100.csv --n 100 --out compose_eval
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
from src.model import DiT_2Cond_models
from src.utils import MCCDDataset
from diffusers.models import AutoencoderKL
from src.loss import create_diffusion
from eval_auto import _gaussian_window, _ssim
from src.loss.gaussian_diffusion import _extract_into_tensor


def compose_ddim_loop(diffusion, model, z, conds_yc, conds_yh, cfg, wI, device,
                      clip_denoised=False):
    """每步 4 路 forward，组合 eps 后走确定性 DDIM 更新。返回最终 (B,4,32,32)。"""
    B = z.shape[0]
    n_class_callig = model.y_callig_embedder.num_classes
    n_class_glyph = model.y_char_embedder.num_classes
    drop_c = torch.full_like(conds_yc, n_class_callig)
    drop_g = torch.full_like(conds_yh, n_class_glyph)

    img = z.clone()
    indices = list(range(diffusion.num_timesteps))[::-1]
    with torch.no_grad():
        for i in indices:
            t = torch.full((B,), i, device=device, dtype=torch.long)
            # 4 路 forward：一次拼 4B 更快
            x4 = torch.cat([img, img, img, img], dim=0)
            t4 = torch.cat([t, t, t, t], dim=0)
            yc4 = torch.cat([conds_yc, conds_yc, drop_c, drop_c], dim=0)
            yh4 = torch.cat([conds_yh, drop_g, conds_yh, drop_g], dim=0)
            out4 = model(x4, t4, yc4, yh4)
            eps4 = out4[:, :model.in_channels]
            eps_full, eps_A, eps_G, eps_0 = eps4.chunk(4, dim=0)

            # 组合 score（eps 空间线性组合 = score 组合）
            eps_c = eps_0 + cfg * (
                wI * (eps_full - eps_0)
                + (1.0 - wI) * ((eps_A - eps_0) + (eps_G - eps_0))
            )

            # eps -> x0 -> DDIM 更新（eta=0）
            xstart = diffusion._predict_xstart_from_eps(img, t, eps_c)
            alpha_bar = _extract_into_tensor(diffusion.alphas_cumprod, t, img.shape)
            alpha_bar_prev = _extract_into_tensor(diffusion.alphas_cumprod_prev, t, img.shape)
            img = (xstart * torch.sqrt(alpha_bar_prev)
                   + torch.sqrt(1.0 - alpha_bar_prev) * eps_c)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", default="5script/eval_strata/clean_unseen_triple_100.csv")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="compose_eval")
    ap.add_argument("--vis-n", type=int, default=8)
    ap.add_argument("--wI", type=float, nargs="*", default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--vae-path", default="pretrained_models/sd-vae-ft-ema")
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

    model_name = saved("model", "DiT-2Cond-S/2")
    num_calligraphers = saved("num_calligraphers", 1011)
    num_characters = saved("num_characters", 35130)
    condition_fusion = saved("condition_fusion", "factorized_add")
    callig_embed_dim = saved("callig_embed_dim", 128)
    char_embed_dim = saved("char_embed_dim", 192)

    print(f"[compose] ckpt={args.ckpt} model={model_name} cfg={args.cfg} "
          f"steps={args.steps} wI={args.wI}")

    model = DiT_2Cond_models[model_name](
        input_size=32, num_calligraphers=num_calligraphers,
        num_characters=num_characters, use_checkpoint=False,
        condition_fusion=condition_fusion,
        callig_embed_dim=callig_embed_dim, char_embed_dim=char_embed_dim,
        cond_drop_all_prob=saved("cond_drop_all_prob", 0.10),
        cond_drop_one_prob=saved("cond_drop_one_prob", 0.30))
    delta = ckpt.get("ema", ckpt.get("delta", ckpt.get("model", ckpt)))
    missing, unexpected = model.load_state_dict(delta, strict=False)
    print(f"[compose] loaded: missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device).eval()
    ddim = create_diffusion(str(args.steps))  # SpacedDiffusion, num_timesteps=steps

    ds = MCCDDataset(args.csv, "", image_size=256, load_canny=False, load_skel=False)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    conds, gts = [], []
    for i, b in enumerate(loader):
        if len(conds) >= args.n:
            break
        conds.append((b["y_callig"].item(), b["y_char"].item()))
        gts.append(b["image"])
    gts = torch.cat(gts).to(device)
    print(f"[compose] {len(conds)} samples")

    win = _gaussian_window(11, 1.5, device)
    results = {f"wI_{w}": {"mse": 0.0, "ssim": 0.0} for w in args.wI}
    os.makedirs(f"{args.out}_imgs", exist_ok=True)
    torch.manual_seed(args.seed)
    B = args.batch

    with torch.no_grad():
        for i in range(0, len(conds), B):
            j = min(i + B, len(conds))
            z = torch.randn(j - i, 4, 32, 32, device=device)
            yc = torch.tensor([c[0] for c in conds[i:j]], device=device)
            yh = torch.tensor([c[1] for c in conds[i:j]], device=device)
            gt = gts[i:j]

            decs = {}
            for w in args.wI:
                samples = compose_ddim_loop(ddim, model, z, yc, yh, args.cfg, w, device)
                decs[w] = vae.decode(samples / 0.18215).sample

            for w in args.wI:
                dec = decs[w]
                results[f"wI_{w}"]["mse"] += F.mse_loss(dec, gt).item() * (j - i)
                for _k in range(dec.shape[0]):
                    results[f"wI_{w}"]["ssim"] += _ssim(
                        (dec[_k:_k+1] + 1) / 2, (gt[_k:_k+1] + 1) / 2, 1.0, 11, win)

            # 可视化：full(wI=1) | pair(wI=0) | GT
            if i < args.vis_n:
                for _k in range(min(j - i, args.vis_n - i)):
                    def _pil(t):
                        a = ((t.clamp(-1, 1) + 1) / 2).squeeze(0).permute(1, 2, 0).cpu().float().numpy()
                        return Image.fromarray((a * 255).clip(0, 255).astype(np.uint8))
                    canvas = Image.new("RGB", (256 * 3, 256))
                    canvas.paste(_pil(decs[1.0][_k:_k+1]), (0, 0))
                    canvas.paste(_pil(decs[0.0][_k:_k+1]), (256, 0))
                    canvas.paste(_pil(gt[_k:_k+1]), (512, 0))
                    canvas.save(f"{args.out}_imgs/{i+_k:04d}_full_pair_gt.png")
            if (i + B) % 40 == 0 or j == len(conds):
                print(f"  [{j}/{len(conds)}] wI=1.0 mse={results['wI_1.0']['mse']/j:.5f} "
                      f"wI=0.0 mse={results['wI_0.0']['mse']/j:.5f}")

    n = len(conds)
    for k in results:
        results[k]["mse"] /= n
        results[k]["ssim"] /= n
    results.update(ckpt=args.ckpt, csv=args.csv, n=n, steps=args.steps, cfg=args.cfg,
                   seed=args.seed, time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    out_json = f"{args.out}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"[compose] saved -> {out_json}")


if __name__ == "__main__":
    main()
