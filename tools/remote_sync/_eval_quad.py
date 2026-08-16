# -*- coding: utf-8 -*-
"""生成 eval 四联对比图：pred | GT | GT-canny | GT-skeleton。

从 ckpt 用 free-sampling（纯噪声 -> DDIM -> CFG）生成前 N 个 eval 样本，
与 GT 原图、GT canny、GT skeleton 拼成一张横向四联图，供 dashboard 展示。

用法:
  /opt/conda/bin/python _eval_quad.py --ckpt <ckpt.pt> \
      --csv 5script/eval_strata/clean_unseen_triple_100.csv --n 5 --out eval_quad.png
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
from dataset import MCCDDataset
from diffusers.models import AutoencoderKL
from diffusion import create_diffusion
from eval_auto import _gaussian_window, _ssim


def _s2b(v):
    return str(v).lower() in ("1", "true", "yes", "on")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", default="5script/eval_strata/clean_unseen_triple_100.csv")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--out", default="eval_quad.png")
    ap.add_argument("--vae-path", default="pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--canny-root", default="final_canny")
    ap.add_argument("--skel-root", default="final_skeleton")
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

    cond_mode = saved("cond_mode", "2cond")
    model_name = saved("model", "DiT-2Cond-S/2")
    num_calligraphers = saved("num_calligraphers", 1011)
    num_scripts = saved("num_scripts", 5)
    num_characters = saved("num_characters", 35130)
    condition_fusion = saved("condition_fusion", "factorized_add")
    callig_embed_dim = saved("callig_embed_dim", 128)
    script_embed_dim = saved("script_embed_dim", None)
    char_embed_dim = saved("char_embed_dim", 192)

    if cond_mode == "2cond":
        model = DiT_2Cond_models[model_name](
            input_size=32, num_calligraphers=num_calligraphers,
            num_characters=num_characters, use_checkpoint=False,
            condition_fusion=condition_fusion,
            callig_embed_dim=callig_embed_dim, char_embed_dim=char_embed_dim,
            cond_drop_all_prob=saved("cond_drop_all_prob", 0.10),
            cond_drop_one_prob=saved("cond_drop_one_prob", 0.30))
    else:
        model = DiT_3Cond_models[model_name](
            input_size=32, num_calligraphers=num_calligraphers,
            num_scripts=num_scripts, num_characters=num_characters,
            use_checkpoint=False, condition_fusion=condition_fusion,
            callig_embed_dim=callig_embed_dim, script_embed_dim=script_embed_dim,
            char_embed_dim=char_embed_dim,
            cond_drop_all_prob=saved("cond_drop_all_prob", 0.05),
            cond_drop_one_prob=saved("cond_drop_one_prob", 0.0))
    delta = ckpt.get("ema", ckpt.get("delta", ckpt.get("model", ckpt)))
    missing, unexpected = model.load_state_dict(delta, strict=False)
    print(f"[quad] loaded {len(delta)} tensors: missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device).eval()
    ddim = create_diffusion(str(args.steps))

    # 读 eval csv 原样行（含 image_path/calligrapher/script/character 名字）
    import csv as _csv
    with open(args.csv, encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))[: args.n]

    # 用 dataset 拿条件 id（y_char 已是 glyph_id）
    ds = MCCDDataset(args.csv, "", image_size=256, load_canny=False, load_skel=False)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    conds = []
    for i, b in enumerate(loader):
        if len(conds) >= args.n:
            break
        conds.append((b["y_callig"].item(), b["y_char"].item()))
    print(f"[quad] {len(conds)} samples, cond_mode={cond_mode}")

    def _img_to_pil(t):
        a = ((t.clamp(-1, 1) + 1) / 2).squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        return Image.fromarray((a * 255).clip(0, 255).astype(np.uint8))

    torch.manual_seed(args.seed)
    B = args.batch
    dec_all, meta = [], []
    with torch.no_grad():
        for i in range(0, len(conds), B):
            j = min(i + B, len(conds))
            z = torch.randn(j - i, 4, 32, 32, device=device)
            yc = torch.tensor([c[0] for c in conds[i:j]], device=device)
            yh = torch.tensor([c[1] for c in conds[i:j]], device=device)
            if cond_mode == "2cond":
                mk = dict(y_callig=yc, y_char=yh, cfg_scale=args.cfg)
            else:
                ys = torch.tensor([c[1] for c in conds[i:j]], device=device)
                mk = dict(y_callig=yc, y_script=ys, y_char=yh, cfg_scale=args.cfg)
            samples = ddim.ddim_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False,
                model_kwargs=mk, progress=False, device=device)
            dec = vae.decode(samples / 0.18215).sample
            for _k in range(dec.shape[0]):
                dec_all.append(_img_to_pil(dec[_k:_k + 1]))
                meta.append(rows[i + _k])

    # 拼四联：pred | GT | GT-canny | GT-skel
    W, H = 256, 256
    canvas = Image.new("RGB", (W * 4, H * len(dec_all)), (20, 20, 20))
    for k, (pred, row) in enumerate(zip(dec_all, meta)):
        img_id = os.path.basename(row["image_path"])[:-4]
        gt = Image.open(row["image_path"]).convert("RGB").resize((W, H))
        canny_p = os.path.join(args.canny_root, f"{img_id}.png")
        skel_p = os.path.join(args.skel_root, f"{img_id}.png")
        canny = Image.open(canny_p).convert("RGB").resize((W, H)) if os.path.exists(canny_p) else Image.new("RGB", (W, H), (60, 60, 60))
        skel = Image.open(skel_p).convert("RGB").resize((W, H)) if os.path.exists(skel_p) else Image.new("RGB", (W, H), (60, 60, 60))
        canvas.paste(pred, (0, k * H))
        canvas.paste(gt, (W, k * H))
        canvas.paste(canny, (W * 2, k * H))
        canvas.paste(skel, (W * 3, k * H))

    canvas.save(args.out)
    print(f"[quad] saved -> {args.out} ({len(dec_all)} rows, pred|GT|canny|skel)")

    # 元数据 json（供 dashboard 显示条件名）
    out_json = os.path.splitext(args.out)[0] + ".json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "ckpt": args.ckpt, "csv": args.csv, "n": len(dec_all),
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "rows": [{
                "calligrapher": r["calligrapher"], "script": r["script"],
                "character": r["character"], "image": r["image_path"],
            } for r in meta],
        }, f, ensure_ascii=False, indent=2)
    print(f"[quad] meta -> {out_json}")


if __name__ == "__main__":
    main()
