#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Strict apples-to-apples eval on eval_strict_top6.csv (271 rows).

Loads a ckpt (ddpm or flow, auto-detected from args.diffusion_type), samples with
the CORRECT sampler for its diffusion type, decodes, and computes per-sample
MSE/SSIM/skel_iou with mean/std/min + 25/50/75 quantiles.

Usage:
  python _strict_eval_compare.py --ckpt <path.pt> --out <prefix>
"""
import argparse, os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)

def log(m):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)

# metrics
try:
    from skimage.morphology import skeletonize
    _HAS_SK = True
except ImportError:
    _HAS_SK = False
    from scipy.ndimage import binary_erosion, generate_binary_structure
    def _skel_fb(b):
        s = np.zeros_like(b); img = b.copy(); st = generate_binary_structure(2, 2)
        while img.any():
            e = binary_erosion(img, structure=st); s |= img & ~e; img = e
        return s

def _ssim_np(a, b, win=7):
    from scipy.ndimage import uniform_filter
    c1, c2 = 0.01**2, 0.03**2; ssims = []
    for ch in range(3):
        x = a[:,:,ch].astype(np.float64); y = b[:,:,ch].astype(np.float64)
        mx = uniform_filter(x,win); my = uniform_filter(y,win)
        mx2=mx**2; my2=my**2; mxy=mx*my
        sx2=uniform_filter(x*x,win)-mx2; sy2=uniform_filter(y*y,win)-my2; sxy=uniform_filter(x*y,win)-mxy
        ssims.append((((2*mxy+c1)*(2*sxy+c2))/((mx2+my2+c1)*(sx2+sy2+c2))).mean())
    return float(np.mean(ssims))

def _skel_iou(a, b, t=0.5):
    g1=a.mean(2); g2=b.mean(2); b1=g1<t; b2=g2<t
    if not b1.any() and not b2.any(): return 1.0
    if not b1.any() or not b2.any(): return 0.0
    s1=skeletonize(b1) if _HAS_SK else _skel_fb(b1)
    s2=skeletonize(b2) if _HAS_SK else _skel_fb(b2)
    i=(s1&s2).sum(); u=(s1|s2).sum()
    return float(i/u) if u>0 else 1.0

def _mse(a, b):
    return float(np.mean((a-b)**2)) * 4.0  # [0,1] -> [-1,1] convention

def _q(arr, p):
    s = np.sort(np.asarray(arr, dtype=np.float64))
    if len(s) == 0: return 0.0
    idx = (len(s)-1) * p
    lo = int(np.floor(idx)); hi = int(np.ceil(idx))
    if lo == hi: return float(s[lo])
    return float(s[lo] + (s[hi]-s[lo]) * (idx-lo))

def build_cache(eval_csv, n=271):
    rows = list(__import__('csv').DictReader(open(eval_csv, encoding="utf-8")))[:n]
    img_root = "final_imgs_256"
    transform = T.Compose([T.Resize((256,256), interpolation=T.InterpolationMode.BICUBIC),
                           T.ToTensor(), T.Normalize(mean=[0.5]*3, std=[0.5]*3)])
    gts = torch.zeros(n, 3, 256, 256, dtype=torch.float32)
    conds = []
    paths = []
    for i, r in enumerate(rows):
        p = r["image_path"]
        if not os.path.isabs(p) and not p.startswith(img_root):
            p = os.path.join(img_root, p)
        gts[i] = transform(Image.open(p).convert("RGB"))
        conds.append((int(r["calligrapher_id"]), int(r.get("glyph_id", r.get("character_id", 0)))))
        paths.append(p)
    g = torch.Generator().manual_seed(0)
    noise = torch.randn(n, 4, 32, 32, generator=g)
    return {"gts": gts, "conds": conds, "noise": noise, "n": n, "paths": paths}

def build_model(args, device):
    from models import DiT_2Cond_models
    ls = args.image_size // 8
    m = DiT_2Cond_models[args.model](
        input_size=ls, in_channels=4,
        num_calligraphers=args.num_calligraphers,
        num_characters=args.num_characters,
        condition_fusion=args.condition_fusion,
        callig_embed_dim=int(args.callig_embed_dim),
        char_embed_dim=int(args.char_embed_dim),
        learn_sigma=True,
        cond_drop_all_prob=float(getattr(args, "cond_drop_all_prob", 0.05)),
        cond_drop_one_prob=float(getattr(args, "cond_drop_one_prob", 0.25)),
        use_glyph_cond=getattr(args, "w_glyph_cond", False),
        skel_head_enabled=getattr(args, "w_skel_head", 0) > 0,
        char_proj_mode=getattr(args, "char_proj_mode", "full"),
        freeze_char_table=getattr(args, "freeze_char_table", False),
    ).to(device).eval()
    return m

def inject_dino(model, args):
    p = getattr(args, "char_dino_embeddings", None)
    idxp = getattr(args, "char_dino_index", None)
    if not p or not os.path.isfile(p) or not os.path.isfile(idxp):
        log("[dino] missing files, skip"); return
    emb = np.load(p)
    glyphs = json.load(open(idxp)).get("glyphs", [])
    table = model.y_char_embedder.embedding_table.weight
    NUM_CH = 7026; n = 0
    with torch.no_grad():
        for gi, (sid, cid) in enumerate(glyphs):
            gid = int(sid)*NUM_CH + int(cid)
            if 0 <= gid < table.shape[0] and gi < emb.shape[0]:
                e = emb[gi]; e = e/(np.linalg.norm(e)+1e-8)
                table.data[gid] = torch.from_numpy(e).float()
                n += 1
    log(f"[dino] injected {n}")

def sample(model, cache, device, diffusion, lc=4, ls=32, cfg=4.0, fwd_batch=256):
    n = cache["n"]
    z = cache["noise"].to(device).half()
    yc = torch.tensor([c[0] for c in cache["conds"]], device=device, dtype=torch.long)
    yh = torch.tensor([c[1] for c in cache["conds"]], device=device, dtype=torch.long)
    n_classes_callig = model.y_callig_embedder.num_classes
    n_classes_char = model.y_char_embedder.num_classes
    all_lat = torch.zeros(n, lc, ls, ls, dtype=torch.float32)
    with torch.no_grad():
        if getattr(diffusion, "is_flow", False):
            # flow: Euler, uses forward_with_cfg (CFG over callig/char)
            from diffusion import FlowMatching
            steps = diffusion.num_timesteps
            dt = 1.0/steps
            x = z.to(device).float()
            for i in range(steps):
                t = 1.0 - i*dt
                out = []
                for s in range(0, n, fwd_batch):
                    e = min(s+fwd_batch, n)
                    bs = e-s
                    xb = x[s:e]; tb = torch.full((bs,), t, device=device); ycb = yc[s:e]; yhb = yh[s:e]
                    # CFG
                    z2 = torch.cat([xb, xb], 0); t2 = torch.cat([tb, tb], 0)
                    yc2 = torch.cat([ycb, torch.full_like(ycb, n_classes_callig)], 0)
                    yh2 = torch.cat([yhb, torch.full_like(yhb, n_classes_char)], 0)
                    with torch.autocast("cuda", dtype=torch.float16):
                        v = model(z2, t2*1000.0, yc2, yh2)
                    if isinstance(v, tuple): v = v[0]
                    v = v[:, :lc].float()
                    cv, uv = v[:bs], v[bs:]
                    hv = uv + cfg*(cv-uv)
                    out.append(hv)
                v_all = torch.cat(out, 0)
                x = x.float() - dt*v_all
            all_lat = x.float().cpu()
        else:
            # ddpm: DDIM (use the library's ddim_sample_loop + forward_with_cfg for correctness)
            from diffusion import create_diffusion
            ddim = create_diffusion(str(50))
            n_classes_callig = model.y_callig_embedder.num_classes
            n_classes_char = model.y_char_embedder.num_classes
            x = z.to(device).float()
            mk = dict(y_callig=yc, y_char=yh, cfg_scale=cfg)
            all_lat = ddim.ddim_sample_loop(
                model.forward_with_cfg, (n, lc, ls, ls),
                noise=x, clip_denoised=False, model_kwargs=mk,
                device=device, progress=False, eta=0.0,
            ).float().cpu()
    return all_lat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-csv", default="5script/eval_strict_top6.csv")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--fwd-batch", type=int, default=128)
    ap.add_argument("--vae-batch", type=int, default=32)
    ap.add_argument("--n", type=int, default=271)
    args = ap.parse_args()

    device = "cuda"
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ta = ckpt["args"]
    dt = getattr(ta, "diffusion_type", "ddpm")
    log(f"ckpt={args.ckpt} diffusion_type={dt} char_embed_dim={getattr(ta,'char_embed_dim',None)}")

    cache = build_cache(args.eval_csv, args.n)
    model = build_model(ta, device)  # fp32
    inject_dino(model, ta)
    # load weights
    sd = ckpt.get("ema", ckpt.get("state_dict", ckpt))
    model.load_state_dict(sd, strict=False)
    model.eval()

    if dt in ("flow", "flow_matching", "fm"):
        from diffusion import create_flow_matching
        diffusion = create_flow_matching(str(args.steps))
    else:
        from diffusion import create_diffusion
        diffusion = create_diffusion(str(args.steps))

    lat = sample(model, cache, device, diffusion, 4, 32, args.cfg, args.fwd_batch)

    # VAE decode
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("pretrained_models/sd-vae-ft-ema").to(device).eval()
    sf = float(getattr(ta, "vae_scaling_factor", 0.18215))
    decoded = []
    with torch.no_grad():
        for i in range(0, args.n, args.vae_batch):
            j = min(i+args.vae_batch, args.n)
            dec = vae.decode(lat[i:j].to(device).float()/sf).sample
            decoded.append(dec.float().cpu())
    decoded = torch.cat(decoded, 0)
    gts = cache["gts"]

    dec_np = ((decoded.clamp(-1,1)+1)/2).numpy()   # (n,3,256,256)
    gt_np = ((gts.clamp(-1,1)+1)/2).numpy()
    ms, ss, sk = [], [], []
    for i in range(args.n):
        a = dec_np[i].transpose(1,2,0)
        b = gt_np[i].transpose(1,2,0)
        ms.append(_mse(a,b)); ss.append(_ssim_np(a,b)); sk.append(_skel_iou(a,b))
    res = {
        "ckpt": args.ckpt, "diffusion_type": dt, "n": args.n,
        "cfg": args.cfg, "steps": args.steps,
        "mse": {"mean": float(np.mean(ms)), "std": float(np.std(ms)),
                "min": float(np.min(ms)), "q25": _q(ms,0.25), "q50": _q(ms,0.5), "q75": _q(ms,0.75)},
        "ssim": {"mean": float(np.mean(ss)), "std": float(np.std(ss)),
                 "min": float(np.min(ss)), "q25": _q(ss,0.25), "q50": _q(ss,0.5), "q75": _q(ss,0.75)},
        "skel_iou": {"mean": float(np.mean(sk)), "std": float(np.std(sk)),
                     "min": float(np.min(sk)), "q25": _q(sk,0.25), "q50": _q(sk,0.5), "q75": _q(sk,0.75)},
    }
    out = args.out if args.out.endswith(".json") else args.out+".json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    log(f"wrote {out}")
    log(json.dumps(res, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
