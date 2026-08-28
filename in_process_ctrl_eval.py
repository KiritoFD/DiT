# -*- coding: utf-8 -*-
"""
In-process GPU eval for flow ControlNet: sample (base no-skel + ctrl with-skel)
in bf16 on GPU, VAE-decode fp32, save PNGs, write pending markers for a CPU
metrics daemon. GPU-only for generation; metrics computed from images on CPU.

Model: ControlNetDiT (frozen main + ctrl_encoder) already on GPU in the
training process — reuses its GPU memory (in-process), no extra model load.
"""
import os, json, csv, time
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


def _log(msg):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [gpu-ctrl-eval] {msg}", flush=True)


def prepare_ctrl_eval_cache(eval_csv, img_root, skel_root, image_size, n,
                            vae_downscale, latent_channels, scaling_factor):
    """Pre-load N eval samples: GT images + conditions + skels + fixed noise (CPU)."""
    rows = list(csv.DictReader(open(eval_csv, encoding="utf-8")))
    if n > len(rows):
        n = len(rows)
    rows = rows[:n]
    transform = T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    gts = torch.zeros(n, 3, image_size, image_size, dtype=torch.float32)
    conds = []
    skels = torch.zeros(n, 1, image_size, image_size, dtype=torch.float32)
    paths = []
    for i, row in enumerate(rows):
        p = row["image_path"]
        if img_root and not os.path.isabs(p) and not p.startswith(img_root):
            p = os.path.join(img_root, p)
        gts[i] = transform(Image.open(p).convert("RGB"))
        callig_id = int(row["calligrapher_id"])
        glyph_id = int(row.get("glyph_id", row.get("character_id", 0)))
        conds.append((callig_id, glyph_id))
        # skel from img id (final_skeleton_d3/<id>.png)
        import re
        m = re.search(r"(\d+)\.png", p)
        if m:
            sk = Image.open(os.path.join(skel_root, f"{int(m.group(1))}.png")).convert("L")
            sk = sk.resize((image_size, image_size), Image.NEAREST)
            skels[i, 0] = torch.from_numpy(np.asarray(sk, np.float32)/255.0)
        paths.append(p)
    latent_spatial = image_size // vae_downscale
    g = torch.Generator().manual_seed(0)
    noise = torch.randn(n, latent_channels, latent_spatial, latent_spatial, generator=g)
    _log(f"cache ready: {n} samples, gt={gts.shape}, skel={skels.shape}")
    return {"gts": gts, "conds": conds, "noise": noise, "skels": skels,
            "n": n, "latent_channels": latent_channels,
            "latent_spatial": latent_spatial, "scaling_factor": scaling_factor}


def _save_batch_pngs(preds, gts, skels, conds, out_dir, step, offset, tag):
    os.makedirs(out_dir, exist_ok=True)
    for k in range(preds.shape[0]):
        i = offset + k
        p = ((preds[k].clamp(-1, 1) + 1) / 2).clamp(0, 1)
        g = ((gts[k].clamp(-1, 1) + 1) / 2).clamp(0, 1)
        Image.fromarray((p.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(
            os.path.join(out_dir, f"{tag}{i}.png"))
        Image.fromarray((g.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(
            os.path.join(out_dir, f"gt{i}.png"))
        if skels is not None:
            s = skels[k, 0].numpy()
            Image.fromarray((s * 255).astype(np.uint8)).save(
                os.path.join(out_dir, f"skel{i}.png"))


def run_ctrl_gpu_eval(ctrl, vae, diffusion, cache, device, step, checkpoint_dir,
                      ddim_steps=50, cfg_scale=4.0, dit_batch=16, vae_batch=16,
                      use_skel=True, tag="ctrl"):
    """Sample with flow Euler on GPU (in-process), decode fp32, save PNGs.

    Returns (n, elapsed). Writes samples.json + returns n. If use_skel=True,
    passes GT skel as cond (ControlNet path); else cond=None (base).
    """
    t0 = time.time()
    n = cache["n"]
    lc = cache["latent_channels"]
    ls = cache["latent_spatial"]
    sf = cache["scaling_factor"]
    conds = cache["conds"]
    gts_all = cache["gts"]
    noise_all = cache["noise"]
    skels = cache.get("skels")

    step_tag = f"step{int(step):07d}"
    out_dir = os.path.join(checkpoint_dir, "eval_samples_ctrl", step_tag, tag)
    os.makedirs(out_dir, exist_ok=True)

    ctrl.eval()
    torch.manual_seed(0)

    # Phase 1: flow sampling (bf16) -> latents on CPU
    all_latents = torch.zeros(n, lc, ls, ls, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, n, dit_batch):
            j = min(i + dit_batch, n)
            z = noise_all[i:j].to(device)
            yc = torch.tensor([c[0] for c in conds[i:j]], device=device, dtype=torch.long)
            yh = torch.tensor([c[1] for c in conds[i:j]], device=device, dtype=torch.long)
            mk = dict(y_callig=yc, y_char=yh, cfg_scale=cfg_scale)
            if use_skel and skels is not None:
                mk["cond"] = skels[i:j].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                samples = diffusion.ddim_sample_loop(
                    ctrl.forward_with_cfg, z.shape, z,
                    clip_denoised=False, model_kwargs=mk, device=device)
            all_latents[i:j] = samples.float().cpu()
            del z, samples
            torch.cuda.empty_cache()
    _log(f"phase 1 ({tag}): {n} latents on CPU")

    # Phase 2: VAE decode (fp32) -> save PNGs
    n_saved = 0
    with torch.no_grad():
        for i in range(0, n, vae_batch):
            j = min(i + vae_batch, n)
            lat = all_latents[i:j].to(device)
            decoded = vae.decode(lat / sf).sample  # fp32
            preds_cpu = decoded.float().cpu()
            gts_cpu = gts_all[i:j].clone()
            skels_cpu = skels[i:j] if skels is not None else None
            _save_batch_pngs(preds_cpu, gts_cpu, skels_cpu, conds, out_dir, step, i, tag)
            n_saved += j - i
            del lat, decoded, preds_cpu
            torch.cuda.empty_cache()
    del all_latents
    elapsed = time.time() - t0
    _log(f"phase 2 ({tag}): saved {n_saved} PNGs to {out_dir} ({elapsed:.1f}s)")
    import json as _json
    with open(os.path.join(out_dir, "samples.json"), "w") as f:
        _json.dump({"step": step, "n": n_saved, "cfg": cfg_scale,
                    "ddim_steps": ddim_steps, "tag": tag}, f, ensure_ascii=False)
    return n_saved, elapsed


def run_ctrl_pair_eval(ctrl, vae, diffusion, cache, device, step, checkpoint_dir,
                       ddim_steps=50, cfg_scale=4.0, dit_batch=16, vae_batch=16):
    """Run both base (no skel) and ctrl (with skel) GPU evals, write one pending marker."""
    n, e_ctrl = run_ctrl_gpu_eval(ctrl, vae, diffusion, cache, device, step, checkpoint_dir,
                                  ddim_steps=ddim_steps, cfg_scale=cfg_scale,
                                  dit_batch=dit_batch, vae_batch=vae_batch,
                                  use_skel=True, tag="ctrl")
    nb, e_base = run_ctrl_gpu_eval(ctrl, vae, diffusion, cache, device, step, checkpoint_dir,
                                   ddim_steps=ddim_steps, cfg_scale=cfg_scale,
                                   dit_batch=dit_batch, vae_batch=vae_batch,
                                   use_skel=False, tag="base")
    import json as _json
    pending = {"step": step, "n": n, "nb": nb,
               "step_tag": f"step{int(step):07d}",
               "elapsed_ctrl": e_ctrl, "elapsed_base": e_base,
               "ddim_steps": ddim_steps, "cfg_scale": cfg_scale}
    pending_path = os.path.join(checkpoint_dir, f"eval_pending_ctrl_{int(step):07d}.json")
    with open(pending_path, "w") as f:
        _json.dump(pending, f, indent=2)
    _log(f"wrote pending marker: {pending_path}")
    return pending
