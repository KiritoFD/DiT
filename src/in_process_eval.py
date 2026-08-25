"""
In-process GPU evaluation: bf16 DDIM sampling → VAE decode → save PNGs.
Called by train.py after each checkpoint save. GPU-only, NO metric computation.

Metrics (MSE/SSIM/skel_iou) are computed by a separate CPU daemon
(eval_metrics_daemon.py) that watches for eval_pending_*.json markers.

Design:
  - prepare_eval_cache()  : called once at train startup, pre-loads GT images +
                            conditions + fixed noise into CPU RAM.
  - run_gpu_eval()         : called after each ckpt save. Uses the EMA model
                            already on GPU. bf16 autocast for DiT + VAE.
                            Saves ALL N pred+gt PNGs to eval_samples/stepXXXXXXX/.
                            Writes eval_pending_{step}.json for the CPU daemon.
  - load_eval_vae()        : lazily loads the real VAE (training uses MockVAE).
"""
import os, json, csv, time, traceback
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _log(msg):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [gpu-eval] {msg}", flush=True)


# ── Eval cache ───────────────────────────────────────────────────────────────

def prepare_eval_cache(eval_csv, img_root, image_size, n,
                       vae_downscale, latent_channels, scaling_factor):
    """Pre-load N eval samples: GT images ([-1,1] tensors) + conditions + fixed noise.

    Returns dict with CPU tensors:
      gts    : (n, 3, H, W) float32 [-1,1]
      conds  : list of (callig_id, glyph_id)
      noise  : (n, C, H//ds, W//ds) float32  — fixed per-sample for reproducibility
    """
    import torchvision.transforms as T
    _log(f"building eval cache from {eval_csv} (n={n})")
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
    for i, row in enumerate(rows):
        p = row["image_path"]
        # If img_root is set and path is already absolute or already starts
        # with img_root, use it directly. Otherwise join with img_root.
        if img_root and not os.path.isabs(p) and not p.startswith(img_root):
            p = os.path.join(img_root, p)
        from PIL import Image as _Img
        img = _Img.open(p).convert("RGB")
        gts[i] = transform(img)
        callig_id = int(row["calligrapher_id"])
        glyph_id = int(row.get("glyph_id", row.get("character_id", 0)))
        conds.append((callig_id, glyph_id))

    latent_spatial = image_size // vae_downscale
    g = torch.Generator().manual_seed(0)
    noise = torch.randn(n, latent_channels, latent_spatial, latent_spatial, generator=g)

    _log(f"cache ready: {n} samples, gt={gts.shape}, latent={latent_channels}x{latent_spatial}x{latent_spatial}")
    return {
        "gts": gts,
        "conds": conds,
        "noise": noise,
        "n": n,
        "latent_channels": latent_channels,
        "latent_spatial": latent_spatial,
        "scaling_factor": scaling_factor,
    }


# ── VAE loading ───────────────────────────────────────────────────────────────

_eval_vae = None

def load_eval_vae(args, device):
    """Lazily load the real VAE for eval (training may use MockVAE)."""
    global _eval_vae
    if _eval_vae is not None:
        return _eval_vae
    from diffusers.models import AutoencoderKL
    vae_path = getattr(args, "vae_path", None)
    if vae_path and os.path.exists(vae_path):
        _log(f"loading VAE from {vae_path}")
        _eval_vae = AutoencoderKL.from_pretrained(vae_path).to(device).eval()
    else:
        vae_name = getattr(args, "vae", "ema")
        _log(f"loading VAE stabilityai/sd-vae-ft-{vae_name}")
        _eval_vae = AutoencoderKL.from_pretrained(
            f"stabilityai/sd-vae-ft-{vae_name}").to(device).eval()
    for p in _eval_vae.parameters():
        p.requires_grad_(False)
    _log(f"VAE ready on {device}")
    return _eval_vae


# ── GPU sampling + decode + save ───────────────────────────────────────────────

def _save_batch_pngs(preds, gts, conds, out_dir, step, start_idx):
    """Save a batch of pred+gt images as PNGs. preds/gts: (B,3,H,W) [-1,1] on CPU."""
    os.makedirs(out_dir, exist_ok=True)
    pred_np = ((preds.clamp(-1, 1) + 1) / 2).numpy()
    gt_np = ((gts.clamp(-1, 1) + 1) / 2).numpy()
    for k in range(preds.shape[0]):
        idx = start_idx + k
        Image.fromarray((pred_np[k].transpose(1, 2, 0) * 255).clip(0, 255).astype("uint8")).save(
            os.path.join(out_dir, f"sample{idx}.png"))
        Image.fromarray((gt_np[k].transpose(1, 2, 0) * 255).clip(0, 255).astype("uint8")).save(
            os.path.join(out_dir, f"gt{idx}.png"))


def _save_thumb(preds, gts, out_path, n=5):
    """Save a small thumbnail: top row preds, bottom row gts."""
    n = min(n, preds.shape[0])
    if n == 0:
        return
    canvas = np.zeros((preds.shape[2] * 2, preds.shape[3] * n, 3), dtype=np.uint8)
    p = ((preds[:n].clamp(-1, 1) + 1) / 2).numpy()
    g = ((gts[:n].clamp(-1, 1) + 1) / 2).numpy()
    h, w = preds.shape[2], preds.shape[3]
    for k in range(n):
        canvas[:h, k*w:(k+1)*w] = (p[k].transpose(1, 2, 0) * 255).clip(0, 255).astype("uint8")
        canvas[h:, k*w:(k+1)*w] = (g[k].transpose(1, 2, 0) * 255).clip(0, 255).astype("uint8")
    Image.fromarray(canvas).save(out_path)


def run_gpu_eval(ema_model, args, cache, step, checkpoint_dir, device,
                 dit_batch=240, vae_batch=32, ddim_steps=50, cfg_scale=4.0):
    """Run DDIM sampling → VAE decode → save PNGs. GPU-only, NO metric computation.

    Two-phase design to minimize VRAM spikes:
      Phase 1 (DiT): bf16 autocast. Large batch (dit_batch). All DDIM sampling
        produces latents, collected on CPU. DiT is compute-bound, bf16 is fine.
        CFG doubles the effective batch internally (cond+uncond).
      Phase 2 (VAE): fp32 (force_upcast=True demands it). Small batch (vae_batch).
        Decode latents → 256×256 images, save to disk immediately.

    Args:
        ema_model   : EMA model already on GPU (eval mode, no grad)
        args        : train args (for VAE config)
        cache       : from prepare_eval_cache()
        step        : current training step
        checkpoint_dir : where to save eval_samples/
        device      : cuda device
        dit_batch   : DiT sampling batch (before CFG doubling)
        vae_batch   : VAE decode batch (fp32, force_upcast=True)
        ddim_steps  : DDIM timesteps
        cfg_scale   : classifier-free guidance scale
    """
    from diffusion import create_diffusion
    t0 = time.time()

    n = cache["n"]
    lc = cache["latent_channels"]
    ls = cache["latent_spatial"]
    sf = cache["scaling_factor"]
    conds = cache["conds"]
    gts_all = cache["gts"]
    noise_all = cache["noise"]

    diffusion = create_diffusion(str(ddim_steps))
    vae = load_eval_vae(args, device)

    step_tag = f"step{int(step):07d}"
    out_dir = os.path.join(checkpoint_dir, "eval_samples", step_tag)
    os.makedirs(out_dir, exist_ok=True)

    ema_model.eval()
    torch.manual_seed(0)

    # ── Phase 1: DiT DDIM sampling (bf16, large batch) → all latents on CPU ──
    _log(f"phase 1: DiT sampling (bf16, batch={dit_batch}, {n} samples)")
    all_latents = torch.zeros(n, lc, ls, ls, dtype=torch.float32)  # CPU
    with torch.no_grad():
        for i in range(0, n, dit_batch):
            j = min(i + dit_batch, n)
            z = noise_all[i:j].to(device)
            yc = torch.tensor([c[0] for c in conds[i:j]], device=device, dtype=torch.long)
            yh = torch.tensor([c[1] for c in conds[i:j]], device=device, dtype=torch.long)
            mk = dict(y_callig=yc, y_char=yh, cfg_scale=cfg_scale)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                samples = diffusion.ddim_sample_loop(
                    ema_model.forward_with_cfg, z.shape, z,
                    clip_denoised=False, model_kwargs=mk, device=device,
                )
            all_latents[i:j] = samples.float().cpu()
            del z, samples
            torch.cuda.empty_cache()
    _log(f"phase 1 done: {n} latents on CPU, "
         f"peak VRAM so far: {torch.cuda.max_memory_allocated()/1024**3:.2f}G")

    # ── Phase 2: VAE decode (fp32, small batch) → save PNGs ──
    _log(f"phase 2: VAE decode (fp32, batch={vae_batch}, {n} samples)")
    n_saved = 0
    with torch.no_grad():
        for i in range(0, n, vae_batch):
            j = min(i + vae_batch, n)
            lat = all_latents[i:j].to(device)
            # fp32 decode — force_upcast=True in kl-f4, do NOT use autocast
            decoded = vae.decode(lat / sf).sample  # (bs, 3, 256, 256) fp32
            preds_cpu = decoded.float().cpu()
            gts_cpu = gts_all[i:j].clone()

            _save_batch_pngs(preds_cpu, gts_cpu, conds, out_dir, step, i)
            n_saved += j - i

            # thumbnail from first batch
            if i == 0:
                _save_thumb(preds_cpu, gts_cpu,
                            os.path.join(checkpoint_dir, "eval_latest.png"), n=5)

            del lat, decoded, preds_cpu
            torch.cuda.empty_cache()

    del all_latents
    elapsed = time.time() - t0
    _log(f"step {step}: saved {n_saved} images to eval_samples/{step_tag}/ ({elapsed:.1f}s), "
         f"peak VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f}G")

    # Write samples.json (completeness marker for poster generation)
    import json as _json
    with open(os.path.join(out_dir, "samples.json"), "w", encoding="utf-8") as _f:
        _json.dump({"step": step, "n": n_saved, "cfg": cfg_scale,
                    "ddim_steps": ddim_steps, "dit_batch": dit_batch,
                    "vae_batch": vae_batch}, _f, ensure_ascii=False)

    # Write pending marker for CPU metrics daemon
    pending = {
        "step": step,
        "n": n_saved,
        "dir": os.path.join("eval_samples", step_tag),
        "elapsed_gpu": elapsed,
        "ddim_steps": ddim_steps,
        "cfg_scale": cfg_scale,
        "dit_batch": dit_batch,
        "vae_batch": vae_batch,
    }
    pending_path = os.path.join(checkpoint_dir, f"eval_pending_{int(step):07d}.json")
    with open(pending_path, "w") as f:
        json.dump(pending, f, indent=2)
    _log(f"wrote pending marker: {pending_path}")

    return n_saved, elapsed


# ── Show5 / Seen5 ─────────────────────────────────────────────────────────────

def prepare_small_cache(csv_path, img_root, image_size, vae_downscale, latent_channels):
    """Build a small cache (5 samples) for show5/seen5 visualization."""
    if not csv_path or not os.path.exists(csv_path):
        return None
    return prepare_eval_cache(csv_path, img_root, image_size, 5,
                              vae_downscale, latent_channels, 0.102079)


def run_show5(ema_model, args, cache, step, checkpoint_dir, device,
               ddim_steps=50, cfg_scale=4.0, tag="show5"):
    """Run eval on a small set and save to seen_samples/ or show_samples/."""
    if cache is None:
        return
    from diffusion import create_diffusion
    n = cache["n"]
    lc = cache["latent_channels"]
    ls = cache["latent_spatial"]
    sf = cache["scaling_factor"]
    conds = cache["conds"]
    noise_all = cache["noise"]
    gts_all = cache["gts"]

    diffusion = create_diffusion(str(ddim_steps))
    vae = load_eval_vae(args, device)

    step_tag = f"step{int(step):07d}"
    subdir = "show_samples" if tag == "show5" else "seen_samples"
    out_dir = os.path.join(checkpoint_dir, subdir, step_tag)

    ema_model.eval()
    with torch.no_grad():
        z = noise_all[:n].to(device)
        yc = torch.tensor([c[0] for c in conds[:n]], device=device, dtype=torch.long)
        yh = torch.tensor([c[1] for c in conds[:n]], device=device, dtype=torch.long)
        mk = dict(y_callig=yc, y_char=yh, cfg_scale=cfg_scale)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            samples = diffusion.ddim_sample_loop(
                ema_model.forward_with_cfg, z.shape, z,
                clip_denoised=False, model_kwargs=mk, device=device,
            )
            decoded = vae.decode(samples / sf).sample
        _save_batch_pngs(decoded.float().cpu(), gts_all[:n].clone(),
                         conds, out_dir, step, 0)
        # Write samples.json (completeness marker)
        import json as _json
        with open(os.path.join(out_dir, "samples.json"), "w", encoding="utf-8") as _f:
            _json.dump({"step": step, "n": n, "cfg": cfg_scale,
                        "ddim_steps": ddim_steps}, _f, ensure_ascii=False)
        _save_thumb(decoded.float().cpu(), gts_all[:n].clone(),
                    os.path.join(checkpoint_dir, f"{tag}_latest.png"), n=5)
        del samples, decoded
        torch.cuda.empty_cache()
    _log(f"{tag}: saved {n} images to {subdir}/{step_tag}/")
