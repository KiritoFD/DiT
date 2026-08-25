#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gpu_batch_eval.py — 独占 GPU 批量评测 + batch 寻优。
加载 ckpt → 大 batch GPU 推理 → 落盘 → CPU 指标 (MSE/SSIM/SkelIoU)。
"""
import argparse, glob, json, os, sys, time, traceback
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

BASE = "/root/Workspace/xy/DiT"

def log(msg):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# ── skel iou ──
try:
    from skimage.morphology import skeletonize
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False
    from scipy.ndimage import binary_erosion, generate_binary_structure
    def _skeletonize_fallback(binary):
        skel = np.zeros_like(binary)
        img = binary.copy()
        struct = generate_binary_structure(2, 2)
        while img.any():
            eroded = binary_erosion(img, structure=struct)
            skel |= img & ~eroded
            img = eroded
        return skel

def _ssim_numpy(img1, img2, win=7):
    from scipy.ndimage import uniform_filter
    c1, c2 = 0.01**2, 0.03**2
    ssims = []
    for ch in range(3):
        x = img1[:, :, ch].astype(np.float64)
        y = img2[:, :, ch].astype(np.float64)
        mu_x = uniform_filter(x, size=win)
        mu_y = uniform_filter(y, size=win)
        mu_x2 = mu_x**2; mu_y2 = mu_y**2; mu_xy = mu_x * mu_y
        sigma_x2 = uniform_filter(x*x, size=win) - mu_x2
        sigma_y2 = uniform_filter(y*y, size=win) - mu_y2
        sigma_xy = uniform_filter(x*y, size=win) - mu_xy
        ssim_map = ((2*mu_xy+c1)*(2*sigma_xy+c2)) / ((mu_x2+mu_y2+c1)*(sigma_x2+sigma_y2+c2))
        ssims.append(ssim_map.mean())
    return float(np.mean(ssims))

def _skel_iou(img1, img2, thresh=0.5):
    g1 = img1.mean(axis=2)
    g2 = img2.mean(axis=2)
    b1 = g1 < thresh
    b2 = g2 < thresh
    if not b1.any() and not b2.any(): return 1.0
    if not b1.any() or not b2.any(): return 0.0
    if _HAS_SKIMAGE:
        s1 = skeletonize(b1); s2 = skeletonize(b2)
    else:
        s1 = _skeletonize_fallback(b1); s2 = _skeletonize_fallback(b2)
    inter = (s1 & s2).sum(); union = (s1 | s2).sum()
    return float(inter / union) if union > 0 else 1.0

def build_model(args, device):
    from models import DiT_2Cond_models
    vae_downscale = getattr(args, "vae_downscale", 4)
    latent_size = args.image_size // vae_downscale
    latent_channels = int(getattr(args, "latent_channels", 4))
    model_cls = DiT_2Cond_models[args.model]
    model = model_cls(
        input_size=latent_size, in_channels=latent_channels,
        num_calligraphers=getattr(args, "num_calligraphers", 1011),
        num_characters=getattr(args, "num_characters", 7765),
        condition_fusion=getattr(args, "condition_fusion", "factorized_add"),
        callig_embed_dim=int(getattr(args, "callig_embed_dim", 128)),
        char_embed_dim=int(getattr(args, "char_embed_dim", 512)),
        learn_sigma=True,
        cond_drop_all_prob=float(getattr(args, "cond_drop_all_prob", 0.05)),
        cond_drop_one_prob=float(getattr(args, "cond_drop_one_prob", 0.25)),
        skel_head_enabled=getattr(args, "w_skel_head", 0) > 0,
        use_glyph_cond=getattr(args, "w_glyph_cond", False),
        glyph_scale_init=float(getattr(args, "glyph_scale_init", 0.4)),
    ).to(device).eval()
    return model

def load_vae(args, device):
    from diffusers.models import AutoencoderKL
    path = getattr(args, "vae_path", None)
    if path and os.path.exists(path):
        log(f"[vae] loading {path}")
        return AutoencoderKL.from_pretrained(path).to(device).eval()
    log(f"[vae] loading stabilityai/sd-vae-ft-{args.vae}")
    return AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device).eval()

def load_ckpt_weights(model, ckpt_path, use_ema=True):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if use_ema and "ema" in ckpt:
        weights = ckpt["ema"]
    elif "model" in ckpt:
        weights = ckpt["model"]
    else:
        weights = ckpt
    missing, unexpected = model.load_state_dict(weights, strict=False)
    log(f"[model] loaded {os.path.basename(ckpt_path)} (missing={len(missing)}, unexpected={len(unexpected)})")
    del ckpt
    return model

def inject_dino(model, args):
    emb_path = getattr(args, "char_dino_embeddings", None)
    idx_path = getattr(args, "char_dino_index", None)
    if not emb_path or not idx_path: return
    emb = np.load(emb_path)
    with open(idx_path, encoding="utf-8") as f:
        idx_data = json.load(f)
    glyphs = idx_data.get("glyphs", idx_data)
    table = model.y_char_embedder.embedding_table.weight
    NUM_CH = 7026
    injected = 0
    with torch.no_grad():
        for gi, (sid, cid) in enumerate(glyphs):
            gid = int(sid) * NUM_CH + int(cid)
            if 0 <= gid < table.shape[0] and gi < emb.shape[0]:
                e = emb[gi]
                e = e / (np.linalg.norm(e) + 1e-8)
                table.data[gid] = torch.from_numpy(e).float()
                injected += 1
    log(f"[dino-init] injected {injected} glyph embeddings")

def build_eval_cache(csv_path, data_dir, image_size, n, cond_mode="2cond",
                     use_glyph_cond=False):
    from dataset import MCCDDataset
    from torch.utils.data import DataLoader
    ds = MCCDDataset(csv_file=csv_path, root_dir=data_dir,
                     image_size=image_size, load_canny=False, load_skel=False,
                     use_glyph_cond=use_glyph_cond)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    conds, gts, g_all = [], [], []
    for b in loader:
        if len(conds) >= n: break
        for i in range(b["y_callig"].shape[0]):
            if cond_mode == "2cond":
                conds.append((b["y_callig"][i].item(), -1, b["y_char"][i].item()))
            else:
                conds.append((b["y_callig"][i].item(), b["y_script"][i].item(), b["y_char"][i].item()))
        gts.append(b["image"].cpu())
        if "g" in b and b["g"].numel() > 0:
            g_all.append(b["g"].cpu())
    gts = torch.cat(gts, dim=0)[:n]
    g_ret = torch.cat(g_all, dim=0)[:n] if g_all else None
    log(f"[cache] {csv_path} -> {len(conds)} samples")
    return {"conds": conds[:n], "gts": gts, "gs": g_ret}

def gpu_sample(model, vae, conds, gts, device, steps, cfg_scale, seed,
               cond_mode, latent_channels, latent_spatial, scaling_factor,
               batch_size=32, gs_batch=None):
    from diffusion import create_diffusion
    ddim = create_diffusion(str(steps))
    decoded_all, gt_all = [], []
    n = len(conds)
    torch.manual_seed(seed)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            bs = j - i
            noise = torch.randn(bs, latent_channels, latent_spatial, latent_spatial, device=device)
            if gs_batch is not None and gs_batch[i:j].shape[0] == bs:
                z = gs_batch[i:j].to(device).clone()
            else:
                z = noise
            if cond_mode == "2cond":
                mk = dict(
                    y_callig=torch.tensor([c[0] for c in conds[i:j]], device=device),
                    y_char=torch.tensor([c[2] for c in conds[i:j]], device=device),
                    cfg_scale=cfg_scale,
                )
            else:
                mk = dict(
                    y_callig=torch.tensor([c[0] for c in conds[i:j]], device=device),
                    y_script=torch.tensor([c[1] for c in conds[i:j]], device=device),
                    y_char=torch.tensor([c[2] for c in conds[i:j]], device=device),
                    cfg_scale=cfg_scale,
                )
            if gs_batch is not None and gs_batch[i:j].shape[0] == bs:
                mk["g"] = gs_batch[i:j].to(device)
            samples = ddim.ddim_sample_loop(model.forward_with_cfg, z.shape, z,
                                            clip_denoised=False, model_kwargs=mk, device=device)
            dec = vae.decode(samples / scaling_factor).sample
            decoded_all.append(dec.cpu())
            gt_all.append(gts[i:j].cpu())
            del z, samples, dec
            torch.cuda.empty_cache()
    return torch.cat(decoded_all, dim=0), torch.cat(gt_all, dim=0)

def save_images(decoded, gts, conds, out_dir, step):
    step_tag = f"step{int(step):07d}"
    step_dir = os.path.join(out_dir, step_tag)
    os.makedirs(step_dir, exist_ok=True)
    dec_np = ((decoded.clamp(-1, 1) + 1) / 2).numpy()
    gt_np = ((gts.clamp(-1, 1) + 1) / 2).numpy()
    n = decoded.shape[0]
    for i in range(n):
        Image.fromarray((dec_np[i].transpose(1,2,0) * 255).clip(0,255).astype("uint8")).save(
            os.path.join(step_dir, f"sample{i}.png"))
        Image.fromarray((gt_np[i].transpose(1,2,0) * 255).clip(0,255).astype("uint8")).save(
            os.path.join(step_dir, f"gt{i}.png"))
    with open(os.path.join(step_dir, "samples.json"), "w", encoding="utf-8") as f:
        json.dump({"step": step, "n": n, "conds": [list(c) for c in conds[:n]]}, f, ensure_ascii=False)

def compute_metrics(decoded, gts):
    n = decoded.shape[0]
    dec_np = ((decoded.clamp(-1, 1) + 1) / 2).numpy()
    gt_np = ((gts.clamp(-1, 1) + 1) / 2).numpy()
    results = []
    for i in range(n):
        pred = dec_np[i].transpose(1, 2, 0)
        gt = gt_np[i].transpose(1, 2, 0)
        mse = float(np.mean((pred - gt)**2))
        ssim = _ssim_numpy(pred, gt)
        skel_iou = _skel_iou(pred, gt)
        results.append({"idx": i, "mse": mse, "ssim": ssim, "skel_iou": skel_iou})
    return results

def _save_latest_thumb(decoded, gts, out_path):
    n = decoded.shape[0]
    h, w = decoded.shape[2], decoded.shape[3]
    canvas = np.zeros((h * 2, w * n, 3), dtype=np.uint8)
    dec_np = ((decoded.clamp(-1, 1) + 1) / 2).numpy()
    gt_np = ((gts.clamp(-1, 1) + 1) / 2).numpy()
    for i in range(n):
        canvas[:h, i*w:(i+1)*w] = (dec_np[i].transpose(1,2,0) * 255).clip(0,255).astype("uint8")
        canvas[h:, i*w:(i+1)*w] = (gt_np[i].transpose(1,2,0) * 255).clip(0,255).astype("uint8")
    Image.fromarray(canvas).save(out_path)

def find_max_batch(model, vae, conds, gts, device, steps, cfg_scale, seed,
                   cond_mode, lc, ls, sf, gs_batch):
    """逐级增大 batch 找最大不 OOM 的 batch size。"""
    from diffusion import create_diffusion
    ddim = create_diffusion(str(steps))
    torch.manual_seed(seed)
    max_bs = 0
    for bs in [32, 64, 96, 128, 192, 256, 384, 455]:
        if bs > len(conds):
            bs = len(conds)
        if bs <= max_bs:
            continue
        log(f"[batch-test] trying batch={bs}...")
        try:
            torch.cuda.reset_peak_memory_stats()
            noise = torch.randn(bs, lc, ls, ls, device=device)
            if cond_mode == "2cond":
                mk = dict(
                    y_callig=torch.tensor([c[0] for c in conds[:bs]], device=device),
                    y_char=torch.tensor([c[2] for c in conds[:bs]], device=device),
                    cfg_scale=cfg_scale,
                )
            else:
                mk = dict(
                    y_callig=torch.tensor([c[0] for c in conds[:bs]], device=device),
                    y_script=torch.tensor([c[1] for c in conds[:bs]], device=device),
                    y_char=torch.tensor([c[2] for c in conds[:bs]], device=device),
                    cfg_scale=cfg_scale,
                )
            with torch.no_grad():
                samples = ddim.ddim_sample_loop(model.forward_with_cfg, noise.shape, noise,
                                                clip_denoised=False, model_kwargs=mk, device=device)
                dec = vae.decode(samples / sf).sample
                peak = torch.cuda.max_memory_allocated() / 1e9
            log(f"  batch={bs}: OK, peak={peak:.2f}G")
            max_bs = bs
            del noise, samples, dec
            torch.cuda.empty_cache()
            if bs >= len(conds):
                break
        except torch.cuda.OutOfMemoryError:
            log(f"  batch={bs}: OOM!")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            log(f"  batch={bs}: error: {e}")
            torch.cuda.empty_cache()
            break
    log(f"[batch-test] max batch={max_bs}")
    return max_bs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--eval-csv", required=True)
    ap.add_argument("--show5-csv", default=None)
    ap.add_argument("--seen5-csv", default=None)
    ap.add_argument("--eval-n", type=int, default=455)
    ap.add_argument("--batch-size", type=int, default=0, help="0=auto find max")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--batch-test", action="store_true", help="only find max batch, don't eval")
    args = ap.parse_args()

    os.chdir(BASE)
    device = "cuda"
    log(f"[main] device={device}, eval_csv={args.eval_csv}, eval_n={args.eval_n}")

    marker = os.path.join(args.results_dir, "_active_ckpt_dir.txt")
    while not os.path.exists(marker):
        log("[watch] no active ckpt dir, waiting..."); time.sleep(10)
    with open(marker) as f:
        ckpt_dir = f.read().strip()
    if not os.path.isabs(ckpt_dir):
        ckpt_dir = os.path.join(BASE, ckpt_dir)
    log(f"[watch] ckpt dir: {ckpt_dir}")

    while True:
        cks = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt.done")))
        if cks: break
        log("[watch] no checkpoints, waiting..."); time.sleep(10)

    first_ckpt = cks[0].replace(".done", "")
    ckpt = torch.load(first_ckpt, map_location="cpu", weights_only=False)
    train_args = ckpt.get("args") or ckpt.get("config")
    del ckpt
    log(f"[main] loaded train args from {os.path.basename(first_ckpt)}")

    model = build_model(train_args, device)
    vae = load_vae(train_args, device)
    inject_dino(model, train_args)

    cond_mode = getattr(train_args, "cond_mode", "2cond")
    use_glyph_cond = getattr(train_args, "w_glyph_cond", False)
    eval_cache = build_eval_cache(args.eval_csv, getattr(train_args, "data_dir", ""),
                                  train_args.image_size, args.eval_n, cond_mode, use_glyph_cond)
    show5_cache = build_eval_cache(args.show5_csv, getattr(train_args, "data_dir", ""),
                                   train_args.image_size, 100, cond_mode, use_glyph_cond) if args.show5_csv else None
    seen5_cache = build_eval_cache(args.seen5_csv, getattr(train_args, "data_dir", ""),
                                   train_args.image_size, 100, cond_mode, use_glyph_cond) if args.seen5_csv else None

    lc = int(getattr(train_args, "latent_channels", 4))
    ls = int(train_args.image_size) // int(getattr(train_args, "vae_downscale", 8))
    sf = float(getattr(train_args, "vae_scaling_factor", 0.18215))

    # 加载第一个 ckpt 的权重 (用于 batch test)
    model = load_ckpt_weights(model, first_ckpt, use_ema=True)

    # batch 寻优
    if args.batch_size <= 0:
        batch_size = find_max_batch(model, vae, eval_cache["conds"], eval_cache["gts"],
                                    device, args.steps, args.cfg, args.seed, cond_mode,
                                    lc, ls, sf, eval_cache.get("gs"))
    else:
        batch_size = args.batch_size

    if args.batch_test:
        log("[batch-test] done, exiting")
        return

    log(f"[main] using batch_size={batch_size}")

    # 轮询
    done = set()
    state_path = os.path.join(ckpt_dir, "gpu_eval_state.json")
    try:
        done = set(json.load(open(state_path)).get("done", []))
    except Exception:
        pass

    while True:
        all_done = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt.done")))
        pending = [d.replace(".done", "") for d in all_done
                   if os.path.basename(d.replace(".done", "")) not in done]
        if not pending:
            log(f"[scan] {len(all_done)} ckpts, {len(done)} done, 0 pending — waiting")
            if args.once: break
            time.sleep(30); continue

        log(f"[scan] {len(all_done)} ckpts total, {len(done)} done, {len(pending)} pending")
        for ckpt_path in pending:
            ckpt_name = os.path.basename(ckpt_path)
            try:
                step = int(ckpt_name.replace(".pt", "").lstrip("0") or "0")
            except ValueError:
                step = 0
            try:
                t0 = time.time()
                model = load_ckpt_weights(model, ckpt_path, use_ema=True)
                log(f"[eval] === {ckpt_name} (step {step}) ===")

                decoded, gts = gpu_sample(model, vae, eval_cache["conds"], eval_cache["gts"],
                                          device, args.steps, args.cfg, args.seed, cond_mode,
                                          lc, ls, sf, batch_size, eval_cache.get("gs"))
                save_dir = os.path.join(ckpt_dir, "eval_samples")
                save_images(decoded, gts, eval_cache["conds"], save_dir, step)
                log(f"[eval] saved {decoded.shape[0]} images to eval_samples/step{step:07d}/")

                metrics = compute_metrics(decoded, gts)
                mses = [m["mse"] for m in metrics]
                ssims = [m["ssim"] for m in metrics]
                skels = [m["skel_iou"] for m in metrics]

                if show5_cache:
                    s5_dec, s5_gt = gpu_sample(model, vae, show5_cache["conds"], show5_cache["gts"],
                                                device, args.steps, args.cfg, args.seed, cond_mode,
                                                lc, ls, sf, 16, show5_cache.get("gs"))
                    save_images(s5_dec, s5_gt, show5_cache["conds"], save_dir, step)
                    _save_latest_thumb(s5_dec, s5_gt, os.path.join(ckpt_dir, "eval_latest.png"))
                    del s5_dec, s5_gt

                if seen5_cache:
                    sn5_dec, sn5_gt = gpu_sample(model, vae, seen5_cache["conds"], seen5_cache["gts"],
                                                  device, args.steps, args.cfg, args.seed, cond_mode,
                                                  lc, ls, sf, 16, seen5_cache.get("gs"))
                    save_images(sn5_dec, sn5_gt, seen5_cache["conds"],
                                os.path.join(ckpt_dir, "seen_samples"), step)
                    del sn5_dec, sn5_gt

                elapsed = time.time() - t0
                result = {
                    "step": step, "mse": float(np.mean(mses)), "ssim": float(np.mean(ssims)),
                    "skel_iou": float(np.mean(skels)),
                    "mse_std": float(np.std(mses)), "ssim_std": float(np.std(ssims)),
                    "skel_iou_std": float(np.std(skels)),
                    "ssim_min": float(np.min(ssims)), "skel_iou_min": float(np.min(skels)),
                    "n": len(metrics), "elapsed": elapsed, "batch": batch_size,
                }
                with open(os.path.join(ckpt_dir, f"eval_auto_{step:07d}.json"), "w") as f:
                    json.dump(result, f, indent=2)
                log(f"[eval] step {step}: MSE={result['mse']:.5f} SSIM={result['ssim']:.4f} "
                    f"SkelIoU={result['skel_iou']:.4f} ({elapsed:.0f}s)")

                del decoded, gts, metrics
                torch.cuda.empty_cache()
            except Exception as e:
                log(f"[eval] step {step}: FAILED: {e}")
                log(traceback.format_exc())

            done.add(ckpt_name)
            with open(state_path, "w") as f:
                json.dump({"done": sorted(done)}, f)

        if args.once: break

if __name__ == "__main__":
    main()
