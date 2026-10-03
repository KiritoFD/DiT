#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gpu_batch_eval_v2.py 鈥?楂樻晥 GPU 鎵归噺璇勬祴銆?
鏍稿績浼樺寲:
1. fp16 鎺ㄧ悊 (妯″瀷+VAE)
2. 鍏ㄩ儴 455 寮犲櫔澹?鏉′欢涓€娆℃€ф斁 GPU, 50 姝?DDIM 鍏ㄧ▼闆?CPU-GPU 浼犺緭
3. 鎵嬪啓 DDIM 寰幆 (涓嶇敤 ddim_sample_loop 灏佽, 娑堥櫎 Python 寮€閿€)
4. 50 姝ラ噰瀹屽悗鎵归噺 VAE decode
5. 鏈€鍚庝竴娆℃€?.cpu() 浼犲洖, 钀界洏 + CPU 绠楁寚鏍?6. CFG: 455鈫?10 forward, 鍒?2-3 澶?batch per step
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

# 鈹€鈹€ metrics 鈹€鈹€
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


def build_model(args, device):
    from src.model import DiT_2Cond_models
    ls = args.image_size // getattr(args, "vae_downscale", 4)
    lc = int(getattr(args, "latent_channels", 4))
    m = DiT_2Cond_models[args.model](
        input_size=ls, in_channels=lc,
        num_calligraphers=getattr(args,"num_calligraphers",1011),
        num_characters=getattr(args,"num_characters",7765),
        condition_fusion=getattr(args,"condition_fusion","factorized_add"),
        callig_embed_dim=int(getattr(args,"callig_embed_dim",128)),
        char_embed_dim=int(getattr(args,"char_embed_dim",512)),
        learn_sigma=True,
        cond_drop_all_prob=float(getattr(args,"cond_drop_all_prob",0.05)),
        cond_drop_one_prob=float(getattr(args,"cond_drop_one_prob",0.25)),
        use_glyph_cond=getattr(args,"w_glyph_cond",False),
        glyph_scale_init=float(getattr(args,"glyph_scale_init",0.4)),
    ).to(device).eval()
    return m.half(), ls, lc  # fp16

def load_vae(args, device):
    from diffusers.models import AutoencoderKL
    v = AutoencoderKL.from_pretrained(args.vae_path).to(device).eval()
    return v.half()  # fp16

def inject_dino(model, args):
    emb = np.load(args.char_dino_embeddings)
    glyphs = json.load(open(args.char_dino_index)).get("glyphs", [])
    table = model.y_char_embedder.embedding_table.weight
    NUM_CH = 7026; n = 0
    with torch.no_grad():
        for gi, (sid, cid) in enumerate(glyphs):
            gid = int(sid)*NUM_CH + int(cid)
            if 0 <= gid < table.shape[0] and gi < emb.shape[0]:
                e = emb[gi]; e = e/(np.linalg.norm(e)+1e-8)
                table.data[gid] = torch.from_numpy(e).float().half()
                n += 1
    log(f"[dino] injected {n}")

def build_cache(csv_path, data_dir, image_size, n, cond_mode, use_glyph_cond):
    from src.utils import MCCDDataset
    from torch.utils.data import DataLoader
    ds = MCCDDataset(csv_file=csv_path, root_dir=data_dir, image_size=image_size,
                     load_canny=False, load_skel=False, use_glyph_cond=use_glyph_cond)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    conds, gts = [], []
    for b in loader:
        if len(conds) >= n: break
        for i in range(b["y_callig"].shape[0]):
            if cond_mode == "2cond":
                conds.append((b["y_callig"][i].item(), -1, b["y_char"][i].item()))
            else:
                conds.append((b["y_callig"][i].item(), b["y_script"][i].item(), b["y_char"][i].item()))
        gts.append(b["image"].cpu())
    gts = torch.cat(gts, dim=0)[:n]
    log(f"[cache] {csv_path} -> {len(conds)} samples")
    return {"conds": conds[:n], "gts": gts}


def ddim_sample_all(model, conds, device, n_steps, cfg_scale, seed,
                    lc, ls, fwd_batch=256):
    """鎵嬪啓 DDIM: 鍏ㄩ儴鏍锋湰鍚屾椂鍦?GPU 涓? 鍒嗗ぇ batch per step, 闆?CPU-GPU 浼犺緭銆?    杩斿洖 fp32 latent (n, lc, ls, ls) on CPU."""
    from src.loss import create_diffusion
    diffusion = create_diffusion(str(n_steps))
    n = len(conds)
    torch.manual_seed(seed)

    # 鍏ㄩ儴鍣０涓€娆℃€ф斁 GPU (fp16)
    z = torch.randn(n, lc, ls, ls, device=device, dtype=torch.float16)

    # 棰勮绠楁墍鏈夋潯浠?tensor (fp16)
    y_callig = torch.tensor([c[0] for c in conds], device=device, dtype=torch.long)
    y_char   = torch.tensor([c[2] for c in conds], device=device, dtype=torch.long)
    n_classes_callig = model.y_callig_embedder.num_classes
    n_classes_char   = model.y_char_embedder.num_classes

    # DDIM timesteps (浠庡ぇ鍒板皬)
    timesteps = list(range(n_steps - 1, -1, -1))  # [49, 48, ..., 0]

    with torch.no_grad():
        for ti, t_val in enumerate(timesteps):
            t = torch.full((n,), t_val, device=device, dtype=torch.long)
            # 鍒嗗ぇ batch 鍋?forward_with_cfg (鍐呴儴浼?cat cond+uncond 鈫?2脳batch)
            outputs = []
            for s in range(0, n, fwd_batch):
                e = min(s + fwd_batch, n)
                bs = e - s
                z_b = z[s:e]
                yc_b = y_callig[s:e]
                ych_b = y_char[s:e]
                t_b = t[s:e]
                # 鎵嬪姩 CFG: cat cond + uncond
                z2 = torch.cat([z_b, z_b], 0)
                t2 = torch.cat([t_b, t_b], 0)
                yc2 = torch.cat([yc_b, torch.full_like(yc_b, n_classes_callig)], 0)
                ych2 = torch.cat([ych_b, torch.full_like(ych_b, n_classes_char)], 0)
                out = model(z2, t2, yc2, ych2)
                # out: (2*bs, out_channels, ls, ls)
                eps, _rest = out[:, :lc], out[:, lc:]
                cond_eps, uncond_eps = eps[:bs], eps[bs:]
                half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
                outputs.append(half_eps)

            eps_all = torch.cat(outputs, 0)  # (n, lc, ls, ls)

            # DDIM update step (鍏ㄩ儴鏍锋湰鍚屾椂)
            alpha_bar_t = diffusion.alphas_cumprod[t_val]
            alpha_bar_prev = diffusion.alphas_cumprod[timesteps[ti+1]] if ti < len(timesteps)-1 else torch.tensor(1.0, device=device)
            # DDIM deterministic step:
            # x0_pred = (x_t - sqrt(1-alpha_bar_t) * eps) / sqrt(alpha_bar_t)
            x0_pred = (z - (1 - alpha_bar_t).sqrt() * eps_all) / alpha_bar_t.sqrt()
            # x_{t-1} = sqrt(alpha_bar_prev) * x0_pred + sqrt(1 - alpha_bar_prev) * eps
            z = alpha_bar_prev.sqrt() * x0_pred + (1 - alpha_bar_prev).sqrt() * eps_all

    return z.float().cpu()


def eval_one_ckpt(model, vae, ckpt_path, step, ckpt_dir,
                  eval_cache, show5_cache, seen5_cache, device,
                  lc, ls, sf, fwd_batch, steps, cfg, seed):
    t0 = time.time()
    # 鍔犺浇鏉冮噸
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["ema"], strict=False)
    del ckpt
    model.half()  # 纭繚 fp16

    # 鈹€鈹€ eval: 455 寮犲叏閲?DDIM 鈹€鈹€
    log(f"[eval] === step {step} ===")
    latents = ddim_sample_all(model, eval_cache["conds"], device, steps, cfg, seed,
                               lc, ls, fwd_batch)
    # VAE decode (鍒?batch)
    decoded_all = []
    with torch.no_grad():
        for i in range(0, latents.shape[0], fwd_batch):
            j = min(i + fwd_batch, latents.shape[0])
            lat_b = latents[i:j].to(device).half() / sf
            dec = vae.decode(lat_b).sample
            decoded_all.append(dec.float().cpu())
    decoded = torch.cat(decoded_all, 0)  # (n,3,256,256) [-1,1]
    gts = eval_cache["gts"]

    # 钀界洏
    save_dir = os.path.join(ckpt_dir, "eval_samples")
    _save_images(decoded, gts, eval_cache["conds"], save_dir, step)

    # 鎸囨爣
    metrics = _compute_metrics(decoded, gts)
    mses = [m["mse"] for m in metrics]
    ssims = [m["ssim"] for m in metrics]
    skels = [m["skel_iou"] for m in metrics]

    # show5
    if show5_cache:
        s5_dec = _sample_and_decode(model, vae, show5_cache, device, lc, ls, sf,
                                     fwd_batch, steps, cfg, seed)
        _save_images(s5_dec, show5_cache["gts"], show5_cache["conds"], save_dir, step)
        _save_thumb(s5_dec, show5_cache["gts"], os.path.join(ckpt_dir, "eval_latest.png"))
        del s5_dec

    # seen5
    if seen5_cache:
        sn5_dec = _sample_and_decode(model, vae, seen5_cache, device, lc, ls, sf,
                                      fwd_batch, steps, cfg, seed)
        _save_images(sn5_dec, seen5_cache["gts"], seen5_cache["conds"],
                     os.path.join(ckpt_dir, "seen_samples"), step)
        del sn5_dec

    elapsed = time.time() - t0
    result = {
        "step": step, "mse": float(np.mean(mses)), "ssim": float(np.mean(ssims)),
        "skel_iou": float(np.mean(skels)),
        "mse_std": float(np.std(mses)), "ssim_std": float(np.std(ssims)),
        "skel_iou_std": float(np.std(skels)),
        "ssim_min": float(np.min(ssims)), "skel_iou_min": float(np.min(skels)),
        "n": len(metrics), "elapsed": elapsed, "batch": fwd_batch, "fp16": True,
    }
    with open(os.path.join(ckpt_dir, f"eval_auto_{step:07d}.json"), "w") as f:
        json.dump(result, f, indent=2)
    log(f"[eval] step {step}: MSE={result['mse']:.5f} SSIM={result['ssim']:.4f} "
        f"SkelIoU={result['skel_iou']:.4f} ({elapsed:.0f}s, fp16, fwd_batch={fwd_batch})")
    torch.cuda.empty_cache()
    return result


def _sample_and_decode(model, vae, cache, device, lc, ls, sf, fwd_batch, steps, cfg, seed):
    latents = ddim_sample_all(model, cache["conds"], device, steps, cfg, seed, lc, ls, fwd_batch)
    decoded = []
    with torch.no_grad():
        for i in range(0, latents.shape[0], fwd_batch):
            j = min(i + fwd_batch, latents.shape[0])
            dec = vae.decode(latents[i:j].to(device).half() / sf).sample
            decoded.append(dec.float().cpu())
    return torch.cat(decoded, 0)


def _save_images(decoded, gts, conds, out_dir, step):
    tag = f"step{int(step):07d}"
    d = os.path.join(out_dir, tag)
    os.makedirs(d, exist_ok=True)
    dec_np = ((decoded.clamp(-1,1)+1)/2).numpy()
    gt_np  = ((gts.clamp(-1,1)+1)/2).numpy()
    n = decoded.shape[0]
    for i in range(n):
        Image.fromarray((dec_np[i].transpose(1,2,0)*255).clip(0,255).astype("uint8")).save(os.path.join(d, f"sample{i}.png"))
        Image.fromarray((gt_np[i].transpose(1,2,0)*255).clip(0,255).astype("uint8")).save(os.path.join(d, f"gt{i}.png"))
    with open(os.path.join(d, "samples.json"), "w", encoding="utf-8") as f:
        json.dump({"step": step, "n": n, "conds": [list(c) for c in conds[:n]]}, f, ensure_ascii=False)


def _save_thumb(decoded, gts, path):
    n = decoded.shape[0]; h, w = decoded.shape[2:]
    c = np.zeros((h*2, w*n, 3), dtype=np.uint8)
    dn = ((decoded.clamp(-1,1)+1)/2).numpy()
    gn = ((gts.clamp(-1,1)+1)/2).numpy()
    for i in range(n):
        c[:h, i*w:(i+1)*w] = (dn[i].transpose(1,2,0)*255).clip(0,255).astype("uint8")
        c[h:, i*w:(i+1)*w] = (gn[i].transpose(1,2,0)*255).clip(0,255).astype("uint8")
    Image.fromarray(c).save(path)


def _compute_metrics(decoded, gts):
    n = decoded.shape[0]
    dn = ((decoded.clamp(-1,1)+1)/2).numpy()
    gn = ((gts.clamp(-1,1)+1)/2).numpy()
    results = []
    for i in range(n):
        p = dn[i].transpose(1,2,0); g = gn[i].transpose(1,2,0)
        results.append({"idx": i, "mse": float(np.mean((p-g)**2)),
                         "ssim": _ssim_np(p, g), "skel_iou": _skel_iou(p, g)})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--eval-csv", required=True)
    ap.add_argument("--show5-csv", default=None)
    ap.add_argument("--seen5-csv", default=None)
    ap.add_argument("--eval-n", type=int, default=455)
    ap.add_argument("--fwd-batch", type=int, default=256, help="DiT forward batch per DDIM step")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--batch-test", action="store_true")
    args = ap.parse_args()

    os.chdir(BASE)
    device = "cuda"
    log(f"[main] device={device} fp16=True, eval_n={args.eval_n}, fwd_batch={args.fwd_batch}")

    # 鎵?ckpt dir
    marker = os.path.join(args.results_dir, "_active_ckpt_dir.txt")
    while not os.path.exists(marker):
        log("[watch] waiting for active ckpt dir..."); time.sleep(10)
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
    train_args = ckpt["args"]
    del ckpt
    log(f"[main] loaded train args from {os.path.basename(first_ckpt)}")

    model, ls, lc = build_model(train_args, device)
    vae = load_vae(train_args, device)
    inject_dino(model, train_args)
    sf = float(getattr(train_args, "vae_scaling_factor", 0.102079))

    cond_mode = getattr(train_args, "cond_mode", "2cond")
    use_gc = getattr(train_args, "w_glyph_cond", False)
    eval_cache = build_cache(args.eval_csv, getattr(train_args, "data_dir", ""),
                             train_args.image_size, args.eval_n, cond_mode, use_gc)
    show5_cache = build_cache(args.show5_csv, getattr(train_args, "data_dir", ""),
                              train_args.image_size, 100, cond_mode, use_gc) if args.show5_csv else None
    seen5_cache = build_cache(args.seen5_csv, getattr(train_args, "data_dir", ""),
                               train_args.image_size, 100, cond_mode, use_gc) if args.seen5_csv else None

    # batch test 妯″紡
    if args.batch_test:
        ckpt = torch.load(first_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["ema"], strict=False); del ckpt; model.half()
        for bs in [64, 128, 256, 384, 455]:
            if bs > args.eval_n: bs = args.eval_n
            log(f"[batch-test] fwd_batch={bs} fp16...")
            try:
                torch.cuda.reset_peak_memory_stats()
                latents = ddim_sample_all(model, eval_cache["conds"][:bs], device,
                                           args.steps, args.cfg, args.seed, lc, ls, bs)
                peak = torch.cuda.max_memory_allocated() / 1e9
                log(f"  fwd_batch={bs}: OK! peak={peak:.2f}G")
                if bs >= args.eval_n:
                    break
            except torch.cuda.OutOfMemoryError:
                log(f"  fwd_batch={bs}: OOM!"); torch.cuda.empty_cache(); break
            except Exception as e:
                log(f"  fwd_batch={bs}: error: {e}"); torch.cuda.empty_cache(); break
        return

    # 杞
    done = set()
    state_path = os.path.join(ckpt_dir, "gpu_eval_state.json")
    try:
        done = set(json.load(open(state_path)).get("done", []))
    except Exception:
        pass

    while True:
        all_done = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt.done")))
        pending = [d.replace(".done","") for d in all_done
                   if os.path.basename(d.replace(".done","")) not in done]
        if not pending:
            log(f"[scan] {len(all_done)} ckpts, {len(done)} done, 0 pending 鈥?waiting")
            if args.once: break
            time.sleep(30); continue

        log(f"[scan] {len(all_done)} total, {len(done)} done, {len(pending)} pending")
        for ckpt_path in pending:
            ckpt_name = os.path.basename(ckpt_path)
            try:
                step = int(ckpt_name.replace(".pt","").lstrip("0") or "0")
            except ValueError:
                step = 0
            try:
                eval_one_ckpt(model, vae, ckpt_path, step, ckpt_dir,
                              eval_cache, show5_cache, seen5_cache, device,
                              lc, ls, sf, args.fwd_batch, args.steps, args.cfg, args.seed)
            except Exception as e:
                log(f"[eval] step {step}: FAILED: {e}")
                log(traceback.format_exc())
            done.add(ckpt_name)
            with open(state_path, "w") as f:
                json.dump({"done": sorted(done)}, f)
        if args.once: break

if __name__ == "__main__":
    main()
