#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全量 VAE 本底噪声监测 (多进程版)。
8 进程并行, 每进程 8 核, 各处理 1/8 数据。
只算 MSE (全量 106k), SSIM 单独采样 1000 张。
4 组: f4_p4, f4_p2, f8_p4, f8_p2 (patch 不影响 VAE 重建, 但跑全 4 组验证)。
"""
import os, sys, csv, json, time, gc, math
import torch
import numpy as np
from PIL import Image
from multiprocessing import Process, Queue

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "/root/Workspace/xy/DiT"
CSV_PATH = os.path.join(BASE, "5script", "train_top30_clean.csv")
IMG_ROOT = BASE
OUT_DIR = os.path.join(BASE, "tools", "vae_noise_results")
os.makedirs(OUT_DIR, exist_ok=True)

VAE_CONFIGS = [
    ("f4_kl-f4_p4", {"path": os.path.join(BASE, "pretrained_models", "kl-f4"),
                     "downscale": 4, "latent_channels": 3, "scaling_factor": 0.102079, "patch": 4}),
    ("f4_kl-f4_p2", {"path": os.path.join(BASE, "pretrained_models", "kl-f4"),
                     "downscale": 4, "latent_channels": 3, "scaling_factor": 0.102079, "patch": 2}),
    ("f8_sd-vae_p4", {"path": os.path.join(BASE, "pretrained_models", "sd-vae-ft-ema"),
                      "downscale": 8, "latent_channels": 4, "scaling_factor": 0.18215, "patch": 4}),
    ("f8_sd-vae_p2", {"path": os.path.join(BASE, "pretrained_models", "sd-vae-ft-ema"),
                      "downscale": 8, "latent_channels": 4, "scaling_factor": 0.18215, "patch": 2}),
]

N_WORKERS = 8
THREADS_PER_WORKER = 8
BATCH_SIZE = 32  # 每进程 batch


def load_image_paths():
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
    paths = []
    for r in rows:
        p = r["image_path"]
        if p.startswith("final_images/"):
            p = p.replace("final_images/", "final_imgs_256/", 1)
        full = os.path.join(IMG_ROOT, p)
        if os.path.isfile(full):
            paths.append(full)
    return paths


def worker_fn(worker_id, paths, vae_path, result_queue):
    """每个 worker 处理 paths 子集, 返回 MSE 列表。"""
    import torch
    torch.set_num_threads(THREADS_PER_WORKER)
    from diffusers.models import AutoencoderKL
    from PIL import Image
    import numpy as np

    device = "cpu"
    vae = AutoencoderKL.from_pretrained(vae_path).to(device).eval()

    mses = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(paths), BATCH_SIZE):
            batch = paths[i:i + BATCH_SIZE]
            imgs = []
            for p in batch:
                img = Image.open(p).convert("RGB").resize((256, 256), Image.LANCZOS)
                imgs.append(np.asarray(img, dtype=np.float32) / 255.0)
            if not imgs:
                continue
            arr = np.stack(imgs)
            x = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)
            latent = vae.encode(x).latent_dist.sample()
            decoded = vae.decode(latent).sample.clamp(0, 1)
            x_np = x.numpy()
            dec_np = decoded.cpu().numpy()
            for j in range(dec_np.shape[0]):
                mse = float(np.mean((x_np[j] - dec_np[j]) ** 2))
                mses.append(mse)
            del x, latent, decoded, arr
            gc.collect()
            if (len(mses) % 128 == 0) and worker_id == 0:
                elapsed = time.time() - t0
                rate = len(mses) / elapsed
                print(f"  [worker0] {len(mses)}/{len(paths)} ({rate:.1f} img/s)", flush=True)
    result_queue.put((worker_id, mses))


def evaluate_config(name, cfg, all_paths):
    """8 进程并行跑一个 VAE 配置。"""
    from diffusers.models import AutoencoderKL
    print(f"\n{'='*60}")
    print(f"=== {name} (ds={cfg['downscale']}, ch={cfg['latent_channels']}, "
          f"scale={cfg['scaling_factor']}, patch={cfg['patch']}) ===")

    # latent shape
    vae_tmp = AutoencoderKL.from_pretrained(cfg["path"])
    with torch.no_grad():
        dummy = torch.randn(1, 3, 256, 256)
        latent = vae_tmp.encode(dummy).latent_dist.sample()
        latent_shape = tuple(latent.shape[1:])
        h, w = latent_shape[1], latent_shape[2]
        tokens = (h // 2) * (w // 2) if cfg["patch"] == 2 else (h // 4) * (w // 4)
    del vae_tmp
    print(f"  latent: {latent_shape}, patch{cfg['patch']} → {tokens} tokens")
    print(f"  images: {len(all_paths)}, workers: {N_WORKERS}x{THREADS_PER_WORKER} threads")

    # 分割路径
    chunk = (len(all_paths) + N_WORKERS - 1) // N_WORKERS
    chunks = [all_paths[i*chunk:(i+1)*chunk] for i in range(N_WORKERS)]

    q = Queue()
    procs = []
    t0 = time.time()
    for wid in range(N_WORKERS):
        p = Process(target=worker_fn, args=(wid, chunks[wid], cfg["path"], q))
        p.start()
        procs.append(p)

    all_mse = []
    for _ in range(N_WORKERS):
        wid, mses = q.get()
        all_mse.extend(mses)
        print(f"  [worker {wid}] done, {len(mses)} imgs", flush=True)

    for p in procs:
        p.join()

    elapsed = time.time() - t0
    all_mse.sort()
    summary = {
        "name": name, "downscale": cfg["downscale"],
        "latent_channels": cfg["latent_channels"],
        "scaling_factor": cfg["scaling_factor"], "patch": cfg["patch"],
        "latent_shape": latent_shape, "tokens": tokens,
        "n_images": len(all_mse),
        "mse_mean": float(np.mean(all_mse)),
        "mse_std": float(np.std(all_mse)),
        "mse_median": float(np.median(all_mse)),
        "mse_p95": float(np.percentile(all_mse, 95)),
        "mse_max": float(np.max(all_mse)),
        "elapsed_sec": elapsed,
        "rate": len(all_mse) / elapsed,
    }
    print(f"  MSE: mean={summary['mse_mean']:.6f} std={summary['mse_std']:.6f} "
          f"median={summary['mse_median']:.6f} p95={summary['mse_p95']:.6f} "
          f"max={summary['mse_max']:.6f}")
    print(f"  time: {elapsed:.0f}s ({summary['rate']:.1f} img/s)")

    # SSIM 采样 1000 张 (单进程)
    print(f"  computing SSIM on 1000 samples...")
    from scipy.ndimage import uniform_filter
    def compute_ssim(img1, img2):
        c1, c2 = 0.01**2, 0.03**2
        win = 7
        ssims = []
        for ch in range(3):
            x = img1[:, :, ch].astype(np.float64)
            y = img2[:, :, ch].astype(np.float64)
            mu_x = uniform_filter(x, size=win)
            mu_y = uniform_filter(y, size=win)
            mu_x2 = mu_x**2; mu_y2 = mu_y**2; mu_xy = mu_x*mu_y
            sigma_x2 = uniform_filter(x*x, size=win) - mu_x2
            sigma_y2 = uniform_filter(y*y, size=win) - mu_y2
            sigma_xy = uniform_filter(x*y, size=win) - mu_xy
            ssim_map = ((2*mu_xy+c1)*(2*sigma_xy+c2)) / ((mu_x2+mu_y2+c1)*(sigma_x2+sigma_y2+c2))
            ssims.append(ssim_map.mean())
        return float(np.mean(ssims))

    vae = AutoencoderKL.from_pretrained(cfg["path"]).to("cpu").eval()
    torch.set_num_threads(32)
    ssim_vals = []
    sample_paths = all_paths[:1000]
    with torch.no_grad():
        for i, p in enumerate(sample_paths):
            img = Image.open(p).convert("RGB").resize((256, 256), Image.LANCZOS)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            lat = vae.encode(x).latent_dist.sample()
            dec = vae.decode(lat).sample.clamp(0, 1).cpu().numpy()[0]
            orig = arr
            rec = dec.transpose(1, 2, 0)
            ssim_vals.append(compute_ssim(orig, rec))
            if (i+1) % 200 == 0:
                print(f"    SSIM {i+1}/1000...", flush=True)
    summary["ssim_mean"] = float(np.mean(ssim_vals))
    summary["ssim_std"] = float(np.std(ssim_vals))
    summary["ssim_n"] = len(ssim_vals)
    print(f"  SSIM (n={len(ssim_vals)}): mean={summary['ssim_mean']:.4f} "
          f"std={summary['ssim_std']:.4f}")

    # 保存
    with open(os.path.join(OUT_DIR, f"summary_{name}.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    del vae
    gc.collect()
    return summary


def main():
    paths = load_image_paths()
    print(f"=== VAE 本底噪声监测 (多进程, {len(paths)} 张) ===")
    print(f"Workers: {N_WORKERS} x {THREADS_PER_WORKER} threads = {N_WORKERS*THREADS_PER_WORKER} cores")

    summaries = []
    for name, cfg in VAE_CONFIGS:
        s = evaluate_config(name, cfg, paths)
        summaries.append(s)

    # 汇总
    print(f"\n{'='*60}")
    print(f"=== 汇总 ===")
    print(f"{'配置':<20} {'VAE':<6} {'patch':<5} {'tokens':<7} "
          f"{'MSE':<12} {'SSIM(1k)':<10}")
    for s in summaries:
        vae_n = "f4" if s["downscale"] == 4 else "f8"
        print(f"{s['name']:<20} {vae_n:<6} {s['patch']:<5} {s['tokens']:<7} "
              f"{s['mse_mean']:.6f}    {s['ssim_mean']:.4f}")
    with open(os.path.join(OUT_DIR, "vae_noise_summary.json"), "w") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {OUT_DIR}/vae_noise_summary.json")


if __name__ == "__main__":
    main()
