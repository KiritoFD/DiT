# -*- coding: utf-8 -*-
"""离线安全增强 + VAE 重编码 — 生成增强后的 latent shards.

书法安全的增强（保持字形身份不变）：
  1. 缩小为主：scale ∈ [0.85, 0.97]，缩小后居中回填到 256×256（只缩不放，四周留白）
  2. 放大用 bbox 保护的中心裁切：先检测笔画 bbox，放大后从 bbox 中心裁切，
     保证 bbox 完整不切笔画（裁切窗 = 放大画布中央，但被 bbox 边界钳制）
  3. 笔触粗细扰动 ±1px（dilate/erode，书法专用：不同毛笔/书家核心差异）
  4. 小幅旋转 ±6°（书法运笔自然倾斜）
  5. 对比度 0.85–1.15 + 亮度 ±0.03（墨色浓淡/纸色差异）

明确不做：水平/垂直翻转（改变字形方向=换字）、>15° 旋转、反相、cutout。

输出: 与 final_latents/ 相同格式的 shard_XXXXX.npz（latents + img_ids），
      latent 用与 encode_latents_klf4.py 相同的 kl-f4 VAE + scaling_factor。

用法 (远程):
  python tools/augment_encode_latents.py --csv 5script/train_3top30_nobeike.csv \
    --img-root final_imgs_256 --vae pretrained_models/sd-vae-ft-ema \
    --out final_latents_aug --shard-size 5000 --variants 4 --seed 42
"""
import os, sys, csv, json, glob, time, re, argparse, math
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy import ndimage


def load_image_ids(csv_path):
    ids = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel = row.get("image_path", "")
            m = re.search(r"(\d+)\.png", rel)
            if m:
                ids.append(int(m.group(1)))
    return sorted(set(ids))


def detect_ink_bbox(arr):
    """arr: (H,W) float [0,1]. 返回笔画 bbox (y0,y1,x0,x1) 或 None."""
    mask = arr < 0.5  # 墨色 < 0.5 视为笔画
    if mask.sum() < 16:
        return None
    ys, xs = np.where(mask)
    return (ys.min(), ys.max(), xs.min(), xs.max())


def augment_once(img, rng, mode="shrink"):
    """对 256×256 灰度 PIL 图做一次安全增强. 返回增强后的 256×256 PIL 图."""
    W, H = img.size  # 256
    arr0 = np.asarray(img.convert("L"), dtype=np.float32) / 255.0

    if mode == "shrink":
        # ── 缩小: scale ∈ [0.85, 0.97]，居中回填 ──
        scale = rng.uniform(0.85, 0.97)
        new_w = max(8, int(W * scale))
        new_h = max(8, int(H * scale))
        small = img.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("L", (W, H), 255)  # 白底
        ox = (W - new_w) // 2 + rng.randint(-3, 4)
        oy = (H - new_h) // 2 + rng.randint(-3, 4)
        ox = max(0, min(ox, W - new_w))
        oy = max(0, min(oy, H - new_h))
        canvas.paste(small, (ox, oy))
        return canvas

    elif mode == "crop":
        # ── 放大: scale ∈ [1.03, 1.10]，bbox 保护中心裁切 ──
        scale = rng.uniform(1.03, 1.10)
        new_w = int(W * scale)
        new_h = int(H * scale)
        big = img.resize((new_w, new_h), Image.LANCZOS)
        bbox = detect_ink_bbox(arr0)
        # 放大后 bbox 同样缩放
        if bbox is not None:
            by0, by1, bx0, bx1 = [int(v * scale) for v in bbox]
            # 裁切窗中心 = bbox 中心，但钳制在画布内且不越出 bbox
            cx = (bx0 + bx1) // 2
            cy = (by0 + by1) // 2
        else:
            cx, cy = new_w // 2, new_h // 2
        # 裁 256 窗口，中心 (cx, cy)，钳制边界保证 bbox 完整
        half = W // 2
        x0 = cx - half
        if x0 < 0:
            x0 = 0
        if x0 + W > new_w:
            x0 = new_w - W
        y0 = cy - half
        if y0 < 0:
            y0 = 0
        if y0 + H > new_h:
            y0 = new_h - H
        return big.crop((x0, y0, x0 + W, y0 + H))

    elif mode == "thickness":
        # ── 笔触粗细扰动: 腐蚀/膨胀 ±1px（墨白颠倒后处理） ──
        arr = arr0.copy()
        op = rng.choice([-1, 0, 1])
        if op != 0:
            mask = arr < 0.5  # 笔画
            struct = np.ones((3, 3))
            if op < 0:
                # 细：膨胀墨区边缘（笔画区域收缩）→ 对墨区做腐蚀
                new_mask = ndimage.binary_erosion(mask, struct, iterations=1)
            else:
                # 粗：墨区膨胀
                new_mask = ndimage.binary_dilation(mask, struct, iterations=1)
            arr = np.where(new_mask, arr, 1.0)  # 非笔画区置白
            # 笔画区保持原灰度（轻微）
        out = (arr * 255).astype(np.uint8)
        return Image.fromarray(out)

    elif mode == "rotate":
        # ── 小幅旋转 ±6°，reflection 填充 ──
        angle = rng.uniform(-6, 6)
        return img.rotate(angle, resample=Image.BICUBIC, fillcolor=255)

    elif mode == "contrast":
        # ── 对比度 0.85–1.15 + 亮度 ±0.03 ──
        c = rng.uniform(0.85, 1.15)
        b = rng.uniform(-0.03, 0.03)
        arr = arr0 * c + b
        arr = np.clip(arr, 0, 1)
        out = (arr * 255).astype(np.uint8)
        return Image.fromarray(out)

    raise ValueError(f"unknown mode {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--img-root", default="final_images")
    ap.add_argument("--vae", default="pretrained_models/kl-f4",
                    help="VAE dir with diffusion_pytorch_model (kl-f4 or sd-vae-ft-ema)")
    ap.add_argument("--out", default="final_latents_aug")
    ap.add_argument("--shard-size", type=int, default=5000)
    ap.add_argument("--scaling-factor", type=float, default=0.102079)
    ap.add_argument("--variants", type=int, default=4,
                    help="增强变体数/样本 (1=只原图, 4=原图+3增强)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--latent-channels", type=int, default=4)
    ap.add_argument("--latent-size", type=int, default=32)
    ap.add_argument("--downscale", type=int, default=4)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    # ── Load VAE ──
    print(f"Loading VAE from {args.vae}...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # ── Images ──
    img_ids = load_image_ids(args.csv)
    print(f"Images to augment: {len(img_ids)} x {args.variants} variants")
    total_out = len(img_ids) * args.variants
    n_shards = (total_out + args.shard_size - 1) // args.shard_size
    print(f"Output latents: {total_out}, shards: {n_shards}")

    modes = ["shrink", "crop", "thickness", "rotate", "contrast"]
    shard_lat = []
    shard_ids = []
    shard_idx = 0
    t0 = time.time()
    n_done = 0
    vae_bs = 32

    for n, iid in enumerate(img_ids):
        img_path = os.path.join(args.img_root, f"{iid}.png")
        if not os.path.exists(img_path):
            print(f"  [skip] missing {img_path}")
            continue
        img = Image.open(img_path).convert("L").resize((256, 256), Image.LANCZOS)

        variants = [img]  # 原图
        for v in range(args.variants - 1):
            mode = modes[v % len(modes)]
            variants.append(augment_once(img, rng, mode))

        # 批量 encode
        tensors = []
        for v in variants:
            a = np.asarray(v, dtype=np.float32) / 127.5 - 1.0  # [-1,1]
            a = np.stack([a, a, a], axis=-1)  # (256,256,3) 灰度三通道
            tensors.append(torch.from_numpy(a).permute(2, 0, 1))
        xs = torch.stack(tensors).to(device)
        with torch.no_grad():
            for i in range(0, len(xs), vae_bs):
                chunk = xs[i:i + vae_bs]
                lat = vae.encode(chunk).latent_dist.sample()
                lat = lat * args.scaling_factor
                # 按需 resize latent 到 latent_size
                if lat.shape[-1] != args.latent_size:
                    lat = torch.nn.functional.interpolate(lat, size=args.latent_size, mode="bilinear")
                lat = lat.float().cpu().numpy()
                shard_lat.append(lat)
                shard_ids.append(np.full(len(chunk), iid, dtype=np.int64))

        n_done += 1
        if len(shard_lat) >= args.shard_size or (n + 1) == len(img_ids):
            lat_cat = np.concatenate(shard_lat, axis=0).astype(np.float16)
            ids_cat = np.concatenate(shard_ids, axis=0)
            out_path = os.path.join(args.out, f"shard_{shard_idx:05d}.npz")
            np.savez_compressed(out_path, latents=lat_cat, img_ids=ids_cat)
            print(f"  [{shard_idx+1}/{n_shards}] {out_path}: {lat_cat.shape} "
                  f"({(time.time()-t0):.0f}s, {n_done} imgs)")
            shard_lat = []
            shard_ids = []
            shard_idx += 1
            # GPU 内存抖动小，无需手动释放

        if (n + 1) % 2000 == 0:
            print(f"  ... {n+1}/{len(img_ids)} imgs, {(time.time()-t0)/60:.1f}min")

    print(f"\nDone. {n_shards} shards -> {args.out}")


if __name__ == "__main__":
    main()
