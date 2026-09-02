#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""3-30-6 增强数据集构建（工业级书法安全增强）→ mid-clean 数据集.

流程:
  Phase A (CPU, 多进程): 对 (script,char,calli) 组合中样本数 <6 的组合,
     生成 (6-n) 个增强变体 → 保存 PNG 到 final_imgs_mid_clean/{new_id}.png,
     同时写 aug_meta.csv (new_id,script,char,calli,calli_id,char_id,glyph_id,script_id).
  Phase B (GPU): 读取 final_imgs_mid_clean/*.png, sd-vae-ft-ema 编码 (scaling=0.18215)
     → 临时 latent shard /tmp/mid_clean_tmp/.
  Phase C (merge): 合并原 latent (只保留清洗后 csv 的 img_id) + 增强 latent → final_latents_mid_clean/,
     写 train_mid_clean.csv (清洗后原行 + 增强行).

输入: 5script/train_3top30_common.csv (GB2312 一级+二级 常用字, 23597 行)
增强算子 (每个变体随机组合):
  1. 弹性形变 Elastic (scipy map_coordinates, 弱: alpha~4-8, sigma~3)
  2. 微仿射: 平移 ±2-5%, 各向同性缩放 ±5%, 微旋转 ±1.5°
  3. 阈值/Gamma 抖动: 模拟墨量变化 (晕染/飞白)
  4. 形态学微噪: 1-2px 十字核腐蚀/膨胀 + 极细微高斯噪点
"""
import os, sys, csv, re, time, glob, argparse, multiprocessing as mp
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates, binary_erosion, binary_dilation
from scipy.ndimage import generate_binary_structure

TARGET = 6
NEXT_ID = 1000000


def count_combos(csv_path):
    rows, combo = [], Counter()
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            combo[(r["script"], r["character"], r["calligrapher"])] += 1
            rows.append(r)
    return rows, combo


def _elastic(img_arr, rng, alpha=6.0, sigma=3.0):
    """弱弹性形变. img_arr: (H,W) float32."""
    h, w = img_arr.shape
    dx = gaussian_filter((rng.rand(h, w) - 0.5) * 2, sigma) * alpha
    dy = gaussian_filter((rng.rand(h, w) - 0.5) * 2, sigma) * alpha
    y, x = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    idx = np.clip(y + dy, 0, h - 1), np.clip(x + dx, 0, w - 1)
    return map_coordinates(img_arr, idx, order=1, mode="reflect").astype(np.float32)


def _morph_noise(img_arr, rng):
    """形态学微噪: 随机轻微腐蚀/膨胀(1-2px 十字核) + 极细微高斯噪点.
    只改动变化像素的灰度, 不破坏整体墨色纹理."""
    a = img_arr.copy()
    struct = generate_binary_structure(2, 1)  # 十字核
    mask = a < 0.5  # 墨迹区
    if rng.rand() < 0.3:
        dil = binary_dilation(mask, structure=struct, iterations=1)
        grew = dil & ~mask  # 新增墨迹像素: 略微变暗
        a[grew] = a[grew] * 0.5
    if rng.rand() < 0.7:
        ero = binary_erosion(mask, structure=struct, iterations=1)
        shrank = mask & ~ero  # 变白区域: 略微提亮
        a[shrank] = a[shrank] * 0.5 + 0.5
    # 极细微高斯噪点 (整图 0.5%, 强度小)
    if rng.rand() < 0.4:
        noise = rng.normal(0, 0.015, a.shape).astype(np.float32) * (rng.rand(a.shape) < 0.005)
        a = np.clip(a + noise, 0, 1)
    return a


def _augment_one(args):
    """CPU 单个变体增强, 返回保存路径. args=(src_img_path, new_id, seed)."""
    src_img_path, new_id, seed = args
    rng = np.random.RandomState(seed)
    img = Image.open(src_img_path).convert("L").resize((256, 256), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0

    # 1) 弹性形变 (p=0.5, 弱)
    if rng.rand() < 0.5:
        arr = _elastic(arr, rng, alpha=rng.uniform(4, 8), sigma=rng.uniform(2.5, 3.5))

    # 2) 微仿射: 旋转±1.5°, 缩放±5%, 平移±2-5%
    angle = rng.uniform(-1.5, 1.5)
    scale = rng.uniform(0.95, 1.05)
    tx_pct = rng.uniform(-0.05, 0.05)
    ty_pct = rng.uniform(-0.05, 0.05)
    img2 = Image.fromarray((arr * 255).astype(np.uint8))
    img2 = img2.rotate(angle, resample=Image.BICUBIC, fillcolor=255)
    new_w = max(8, int(256 * scale))
    new_h = max(8, int(256 * scale))
    small = img2.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("L", (256, 256), 255)
    ox = max(0, min((256 - new_w) // 2 + int(tx_pct * 256), 256 - new_w))
    oy = max(0, min((256 - new_h) // 2 + int(ty_pct * 256), 256 - new_h))
    canvas.paste(small, (ox, oy))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0

    # 3) 形态学微噪
    arr = _morph_noise(arr, rng)

    # 4) 阈值/Gamma 抖动 (墨量变化)
    gamma = rng.uniform(0.9, 1.1)
    arr = np.clip(arr ** gamma, 0, 1)

    out_png = arr * 255.0
    out_pil = Image.fromarray(out_png.astype(np.uint8))
    out_path = os.path.join(OUT_IMGS, f"{new_id}.png")
    out_pil.save(out_path)
    return out_path


def phase_a(args, rows, need_aug, combo2rows):
    """CPU 多进程增强 → 存 PNG + aug_meta."""
    rng = np.random.RandomState(args.seed)
    tasks, aug_meta = [], []
    for key, n_needed in sorted(need_aug.items(), key=lambda x: -x[1]):
        script, char, calli = key
        src = rng.choice(combo2rows[key])
        iid = int(re.search(r"(\d+)\.png", src["image_path"]).group(1))
        img_path = os.path.join(args.img_root, f"{iid}.png")
        if not os.path.exists(img_path):
            print(f"  [skip] missing {img_path}")
            continue
        for vi in range(n_needed):
            new_id = NEXT_ID + len(tasks)
            tasks.append((img_path, new_id, args.seed * 1000 + vi))
            aug_meta.append((new_id, script, char, calli,
                             src["calligrapher_id"], src["character_id"], src["glyph_id"],
                             src.get("script_id", "")))
    print(f"  {len(tasks)} augmentation tasks")

    t1 = time.time()
    print(f"CPU augmentation with {args.workers} workers...")
    with mp.Pool(args.workers) as pool:
        list(pool.imap(_augment_one, tasks, chunksize=128))
    print(f"  done in {time.time()-t1:.0f}s ({len(tasks)/(time.time()-t1):.0f} img/s)")
    return aug_meta


def phase_b(args, aug_meta):
    """GPU 编码增强 PNG → 临时 latent shard."""
    import torch
    device = torch.device("cuda")
    print("Loading VAE...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    os.makedirs(args.out_temp, exist_ok=True)
    t2 = time.time()
    vae_bs = 32
    lat_buf, id_buf = [], []
    shard_idx = 0
    total = len(aug_meta)
    for batch_start in range(0, total, vae_bs):
        batch_end = min(batch_start + vae_bs, total)
        xs = np.stack([
            np.asarray(Image.open(os.path.join(OUT_IMGS, f"{aug_meta[batch_start+k][0]}.png"))
                       .convert("L"), dtype=np.float32) / 127.5 - 1.0
            for k in range(batch_end - batch_start)
        ], axis=0)
        xs = np.stack([xs, xs, xs], axis=-1)  # (B,256,256,3)
        x_t = torch.from_numpy(np.transpose(xs, (0, 3, 1, 2))).to(device)
        with torch.no_grad():
            lat = vae.encode(x_t).latent_dist.sample() * args.scaling_factor
            if lat.shape[-1] != 32:
                lat = torch.nn.functional.interpolate(lat, size=32, mode="bilinear")
            lat = lat.float().cpu().numpy()
        for vi, l in enumerate(lat):
            lat_buf.append(l.astype(np.float16))
            id_buf.append(aug_meta[batch_start + vi][0])
            if len(lat_buf) >= 5000:
                _save_shard(lat_buf, id_buf, args.out_temp, shard_idx)
                shard_idx += 1
                lat_buf, id_buf = [], []
        if (batch_start // vae_bs) % 100 == 0 or batch_end == total:
            print(f"  ... {batch_end}/{total} encoded, {(time.time()-t2)/60:.1f}min", flush=True)
    if lat_buf:
        _save_shard(lat_buf, id_buf, args.out_temp, shard_idx)
    print(f"  encode done in {(time.time()-t2)/60:.1f}min")


def phase_c(args, rows, aug_meta):
    """合并原 latent + 增强 latent → final_latents_mid_clean + CSV.

    只保留 rows 中出现的 img_id (清洗后), 丢弃被剔除的国标外样本 latent.
    """
    import glob
    os.makedirs(args.out_latent, exist_ok=True)

    # 收集清洗后 csv 的 img_id 集合 (只保留这些原 latent)
    keep_ids = set()
    for r in rows:
        m = re.search(r"(\d+)\.png", r["image_path"])
        if m:
            keep_ids.add(int(m.group(1)))
    print(f"  keeping {len(keep_ids)} original latents (filtered from full set)")

    # 过滤原 latent shard → 只写保留的 img_id
    orig_shards = sorted(glob.glob(os.path.join(args.latent_dir, "shard_*.npz")))
    n_orig = 0
    out_shard_idx = 0
    lat_buf, id_buf = [], []
    SHARD = args.shard_size
    for sp in orig_shards:
        d = np.load(sp)
        lat, iid = d["latents"], d["img_ids"]
        for i in range(len(iid)):
            if int(iid[i]) in keep_ids:
                lat_buf.append(lat[i])
                id_buf.append(int(iid[i]))
                if len(lat_buf) >= SHARD:
                    _save_shard(lat_buf, id_buf, args.out_latent, out_shard_idx)
                    out_shard_idx += 1
                    n_orig += len(lat_buf)
                    lat_buf, id_buf = [], []
        d.close()
    if lat_buf:
        _save_shard(lat_buf, id_buf, args.out_latent, out_shard_idx)
        n_orig += len(lat_buf)
        out_shard_idx += 1
    print(f"  wrote {out_shard_idx} filtered orig shards ({n_orig} latents)")

    # 追加增强 latent shard
    aug_shards = sorted(glob.glob(os.path.join(args.out_temp, "shard_*.npz")))
    total_aug = 0
    for si, sp in enumerate(aug_shards):
        d = np.load(sp)
        lat, iid = d["latents"], d["img_ids"]
        total_aug += int(iid.shape[0])
        np.savez_compressed(os.path.join(args.out_latent, f"shard_{out_shard_idx + si:05d}.npz"),
                            latents=lat, img_ids=iid)
        d.close()
    print(f"  wrote {len(aug_shards)} aug shards -> {args.out_latent} (total {total_aug} aug latents)")

    CSV_FIELDS = ["image_path", "calligrapher", "script", "character",
                  "calligrapher_id", "script_id", "character_id", "glyph_id"]
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        for new_id, script, char, calli, caid, chid, gid, sid in aug_meta:
            w.writerow({
                "image_path": f"final_imgs_mid_clean/{new_id}.png",
                "calligrapher": calli, "script": script, "character": char,
                "calligrapher_id": caid, "script_id": sid,
                "character_id": chid, "glyph_id": gid,
            })
    print(f"  CSV: {args.out_csv} ({len(rows) + len(aug_meta)} rows)")


def _save_shard(lat_buf, id_buf, out_dir, idx):
    path = os.path.join(out_dir, f"shard_{idx:05d}.npz")
    np.savez_compressed(path, latents=np.stack(lat_buf, axis=0),
                        img_ids=np.array(id_buf, dtype=np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="5script/train_3top30_common.csv")
    ap.add_argument("--img-root", default="final_imgs_256")
    ap.add_argument("--latent-dir", default="final_latents")
    ap.add_argument("--vae", default="pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--out-imgs", default="final_imgs_mid_clean")
    ap.add_argument("--out-latent", default="final_latents_mid_clean")
    ap.add_argument("--out-csv", default="5script/train_mid_clean.csv")
    ap.add_argument("--out-temp", default="/tmp/mid_clean_tmp")
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--shard-size", type=int, default=5000)
    ap.add_argument("--scaling-factor", type=float, default=0.18215)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--phase", default="all", help="all|aug|encode|merge")
    args = ap.parse_args()

    global OUT_IMGS
    OUT_IMGS = args.out_imgs
    os.makedirs(OUT_IMGS, exist_ok=True)
    os.makedirs(args.out_temp, exist_ok=True)

    rows, combo = count_combos(args.train_csv)
    need_aug = {k: args.target - n for k, n in combo.items() if n < args.target}
    total_needed = sum(need_aug.values())
    print(f"Orig: {len(rows)} images, {len(combo)} combos. Need {total_needed} aug "
          f"({len(need_aug)} sparse combos) -> target {args.target}")

    combo2rows = defaultdict(list)
    for r in rows:
        combo2rows[(r["script"], r["character"], r["calligrapher"])].append(r)

    aug_meta = None
    if args.phase in ("all", "aug"):
        aug_meta = phase_a(args, rows, need_aug, combo2rows)
        # persist meta
        with open(os.path.join(args.out_temp, "aug_meta.csv"), "w", encoding="utf-8") as f:
            f.write("new_id,script,char,calli,calli_id,char_id,glyph_id,script_id\n")
            for m in aug_meta:
                f.write(",".join(str(x) for x in m) + "\n")
    if args.phase in ("all", "encode"):
        if aug_meta is None:
            aug_meta = _load_meta(args.out_temp)
        phase_b(args, aug_meta)
    if args.phase in ("all", "merge"):
        if aug_meta is None:
            aug_meta = _load_meta(args.out_temp)
        phase_c(args, rows, aug_meta)
    print("DONE")


def _load_meta(temp_dir):
    meta = []
    with open(os.path.join(temp_dir, "aug_meta.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            meta.append((int(r["new_id"]), r["script"], r["char"], r["calli"],
                         r["calli_id"], r["char_id"], r["glyph_id"], r["script_id"]))
    return meta


if __name__ == "__main__":
    main()
