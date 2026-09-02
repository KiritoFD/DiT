# -*- coding: utf-8 -*-
"""
build_fame_skel1px.py — 为 fame 构建 **1px 细骨架** 及其 VAE latent（GPU 编码）。

背景与动机
----------
历史 ControlNet (fame-ctrl) 用的是 3px 膨胀骨架（GT 图 → skeletonize → 8 邻域
dilate×3）。3px 骨架在 256px 图上占了可观面积，其 VAE latent 与真实书法图的
latent 相当接近 —— 我们怀疑 0.8045 的 "高保真复刻" 中有一部分来自这种
**条件本身的泄露**（给的条件已经很像目标图），而非真正的结构控制能力。

改成 1px 可以验证这一点：
  - 若 1px 下 SSIM 仍 ≈0.80 且跟随 IoU 不降 → 说明模型真的在学结构，泄露不是主因
  - 若 1px 下 SSIM 明显下降但**跟随 IoU 反升/持平** → 说明 3px 的高分确实含泄露，
    1px 才是更干净的 "纯结构" 条件，且对下游（零样本、可编辑）更有价值

设计要点
--------
1. **一次性覆盖 train + eval**：历史教训是 fame 的 skel latent 只覆盖 train，
   eval 骨架后来单独补（/tmp 路径导致 stale bug，见 docs/system/13 §5）。
   本脚本直接对两份 csv 的并集构建，杜绝 train/eval 条件域不一致。
2. **GPU 编码**：VAE encode 在 cuda 上批跑（用户要求，不再走 CPU 路径）。
   骨架提取（skeletonize）本身是 CPU 算法，用多进程并行；真正的耗时项
   （VAE encode）走 GPU。
3. **shard 布局对齐 final_skel_latents_fame**：同样 20 个 shard、按 img_id
   排序切分，使新的 latent 目录可以直接替换旧的、而 dataloader 的
   img_id 索引逻辑无需改动。
4. **极性**：白底黑线（线=0，底=255），与 GT 书法图同极性 —— 与 3px 版一致。

用法（远程，GPU 空闲时）
------------------------
  python tools/build_fame_skel1px.py \
      --train-csv 5script/train_fame.csv \
      --eval-csv 5script/eval_fame_strict.csv \
      --img-root final_imgs_256 \
      --skel1-dir final_skel1_fame \
      --latent-out final_skel_latents_fame_1px \
      --workers 32 --vae-batch 96
"""
import os, sys, csv, re, glob, time, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from PIL import Image
import torch


# ---------------------------------------------------------------------------
# 骨架提取（CPU；与 eval 口径一致，带 skimage fallback）
# ---------------------------------------------------------------------------
def _skel_impl():
    try:
        from skimage.morphology import skeletonize
        return skeletonize
    except ImportError:
        from scipy.ndimage import binary_erosion, generate_binary_structure

        def skeletonize(binary):
            skel = np.zeros_like(binary)
            img = binary.copy()
            st = generate_binary_structure(2, 2)
            while img.any():
                er = binary_erosion(img, structure=st)
                skel |= img & ~er
                img = er
            return skel
        return skeletonize


_SKEL = None


def extract_1px(img_path):
    """GT 图 → 1px 骨架（白底黑线 uint8）。返回 (256,256) 数组。"""
    global _SKEL
    if _SKEL is None:
        _SKEL = _skel_impl()
    a = np.asarray(Image.open(img_path).convert("L"))
    sk = _SKEL(a < 127)          # 白底黑字：暗像素 = 笔画
    return np.where(sk, 0, 255).astype(np.uint8)


def _worker(args):
    img_path, out_path = args
    if os.path.exists(out_path):
        return None
    try:
        arr = extract_1px(img_path)
        Image.fromarray(arr, "L").save(out_path)
        return None
    except Exception as e:
        return f"FAIL {img_path}: {e}"


def build_pngs(ids, img_root, skel1_dir, workers):
    """多进程生成 1px 骨架 PNG（跳过已存在）。"""
    os.makedirs(skel1_dir, exist_ok=True)
    tasks = []
    for iid in ids:
        op = os.path.join(skel1_dir, f"{iid}.png")
        if not os.path.exists(op):
            tasks.append((os.path.join(img_root, f"{iid}.png"), op))
    print(f"[skel1] {len(tasks)} todo / {len(ids)} (skip {len(ids)-len(tasks)})",
          flush=True)
    if not tasks:
        return
    import multiprocessing as mp
    t0 = time.time()
    fails = 0
    with mp.Pool(workers) as pool:
        for i, err in enumerate(pool.imap_unordered(_worker, tasks, chunksize=128)):
            if err:
                fails += 1
                if fails <= 5:
                    print(f"  {err}", flush=True)
            if (i + 1) % 20000 == 0 or i + 1 == len(tasks):
                print(f"[skel1] {i+1}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[skel1] PNG done, fails={fails} ({time.time()-t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------------
# GPU VAE encode → shards
# ---------------------------------------------------------------------------
def build_latents(ids, skel1_dir, latent_out, vae_path, nshard=20,
                  batch=96, scaling=0.18215):
    from diffusers.models import AutoencoderKL
    os.makedirs(latent_out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKL.from_pretrained(vae_path).to(dev).eval()
    print(f"[vae] loaded on {dev}", flush=True)

    # 全量一次性编码到内存（与 build_fame_dataset 相同的做法，保证 shard 切分一致）
    n = len(ids)
    all_lat = np.empty((n, 4, 32, 32), dtype=np.float16)
    all_ids = np.array(ids, dtype=np.int64)

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, batch):
            chunk = ids[i:i + batch]
            arrs = []
            for iid in chunk:
                p = os.path.join(skel1_dir, f"{iid}.png")
                arrs.append(np.asarray(Image.open(p).convert("L")))
            x = torch.from_numpy(
                np.stack(arrs).astype(np.float32) / 255.0 * 2.0 - 1.0)[:, None]
            x = x.repeat(1, 3, 1, 1).to(dev)
            lat = (vae.encode(x).latent_dist.mode() * scaling).half().cpu().numpy()
            all_lat[i:i + len(chunk)] = lat
            if (i // batch) % 100 == 0 or i + batch >= n:
                print(f"[vae] {min(i+batch, n)}/{n} ({time.time()-t0:.0f}s)", flush=True)

    per = (n + nshard - 1) // nshard
    for s in range(nshard):
        lo, hi = s * per, min((s + 1) * per, n)
        np.savez_compressed(
            os.path.join(latent_out, f"shard_{s:04d}.npz"),
            latents=all_lat[lo:hi], img_ids=all_ids[lo:hi])
    np.savez_compressed(
        os.path.join(latent_out, "_all.npz"), latents=all_lat, img_ids=all_ids)
    print(f"[vae] DONE {n} in {time.time()-t0:.0f}s -> {latent_out} "
          f"({nshard} shards)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="5script/train_fame.csv")
    ap.add_argument("--eval-csv", default="5script/eval_fame_strict.csv")
    ap.add_argument("--img-root", default="final_imgs_256")
    ap.add_argument("--skel1-dir", default="final_skel1_fame")
    ap.add_argument("--latent-out", default="final_skel_latents_fame_1px")
    ap.add_argument("--vae-path", default="pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--vae-batch", type=int, default=96)
    ap.add_argument("--nshard", type=int, default=20)
    ap.add_argument("--skip-png", action="store_true", help="PNG 已生成时跳过")
    args = ap.parse_args()

    def ids_of(p):
        out = []
        with open(p, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                m = re.search(r"(\d+)\.png", row.get("image_path", ""))
                if m:
                    out.append(int(m.group(1)))
        return sorted(set(out))

    tr = ids_of(args.train_csv)
    ev = ids_of(args.eval_csv)
    all_ids = sorted(set(tr) | set(ev))
    print(f"[plan] train={len(tr)} eval={len(ev)} union={len(all_ids)}", flush=True)
    print(f"[plan] skel1-dir={args.skel1_dir}  latent-out={args.latent_out}", flush=True)

    if not args.skip_png:
        build_pngs(all_ids, args.img_root, args.skel1_dir, args.workers)
    else:
        print("[skel1] --skip-png, 跳过 PNG 阶段", flush=True)

    build_latents(all_ids, args.skel1_dir, args.latent_out, args.vae_path,
                  args.nshard, args.vae_batch)

    # 自检：eval 侧必须全部覆盖（历史就是这里出过 stale bug）
    got = set()
    for sp in glob.glob(os.path.join(args.latent_out, "shard_*.npz")):
        with np.load(sp) as d:
            got.update(int(x) for x in d["img_ids"])
    print(f"\n[check] latent ids = {len(got)}")
    print(f"[check] eval 覆盖 = {len(set(ev) & got)}/{len(ev)}")
    print(f"[check] train 覆盖 = {len(set(tr) & got)}/{len(tr)}")
    miss_ev = sorted(set(ev) - got)
    if miss_ev:
        print(f"[check] !! eval 缺失 {len(miss_ev)}: {miss_ev[:10]}")


if __name__ == "__main__":
    main()
