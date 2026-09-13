# -*- coding: utf-8 -*-
"""
GPU 高效预处理：final_images -> 256x256 图 + canny + skeleton（全批量 GPU）。
- final_imgs_256/{img_id}.png   (INTER_CUBIC 直接拉伸, 匹配 latent)
- final_canny/{img_id}.png
- final_skeleton/{img_id}.png
canny 用 GPU Sobel 边缘 + 双阈值；skeleton 用 GPU 批量 Zhang-Suen。
"""
import os, sys, glob, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, cv2, torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

SRC = "final_images"
DIRS = {"img": "final_imgs_256", "canny": "final_canny", "skel": "final_skeleton"}
SIZE = 256
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)


def read(p):
    buf = np.fromfile(p, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def gpu_otsu_batch(gray_b):
    """gray_b: (B,H,W) float32 [0,255]. 返回 (B,H,W) 0/1 uint8，前景=1。"""
    # 对每张图按均值方向：浅底深字 -> 反色；否则保持
    mean = gray_b.mean(dim=(1, 2), keepdim=True)  # (B,1,1)
    # 简化：Otsu 用全局最佳阈值（直方图，批量近似）
    # 用固定经验：浅底(mean>127)取反，深底不取反
    bin_b = (gray_b > 127).float()
    inv = (mean > 127).float()
    out = torch.where(inv > 0, 1.0 - bin_b, bin_b)  # 前景=笔画=1
    return out


def gpu_zhang_suen(bin_b, max_iter=200):
    """bin_b: (B,H,W) float32 0/1 (前景=1)。返回 (B,H,W) float32 0/1 骨架。"""
    B, H, W = bin_b.shape
    device = bin_b.device
    img = bin_b.clone()
    # 右、下各扩 1（循环邻域用 shift）
    for it in range(max_iter):
        # 8 邻域 (P2..P9) 用 shift
        def shift(a, dx, dy):
            if dx == 0 and dy == 0:
                return a
            return torch.roll(a, shifts=(dy, dx), dims=(1, 2))
        P2 = shift(img, 0, -1)
        P3 = shift(img, 1, -1)
        P4 = shift(img, 1, 0)
        P5 = shift(img, 1, 1)
        P6 = shift(img, 0, 1)
        P7 = shift(img, -1, 1)
        P8 = shift(img, -1, 0)
        P9 = shift(img, -1, -1)
        neigh = torch.stack([P2, P3, P4, P5, P6, P7, P8, P9], dim=1)  # (B,8,H,W)
        Bsum = neigh.sum(dim=1)  # 前景邻域数
        # 0->1 变换次数 A (P2..P9, 循环)
        A = torch.zeros_like(Bsum)
        for i in range(8):
            cur = neigh[:, i]
            nxt = neigh[:, (i + 1) % 8]
            A += ((cur == 0) & (nxt == 1)).float()
        center = img
        # 子迭代1
        cond = (center == 1) & (Bsum >= 2) & (Bsum <= 6) & (A == 1)
        cond &= (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
        img = img * (1 - cond.float())
        # 重新算邻域（子迭代2 用更新后的 img）
        P2 = shift(img, 0, -1); P3 = shift(img, 1, -1); P4 = shift(img, 1, 0)
        P5 = shift(img, 1, 1); P6 = shift(img, 0, 1); P7 = shift(img, -1, 1)
        P8 = shift(img, -1, 0); P9 = shift(img, -1, -1)
        neigh = torch.stack([P2, P3, P4, P5, P6, P7, P8, P9], dim=1)
        Bsum = neigh.sum(dim=1)
        A = torch.zeros_like(Bsum)
        for i in range(8):
            cur = neigh[:, i]; nxt = neigh[:, (i + 1) % 8]
            A += ((cur == 0) & (nxt == 1)).float()
        cond = (img == 1) & (Bsum >= 2) & (Bsum <= 6) & (A == 1)
        cond &= (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)
        img = img * (1 - cond.float())
        if not torch.any(img > 0):
            break
    return img


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(SRC, "*.png")))
    if args.limit > 0:
        files = files[:args.limit]
    print("files:", len(files))
    device = "cuda"
    # Sobel 核
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=device).view(1,1,3,3)
    sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=device).view(1,1,3,3)

    done = 0
    pool = ThreadPoolExecutor(max_workers=16)
    t0 = time.time()
    BATCH = 64
    for i in range(0, len(files), BATCH):
        chunk = files[i:i+BATCH]
        futs = {p: pool.submit(read, p) for p in chunk}
        ids, imgs = [], []
        for p, f in futs.items():
            img = f.result()
            if img is None:
                continue
            ids.append(os.path.splitext(os.path.basename(p))[0])
            imgs.append(cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC))
        if not imgs:
            continue
        arr = np.stack(imgs)                       # (B,256,256,3) BGR
        t = torch.from_numpy(arr).permute(0,3,1,2).float().to(device) / 255.0
        # 灰度
        gray = 0.114*t[:,0:1] + 0.587*t[:,1:2] + 0.299*t[:,2:3]  # (B,1,256,256)
        gray_c = gray.squeeze(1) * 255.0
        # canny: Sobel 幅度 -> 双阈值
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        mag = torch.sqrt(gx**2 + gy**2) * 255.0    # (B,1,256,256)
        canny = ((mag > 150).float() * 255).squeeze(1).byte().cpu().numpy()
        # skeleton
        bin_b = gpu_otsu_batch(gray_c)
        skel = gpu_zhang_suen(bin_b)
        skel = (skel * 255).byte().cpu().numpy()

        for j, pid in enumerate(ids):
            cv2.imencode(".png", imgs[j])[1].tofile(os.path.join(DIRS["img"], f"{pid}.png"))
            cv2.imencode(".png", canny[j])[1].tofile(os.path.join(DIRS["canny"], f"{pid}.png"))
            cv2.imencode(".png", skel[j])[1].tofile(os.path.join(DIRS["skel"], f"{pid}.png"))
        done += len(imgs)
        if (i//BATCH+1) % 20 == 0:
            el = time.time()-t0
            print(f"  {done}/{len(files)} {done/el:.0f}/s", flush=True)
    el = time.time()-t0
    print(f"Done {done} in {el:.0f}s ({done/el:.0f}/s)")

if __name__ == "__main__":
    main()
