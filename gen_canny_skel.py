# -*- coding: utf-8 -*-
"""
高性能 Step2：final_imgs_256 -> final_canny/{id}.png + final_skeleton/{id}.png（全 GPU 批量）。
canny: GPU Sobel 幅度 + 双阈值。
skel : GPU 批量 Zhang-Suen。
增量：跳过已存在输出。
"""
import os, sys, glob, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, cv2, torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

SRC = "final_imgs_256"
DIRS = {"canny": "final_canny", "skel": "final_skeleton"}
SIZE = 256
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

IO = int(os.environ.get("IO_WORKERS", "32"))
BATCH = int(os.environ.get("GPU_BATCH", "128"))
MAX_ITER = int(os.environ.get("MAX_ITER", "80"))

sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)


def read(p):
    buf = np.fromfile(p, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def gpu_zhang_suen(bin_b, max_iter):
    B, H, W = bin_b.shape
    img = bin_b.clone()
    # pad 一层(0)，用切片取 8 邻域，避免 torch.roll 卷绕
    for it in range(max_iter):
        P = F.pad(img, (1, 1, 1, 1), mode="constant", value=0)
        # 8 邻域 (P2..P9)
        P2 = P[:, :-2, 1:-1]  # (row-1, col)   上
        P3 = P[:, :-2, 2:]    # 上右
        P4 = P[:, 1:-1, 2:]   # 右
        P5 = P[:, 2:, 2:]     # 下右
        P6 = P[:, 2:, 1:-1]   # 下
        P7 = P[:, 2:, :-2]    # 下左
        P8 = P[:, 1:-1, :-2]  # 左
        P9 = P[:, :-2, :-2]   # 上左
        neigh = torch.stack([P2, P3, P4, P5, P6, P7, P8, P9], dim=1)
        Bsum = neigh.sum(dim=1)
        # A: 0->1 变换次数 (P2..P9 循环)，向量化
        zeros = neigh[:, :-1] == 0
        ones = neigh[:, 1:] == 1
        A = (zeros & ones).sum(dim=1) + ((neigh[:, -1] == 0) & (neigh[:, 0] == 1)).float()
        cond = (img == 1) & (Bsum >= 2) & (Bsum <= 6) & (A == 1)
        cond &= (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
        img = img * (1 - cond.float())
        # 子迭代2
        P = F.pad(img, (1, 1, 1, 1), mode="constant", value=0)
        P2 = P[:, :-2, 1:-1]; P3 = P[:, :-2, 2:]; P4 = P[:, 1:-1, 2:]
        P5 = P[:, 2:, 2:]; P6 = P[:, 2:, 1:-1]; P7 = P[:, 2:, :-2]
        P8 = P[:, 1:-1, :-2]; P9 = P[:, :-2, :-2]
        neigh = torch.stack([P2, P3, P4, P5, P6, P7, P8, P9], dim=1)
        Bsum = neigh.sum(dim=1)
        zeros = neigh[:, :-1] == 0
        ones = neigh[:, 1:] == 1
        A = (zeros & ones).sum(dim=1) + ((neigh[:, -1] == 0) & (neigh[:, 0] == 1)).float()
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
    # 增量
    todo = []
    for p in files:
        pid = os.path.splitext(os.path.basename(p))[0]
        if not os.path.exists(os.path.join(DIRS["canny"], pid + ".png")) or \
           not os.path.exists(os.path.join(DIRS["skel"], pid + ".png")):
            todo.append(p)
    if args.limit > 0:
        todo = todo[:args.limit]
    print("files:", len(files), "todo:", len(todo), "done:", len(files)-len(todo))
    device = "cuda"
    kx = sobel_x.to(device); ky = sobel_y.to(device)
    done = 0
    pool = ThreadPoolExecutor(max_workers=IO)
    t0 = time.time()
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i+BATCH]
        futs = {p: pool.submit(read, p) for p in chunk}
        ids, imgs = [], []
        for p, f in futs.items():
            img = f.result()
            if img is None:
                continue
            ids.append(os.path.splitext(os.path.basename(p))[0])
            imgs.append(img)
        if not imgs:
            continue
        t = torch.from_numpy(np.stack(imgs)).permute(0,3,1,2).float().to(device) / 255.0  # (B,3,256,256)
        gray = (0.114*t[:,0:1] + 0.587*t[:,1:2] + 0.299*t[:,2:3]) * 255.0  # (B,1,256,256)
        gray_c = gray.squeeze(1)
        # canny: Sobel 幅度双阈值
        gx = F.conv2d(gray, kx, padding=1)
        gy = F.conv2d(gray, ky, padding=1)
        mag = torch.sqrt(gx**2 + gy**2)
        canny = ((mag > 150).float() * 255.0).squeeze(1)
        # skel: 二值化
        mean = gray_c.mean(dim=(1,2), keepdim=True)
        bin_b = ((gray_c > 127).float())
        bin_b = torch.where(mean > 127, 1.0 - bin_b, bin_b)
        skel = gpu_zhang_suen(bin_b, MAX_ITER) * 255.0
        canny_np = canny.byte().cpu().numpy()
        skel_np = skel.byte().cpu().numpy()
        for pid, cn, sk in zip(ids, canny_np, skel_np):
            cv2.imencode(".png", cn)[1].tofile(os.path.join(DIRS["canny"], f"{pid}.png"))
            cv2.imencode(".png", sk)[1].tofile(os.path.join(DIRS["skel"], f"{pid}.png"))
        done += len(imgs)
        if (i//BATCH+1) % 20 == 0:
            el = time.time()-t0
            print(f"  {done}/{len(todo)} {done/el:.0f}/s", flush=True)
    el = time.time()-t0
    print(f"Done {done} in {el:.0f}s ({done/el:.0f}/s)")

if __name__ == "__main__":
    main()
