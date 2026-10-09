# -*- coding: utf-8 -*-
"""check_lpips_net.py — 校验 torch 缓存里的 LPIPS backbone 权重是否**真的能用**。

不做下载 (下载交给 curl.exe 这类标准工具)。只回答:
  1. 文件在不在、多少字节;
  2. sha256 前缀是否与 torch 官方 URL 文件名里的那 8 位一致;
  3. 能不能装进 lpips 并算出合理距离 (同图≈0, 异图显著>0)。

用法: python tools/check_lpips_net.py --net alex
"""
import argparse
import hashlib
import os
import sys

NETS = {
    "alex": ("alexnet-owt-7be5be79.pth", "7be5be79"),
    "vgg": ("vgg16-397923af.pth", "397923af"),
    # lpips pip 包自带的线性层权重 (不用外网, 用来确认包本身是好的)
    "alex_lin": ("alex.pth", None),
}
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="alex", choices=["alex", "vgg"])
    a = ap.parse_args()
    fname, prefix = NETS[a.net]
    p = os.path.join(CACHE_DIR, fname)

    print(f"[file] {p}")
    if not os.path.isfile(p):
        print("  ✗ 不存在")
        return 1
    size = os.path.getsize(p)
    h = sha256(p)
    print(f"  size   = {size:,} 字节 ({size/2**20:.2f} MiB)")
    print(f"  sha256 = {h}")
    print(f"  期望前缀 = {prefix}  ->  {'✓ 一致' if h.startswith(prefix) else '✗ 不一致'}")

    # zip 尾部完整性 (torch checkpoint 是 zip 容器, 截断必在尾部报错)
    try:
        import zipfile
        with zipfile.ZipFile(p) as z:
            n = len(z.namelist())
        print(f"  zip 容器 = ✓ 可读, 含 {n} 个条目")
    except Exception as e:
        print(f"  zip 容器 = ✗ {e!r}  <-- 文件被截断/损坏")
        return 1

    print("\n=== lpips 加载 + 合理性验证 ===")
    try:
        import numpy as np
        import torch
        import lpips
        fn = lpips.LPIPS(net=a.net, verbose=False).eval()
        print(f"  ✓ lpips.LPIPS(net={a.net!r}) 加载成功")
        x = np.full((256, 256, 3), 255, np.uint8)
        x[80:150, 60:130] = 0
        y = np.full((256, 256, 3), 255, np.uint8)
        y[90:160, 150:220] = 0
        rng = np.random.default_rng(0)
        z = np.clip(x.astype(np.int16) + rng.integers(-6, 7, x.shape), 0, 255).astype(np.uint8)
        t = lambda arr: torch.from_numpy(arr).permute(2, 0, 1)[None].float() / 127.5 - 1.0
        with torch.no_grad():
            d_same = float(fn(t(x), t(z)).item())
            d_diff = float(fn(t(x), t(y)).item())
        print(f"  同图(轻噪声) = {d_same:.5f}   异图(位移) = {d_diff:.5f}")
        ok = d_diff > d_same and d_same < 0.05
        print("  " + ("✓ 可用" if ok else "✗ 距离不单调, 不可信"))
        return 0 if ok else 1
    except Exception as e:
        print(f"  ✗ 失败: {e!r}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
