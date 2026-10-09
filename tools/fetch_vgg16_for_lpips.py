# -*- coding: utf-8 -*-
"""fetch_vgg16_for_lpips.py — 手工下载 LPIPS(vgg) 需要的 vgg16 权重并校验 sha256。

背景: torchvision 的自动下载在本机反复失败 —— 缓存里只剩 3 个 `.partial`
      (1.2MB/131KB/1.7MB), 且曾出现 "下载到 528M 但 hash 不匹配"。
      本脚本: 直连下载 -> 校验 sha256 前 8 位 == 397923af -> 才放进 torch cache。
      (torch 的 URL 文件名 `vgg16-397923af.pth` 里的 8 位就是 sha256 的前缀)

用法:
    python tools/fetch_vgg16_for_lpips.py
"""
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request

URL = "https://download.pytorch.org/models/vgg16-397923af.pth"
EXPECT_PREFIX = "397923af"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
DEST = os.path.join(CACHE_DIR, "vgg16-397923af.pth")

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def sha256_prefix(path, nbytes=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 1. 清掉所有 .partial 残留
    removed = 0
    for f in os.listdir(CACHE_DIR):
        if f.startswith("vgg16-") and f.endswith(".partial"):
            os.remove(os.path.join(CACHE_DIR, f))
            removed += 1
    print(f"[clean] 删除 {removed} 个 .partial 残留")

    # 2. 若已有完好文件, 直接校验
    if os.path.isfile(DEST):
        h = sha256_prefix(DEST)
        size = os.path.getsize(DEST) / 2 ** 20
        if h.startswith(EXPECT_PREFIX):
            print(f"[ok] 已存在且校验通过: {DEST} ({size:.1f} MiB, sha256={h[:12]}...)")
            return 0
        print(f"[warn] 已存在但 sha256={h[:12]}... 不匹配 ({size:.1f} MiB) -> 重新下载")
        os.remove(DEST)

    # 3. 下载到临时文件并校验
    tmp = tempfile.NamedTemporaryFile(suffix=".pth", delete=False).name
    print(f"[get] {URL}")
    try:
        with urllib.request.urlopen(URL, timeout=60) as r, open(tmp, "wb") as out:
            total = int(r.headers.get("Content-Length", 0))
            got = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                got += len(chunk)
                if total:
                    pct = 100.0 * got / total
                    print(f"\r       {got/2**20:7.1f}/{total/2**20:.1f} MiB ({pct:5.1f}%)",
                          end="", flush=True)
        print()
    except Exception as e:
        print(f"\n[FAIL] 下载出错: {e!r}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return 1

    h = sha256_prefix(tmp)
    size = os.path.getsize(tmp) / 2 ** 20
    print(f"[check] {size:.1f} MiB  sha256={h[:16]}...  期望前缀={EXPECT_PREFIX}")
    if not h.startswith(EXPECT_PREFIX):
        print("[FAIL] sha256 不匹配 —— 内容不对, 不放进缓存")
        os.remove(tmp)
        return 1

    shutil.move(tmp, DEST)
    print(f"[done] ✓ 校验通过, 已放入 {DEST}")

    # 4. 立刻验证 lpips 能加载
    try:
        import torch
        import lpips
        f = lpips.LPIPS(net="vgg").eval()
        print(f"[verify] ✓ lpips.LPIPS(net='vgg') 加载成功 "
              f"(参数 {sum(p.numel() for p in f.parameters())/1e6:.1f}M)")
    except Exception as e:
        print(f"[verify] ✗ lpips 仍加载失败: {e!r}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
