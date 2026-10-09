# -*- coding: utf-8 -*-
"""fetch_net_for_lpips.py — 取 LPIPS 需要的 backbone 权重, 支持**断点续传 + 尺寸/哈希校验**。

本机实测到的真实故障 (不是"网络断了"):
    从 download.pytorch.org 直下会被**截断** ——
      alexnet 官方 233.1 MiB, 实拿 232.9 MiB (少 0.6 MiB);
      vgg16   官方 553.4 MiB, 实拿 ... 且留下 3 个 .partial。
    装的时候报 `PytorchStreamReader failed reading zip archive:
    failed finding central directory` (zip 中央目录在文件尾部, 截断即毁)。
    torch 自带的下载器只报 `invalid hash value`, 不告诉你"少了多少字节"。

本脚本的做法:
    1. 用 Range 请求**续传**到同一个临时文件, 每轮比对 Content-Length / 官方 size, 直到补齐;
    2. 校验 sha256 前缀 (torch 的 URL 文件名那 8 位就是 sha256 前缀);
    3. 通过后才放进 torch cache (放进去后 torch 不再校验);
    4. 最后做**合理性验证**: 同图(轻噪声)≈0, 异图(位移)显著>0。

用法:
    python tools/fetch_net_for_lpips.py --net alex
    python tools/fetch_net_for_lpips.py --net vgg --force
"""
import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import time
import urllib.request

NETS = {
    "alex": ("https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
             "alexnet-owt-7be5be79.pth", "7be5be79", 244_862_944),
    "vgg": ("https://download.pytorch.org/models/vgg16-397923af.pth",
            "vgg16-397923af.pth", "397923af", 553_433_080),
}
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
UA = {"User-Agent": "python-urllib/3"}
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def resume_download(url, want_size, max_rounds=40):
    """断点续传到临时文件, 直到大小 == want_size。返回临时文件路径或 None。"""
    tmp = tempfile.NamedTemporaryFile(suffix=".pth", delete=False).name
    got = 0
    round_no = 0
    while got < want_size and round_no < max_rounds:
        round_no += 1
        hdr = dict(UA)
        if got:
            hdr["Range"] = f"bytes={got}-"
        try:
            req = urllib.request.Request(url, headers=hdr)
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "ab" if got else "wb") as out:
                # 206 = 部分内容; 若服务器忽略 Range 返回 200, 则要从头写
                if got and r.status == 200:
                    print(f"\n[round {round_no}] 服务器不支持 Range (HTTP 200) -> 从头重来")
                    out.seek(0)
                    out.truncate()
                    got = 0
                clen = int(r.headers.get("Content-Length", 0) or 0)
                crange = r.headers.get("Content-Range", "")
                print(f"\n[round {round_no}] HTTP {r.status} len={clen} range={crange} "
                      f"(已有 {got/2**20:.1f} MiB)")
                while True:
                    c = r.read(1 << 20)
                    if not c:
                        break
                    out.write(c)
                    got += len(c)
                    print(f"\r         {got/2**20:7.2f}/{want_size/2**20:7.2f} MiB", end="", flush=True)
        except Exception as e:
            print(f"\n[warn] 第 {round_no} 轮出错: {e!r} -> 续传重试")
            time.sleep(2)
            continue
        if got >= want_size:
            break
        print(f"\n[warn] 本轮结束但未补齐 (差 {want_size-got} 字节) -> 续传")
        time.sleep(2)
    print()
    if got != want_size:
        print(f"[FAIL] 续传 {round_no} 轮仍未补齐: {got} != {want_size} (差 {want_size-got} 字节)")
        return None
    print(f"[ok] 尺寸齐全: {got} 字节 = {got/2**20:.2f} MiB")
    return tmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="alex", choices=list(NETS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    url, fname, want_prefix, want_size = NETS[a.net]
    dst = os.path.join(CACHE_DIR, fname)
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"[net] {a.net}\n[url] {url}\n[dst] {dst}")
    print(f"[官方] size={want_size} 字节 ({want_size/2**20:.2f} MiB)  sha256前缀={want_prefix}")

    for f in list(os.listdir(CACHE_DIR)):
        if fname.split("-")[0] in f and (f.endswith(".partial") or f == fname):
            p = os.path.join(CACHE_DIR, f)
            print(f"[clean] 删除 {f} ({os.path.getsize(p)/2**20:.2f} MiB)")
            os.remove(p)

    tmp = resume_download(url, want_size)
    if tmp is None:
        return 1
    h = sha256(tmp)
    if h.startswith(want_prefix):
        print(f"[hash] ✓ sha256={h[:16]}... 与官方前缀一致")
    else:
        print(f"[hash] ✗ sha256={h[:16]}... != {want_prefix} (尺寸对但内容不同)")
        os.remove(tmp)
        return 1
    shutil.move(tmp, dst)
    print(f"[put ] 已放入 {dst}")

    print("\n=== 合理性验证 (同图≈0 / 异图应显著>0) ===")
    try:
        import numpy as np
        import torch
        import lpips
        fn = lpips.LPIPS(net=a.net, verbose=False).eval()
        print(f"  ✓ lpips.LPIPS(net={a.net!r}) 加载成功")
        rng = np.random.default_rng(0)
        x = np.full((256, 256, 3), 255, np.uint8)
        x[80:150, 60:130] = 0
        y = np.full((256, 256, 3), 255, np.uint8)
        y[90:160, 150:220] = 0
        z = np.clip(x.astype(np.int16) + rng.integers(-6, 7, x.shape), 0, 255).astype(np.uint8)
        t = lambda arr: torch.from_numpy(arr).permute(2, 0, 1)[None].float() / 127.5 - 1.0
        with torch.no_grad():
            d_same = float(fn(t(x), t(z)).item())
            d_diff = float(fn(t(x), t(y)).item())
        print(f"  同图(轻噪声) = {d_same:.5f}  (应≈0)")
        print(f"  异图(位移)   = {d_diff:.5f}  (应显著>同图)")
        if d_diff > d_same and d_same < 0.05:
            print("  ✓ 权重可用")
            return 0
        print("  ✗ 距离不单调 -> 权重不可信")
        return 1
    except Exception as e:
        print(f"  ✗ 加载/验证失败: {e!r}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
