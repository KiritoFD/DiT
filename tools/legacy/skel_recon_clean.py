# -*- coding: utf-8 -*-
"""skel_recon_clean.py — 局部自适应骨架重建清洗 + 去噪评估指标.

=== 动机 ===
噪点在空间的本质特征: 离开字骨架的离群点.
但笔画粗细在同一字内、不同书体间差异极大 (篆隶粗, 行草细),
用"全局阈值距离"判离群会误杀粗笔画、放过细笔画旁的噪点.

本方法用**局部笔画宽度**做自适应:
  * 每个骨架点 s 有局部笔画半宽 w(s) (该点处笔画内接圆半径)
  * 重建合法笔画区 R = ∪_{s∈skel} disk(s, tau * w(s))
  * 落在 R 之外的前景 = 离群噪点

关键点 — 骨架必须用 1px skeletonize 而非 dil3:
  skeletonize 把 3x3 噪点团块压成 1px (过滤);
  dil3 会把噪点残留放大成 7x7 (放大 49 倍). 见 --demo-dil3.

!!! 重要: 本文件的**清洗**能力已被合成噪点基准证伪, 请勿用作清洗 !!!
见 tools/noise_bench.py 结果: A_skel_recon 的 NR(去噪率) = 0.0%.

失效机理 (根本性, 非调参可解):
  **任何连通域都有骨架.** 噪点团块经 skeletonize 后自身产生骨架点,
  而局部宽度 w(s) 又取自该团块自身 -> 重建区 R 把噪点完整包住.
  于是"到骨架的距离"对噪点恒小, 噪点在自己的骨架上, 永远不是离群点.
  (真实样本上那 1~6% 删除量, 实际功劳来自 binary_opening 预处理, 而非骨架判据.)

结论: "噪点 = 离开骨架的离群点" 这一直觉在**几何**上不成立.
      噪点不是散点, 而是自带完整骨架的小结构.
      有效的判据应是**结构归属**(该结构是否属于字), 而非**到骨架的距离**.

本文件保留的价值: OSIR 作为**筛选**指标仍有效 (见下), 因为它衡量
"结构外零散像素占比", 对极细碎沙粒状噪点仍有区分度 (实测 p50=0%,
p90=2.62%, p95=3.93%). 但不要用它评估清洗效果 (自证循环).

=== 指标 (用于筛选 + 评估) ===
OSIR (Off-Skeleton Ink Ratio) 离群墨占比:
    OSIR = |ink \\ R| / |ink|
  越高 = 图越脏. 用于**筛选**脏样本 (黑名单独立阈值).
  注意: 不可用于评估清洗效果——用 R 定义又删 R 外, 必然趋近 0 (自证).
  严格评估请用 tools/noise_bench.py 的 NR/SR (合成噪点, 有 ground truth).

SF (Stroke Fidelity) 笔画保真度:
    SF = |ink ∩ R ∩ kept| / |ink ∩ R|
  清洗后仍保留的"合法笔画像素"比例. 用于**评估**清洗是否误伤.
  任何去噪都能把 OSIR 打到 0 (全删光即可), 所以必须配 SF 防作弊.

=== 用法 ===
  # 单图可视化对比
  python tools/skel_recon_clean.py --sample --k 1.8
  # 全量/抽样评估 OSIR 分布
  python tools/skel_recon_clean.py --scan --n 200
"""
import os
import sys
import glob
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.morphology import skeletonize

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "_sync_work", "samples")
IMG_ROOT = os.path.join(ROOT, "final_imgs_256")
OUT = os.path.join(SAMPLES, "skel_recon")


# ---------------- 骨架与局部笔画宽度 ----------------
def robust_skeleton(ink, open_size=2):
    """1px 骨架 (有过滤作用). open_size 小核仅去 1-2px 孤立点.

    注意: 绝不用 dil3 做锚——那会把噪点残留放大 49 倍."""
    if open_size > 1:
        st = np.ones((open_size, open_size), bool)
        cleaned = ndimage.binary_opening(ink, structure=st)
        if not cleaned.any():
            cleaned = ink
    else:
        cleaned = ink
    sk = skeletonize(cleaned)
    if not sk.any():                       # 极端: 字被开没了, 回退
        sk = skeletonize(ink)
    return cleaned, sk


def disk(r):
    rad = max(int(np.ceil(r)), 1)
    y, x = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    return (x * x + y * y) <= r * r


def reconstruct(ink, sk, tau=1.8, bands=8):
    """R = ∪_{s∈skel} disk(s, tau*w(s)); 按 w 分档膨胀近似变半径膨胀.

    dt: 原始 ink 的距离变换 = 每个前景像素到最近背景的距离 (内接圆半径).
    骨架点处的 dt 即该处笔画半宽 w(s) —— 天然局部自适应,
    粗笔画 (篆隶) 合法区大, 细笔画 (行草) 合法区小."""
    dt = ndimage.distance_transform_edt(ink)
    R = np.zeros_like(ink)
    ws = dt[sk]
    if ws.size == 0:
        return R, dt
    wmax = max(float(ws.max()), 1e-6)
    edges = np.linspace(0, wmax, bands + 1)
    for i in range(bands):
        lo, hi = edges[i], edges[i + 1]
        sel = sk & (dt >= lo) & (dt < hi) if i < bands - 1 else sk & (dt >= lo)
        if not sel.any():
            continue
        r = tau * hi                       # 用档上界, 偏保守 (少删)
        R |= ndimage.binary_dilation(sel, structure=disk(r))
    return R, dt


def elong_exempt(ink, mask, fill_max=0.45, min_area=30):
    """细长连通域豁免: 断裂笔画段虽远离主骨架, 但细长 -> 保留."""
    exempt = np.zeros_like(mask)
    lab, n = ndimage.label(mask)
    if n == 0:
        return exempt
    for lb in range(1, n + 1):
        comp = lab == lb
        area = int(comp.sum())
        if area < min_area:
            continue
        ys, xs = np.where(comp)
        bh = ys.max() - ys.min() + 1
        bw = xs.max() - xs.min() + 1
        if area / max(bh * bw, 1) < fill_max:
            exempt |= comp
    return exempt


# ---------------- 指标 ----------------
def osir(ink):
    """离群墨占比 (越高越脏). 内部自算骨架/重建."""
    if not ink.any():
        return 0.0, None
    _, sk = robust_skeleton(ink)
    R, dt = reconstruct(ink, sk)
    off = ink & ~R
    return float(off.sum()) / float(ink.sum()), (sk, R, dt, off)


def stroke_fidelity(ink, kept):
    """笔画保真度 (越接近 1 越好). kept = 清洗后保留的前景掩码."""
    _, sk = robust_skeleton(ink)
    R, _ = reconstruct(ink, sk)
    legal = ink & R
    if not legal.any():
        return 1.0
    return float((legal & kept).sum()) / float(legal.sum())


# ---------------- 清洗 ----------------
def clean(orig_u8, tau=1.8):
    """返回 (cleaned_u8, info)."""
    ink = orig_u8 < 128
    _, sk = robust_skeleton(ink)
    R, dt = reconstruct(ink, sk, tau=tau)
    off = ink & ~R
    exempt = elong_exempt(ink, off)
    drop = off & ~exempt
    out = orig_u8.copy()
    out[drop] = 255
    info = dict(ink=int(ink.sum()), sk=sk, R=R, dt=dt, off=off,
                exempt=exempt, drop=drop)
    return out, info


# ---------------- 可视化 ----------------
def to_rgb(u8):
    return np.stack([u8] * 3, -1)


def mark_red(u8, m):
    rgb = to_rgb(u8).copy()
    rgb[m] = [255, 0, 0]
    return rgb


def grid(tiles, path, th=256):
    tw = th
    cols = len(tiles)
    canvas = Image.new("RGB", (tw * cols + 8 * (cols - 1), th + 22), (255,) * 3)
    d = ImageDraw.Draw(canvas)
    for i, (name, im) in enumerate(tiles):
        x = i * (tw + 8)
        d.text((x + 2, 3), name, fill=(0, 0, 0))
        canvas.paste(Image.fromarray(im).resize((tw, th)), (x, 22))
    canvas.save(path)


# ---------------- 主流程 ----------------
def run_sample(tau):
    os.makedirs(OUT, exist_ok=True)
    cases = [(os.path.basename(p).replace("_orig.png", ""), p)
             for p in sorted(glob.glob(os.path.join(SAMPLES, "*_orig.png")))]
    print(f"[sample] {len(cases)} cases  tau={tau}")
    for iid, p in cases:
        orig = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
        o0, aux = osir(orig < 128)
        cl, info = clean(orig, tau=tau)
        o1, _ = osir(cl < 128)
        sf = stroke_fidelity(orig < 128, cl < 128)
        n = info["ink"]
        print(f"\n=== {iid}  ink={n}")
        print(f"  OSIR  {o0*100:6.2f}%  ->  {o1*100:6.2f}%")
        print(f"  SF(保真) {sf*100:6.2f}%   删 {int(info['drop'].sum())} px "
              f"({info['drop'].sum()/max(n,1)*100:.2f}%)  豁免 {int(info['exempt'].sum())} px")
        grid([
            ("orig", to_rgb(orig)),
            ("1px skel", to_rgb(np.where(info["sk"], 0, 255).astype("uint8"))),
            ("R 合法区", to_rgb(np.where(info["R"], 0, 255).astype("uint8"))),
            ("离群(红)", mark_red(orig, info["drop"])),
            ("cleaned", to_rgb(cl)),
        ], os.path.join(OUT, f"recon_{iid}_tau{tau}.png"))
    print(f"\n-> {OUT}")


def run_demo_dil3():
    """演示: skeletonize 压缩噪点 vs dil3 放大噪点."""
    # 合成: 一条笔画 + 若干孤立噪点团块
    a = np.full((128, 128), 255, np.uint8)
    a[60:64, 20:108] = 0                       # 水平笔画 (宽4)
    rng = np.random.default_rng(0)
    for _ in range(12):                        # 孤立 3x3 噪点
        y, x = rng.integers(5, 118, 2)
        a[y:y + 3, x:x + 3] = 0
    ink = a < 128
    sk = skeletonize(ink)
    st = ndimage.generate_binary_structure(2, 2)
    d3 = ndimage.binary_dilation(sk, structure=st, iterations=3)
    print("[demo] 合成图: 1 条笔画 + 12 个 3x3 孤立噪点团块")
    print(f"  原噪点面积      12*9  = {12*9} px")
    print(f"  skeletonize 后  {int(sk.sum())} px  (含噪点残留)  <- 压缩")
    print(f"  dil3 后         {int(d3.sum())} px              <- 放大")
    grid([("orig(笔画+噪点)", to_rgb(a)),
          ("1px skeletonize", to_rgb(np.where(sk, 0, 255).astype("uint8"))),
          ("dil3", to_rgb(np.where(d3, 0, 255).astype("uint8")))],
         os.path.join(OUT, "demo_dil3_amp.png"), th=128)
    print(f"  -> {OUT}/demo_dil3_amp.png")


def run_scan(n, tau):
    """抽样扫描 OSIR 分布, 给出筛选阈值参考."""
    files = [f for f in os.listdir(IMG_ROOT) if f.endswith(".png")]
    rng = random.Random(0)
    pick = rng.sample(files, min(n, len(files)))
    vals = []
    for i, f in enumerate(pick):
        try:
            a = np.asarray(Image.open(os.path.join(IMG_ROOT, f)).convert("L"),
                           dtype=np.uint8)
            ink = a < 128
            if ink.sum() < 50:
                continue
            v, _ = osir(ink)
            vals.append((v, f))
        except Exception:
            continue
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pick)}", flush=True)
    vals.sort()
    arr = np.array([v for v, _ in vals])
    print(f"\n[scan] n={len(arr)}  OSIR分布")
    for q in (50, 75, 90, 95, 99):
        print(f"  p{q:<3d} = {np.percentile(arr, q)*100:6.2f}%")
    print(f"  mean = {arr.mean()*100:.2f}%   max = {arr.max()*100:.2f}%")
    print("\n  最脏 10 张 (筛选候选):")
    for v, f in vals[-10:][::-1]:
        print(f"    {f:>14s}  OSIR={v*100:6.2f}%")
    out = os.path.join(OUT, "osir_scan.csv")
    os.makedirs(OUT, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        import csv
        w = csv.writer(fh)
        w.writerow(["img_id", "osir"])
        for v, f in vals:
            w.writerow([f[:-4], f"{v:.6f}"])
    print(f"\n  -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="样本可视化")
    ap.add_argument("--scan", action="store_true", help="抽样扫描 OSIR 分布")
    ap.add_argument("--demo-dil3", action="store_true", help="演示 dil3 放大噪点")
    ap.add_argument("--tau", type=float, default=1.8, help="合法区半径 = tau*w(s)")
    ap.add_argument("--n", type=int, default=200, help="scan 抽样数")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.demo_dil3:
        run_demo_dil3()
    if args.sample:
        run_sample(args.tau)
    if args.scan:
        run_scan(args.n, args.tau)
    if not (args.sample or args.scan or args.demo_dil3):
        run_demo_dil3()


if __name__ == "__main__":
    main()
