# -*- coding: utf-8 -*-
"""noise_bench.py — 清洗算法的合成噪点基准 (有 ground truth 的严格评估).

=== 为什么需要它 ===
OSIR 等指标用"骨架重建区 R"定义, 又删掉 R 外像素, 再拿它评估清洗 -> 自证循环.
要严格比较清洗算法, 必须有 ground truth.

本基准的做法:
  1. 选 OSIR≈0 的干净 GT 图作底图 (真笔画 = ink0)
  2. 在背景区注入合成噪点团块 (模拟拓片墨渍), 噪点位置已知 = GT
  3. 跑各清洗算法, 得到 kept 掩码
  4. 算:
     NR  (Noise Recall)    = |deleted ∩ noise| / |noise|   去噪能力 ↑
     SR  (Stroke Retain)   = |kept ∩ ink0| / |ink0|        笔画保真 ↑
     FPR (误删率)          = 1 - SR                         ↑越低越好

分三档难度 (噪点到字的距离), 因为"紧贴字的噪点"最难且最易误伤:
  easy   d>=10   远离字
  mid    d>=4
  hard   d>=1    紧贴字边缘

用法:
  python tools/noise_bench.py --n 20
  python tools/noise_bench.py --n 20 --vis   # 额外出可视化
"""
import os
import sys
import csv
import random
import argparse
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_ROOT = os.path.join(ROOT, "final_imgs_256")
OUT = os.path.join(ROOT, "_sync_work", "samples", "noise_bench")

sys.path.insert(0, ROOT)
from tools.skel_recon_clean import robust_skeleton, reconstruct, elong_exempt


# ---------------- 合成噪点 ----------------
def inject(noisy_u8, ink0, rng, n_blobs, rmin, rmax, dmin):
    """在离字 >= dmin 的背景区注入噪点团块. 返回 noise mask."""
    H, W = ink0.shape
    d = ndimage.distance_transform_edt(~ink0)   # 背景像素到最近前景(字)距离
    ys, xs = np.where((d >= dmin) & (d < dmin + 40))
    if len(ys) == 0:
        ys, xs = np.where(~ink0)
    if len(ys) == 0:
        return np.zeros_like(ink0)
    noise = np.zeros_like(ink0)
    out = noisy_u8.copy()
    for _ in range(n_blobs):
        i = rng.integers(0, len(ys))
        cy, cx = int(ys[i]), int(xs[i])
        r = float(rng.uniform(rmin, rmax))
        rad = max(int(np.ceil(r)), 1)
        y0, y1 = max(0, cy - rad), min(H, cy + rad + 1)
        x0, x1 = max(0, cx - rad), min(W, cx + rad + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        m = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r
        m &= ~ink0[y0:y1, x0:x1]                 # 严格不覆盖真笔画
        g = int(rng.uniform(40, 110))            # 墨渍灰度
        out[y0:y1, x0:x1][m] = np.minimum(out[y0:y1, x0:x1][m], g)
        noise[y0:y1, x0:x1] |= m
    return out, noise


# ---------------- 待评算法 ----------------
def a_skel_recon(u8, tau=1.8):
    """A: 局部自适应骨架重建 (本文方法)."""
    ink = u8 < 128
    _, sk = robust_skeleton(ink)
    R, _ = reconstruct(ink, sk, tau=tau)
    off = ink & ~R
    off &= ~elong_exempt(ink, off)
    return ink & ~off


def a_cc_main(u8):
    """B: 只保最大连通域 (现有 clean_gt_images.py 思路)."""
    ink = u8 < 128
    lab, n = ndimage.label(ink)
    if n == 0:
        return ink
    sz = np.bincount(lab.ravel())
    sz[0] = 0
    return lab == sz.argmax()


def a_cc_big(u8, frac=0.005):
    """C: 保面积 > frac*全图 的连通域 (B 的宽容版)."""
    ink = u8 < 128
    lab, n = ndimage.label(ink)
    if n == 0:
        return ink
    H, W = ink.shape
    keep = np.zeros_like(ink)
    for lb in range(1, n + 1):
        comp = lab == lb
        if comp.sum() >= frac * H * W:
            keep |= comp
    return keep


def a_open5(u8, k=5):
    """D: 形态学开 5x5 (去 <5px 细碎)."""
    ink = u8 < 128
    return ndimage.binary_opening(ink, structure=np.ones((k, k), bool))


def a_median_bg(u8, size=21, thr=40):
    """E: 中值背景估计 + 残差阈值.
    注意符号: 墨(笔画)比背景暗, 故判据是 (bg - orig) > thr.
    原 noise_clean_lab.py 写的是 (orig - bg) > thr —— 选的是比背景亮的像素,
    与注释相反, 是个 bug; 这里已修正."""
    bg = ndimage.median_filter(u8, size=size)
    res = bg.astype(np.float32) - u8.astype(np.float32)   # 正 = 比背景暗
    k = res > thr
    return ndimage.binary_closing(k, structure=np.ones((3, 3), bool))


def a_topo(u8, min_area=12, fill_max=0.45, big_frac=0.02, slim_area=40):
    """G: 连通域 + 形状(细长比) 过滤.
    关键区分 —— 噪点 vs 断裂笔画:
      噪点团块: 小而"圆"(填充率高)  -> 删
      断裂笔画: 细长(填充率低)      -> 保
      主字大块: 面积大              -> 保"""
    ink = u8 < 128
    lab, n = ndimage.label(ink)
    if n == 0:
        return ink
    H, W = ink.shape
    keep = np.zeros_like(ink)
    for lb in range(1, n + 1):
        comp = lab == lb
        area = int(comp.sum())
        if area < min_area:               # 极小点直接删
            continue
        ys, xs = np.where(comp)
        bh = ys.max() - ys.min() + 1
        bw = xs.max() - xs.min() + 1
        fill = area / max(bh * bw, 1)
        if area > big_frac * H * W:                    # 大块 = 主字部件
            keep |= comp
        elif fill < fill_max and area > slim_area:     # 细长 = 笔画段
            keep |= comp
        # 其余: 小而圆 -> 删
    return keep


def a_hyst(u8, t_s=100, t_w=128):
    """F: 滞后阈值 (深墨种子 + 连通中灰保留)."""
    strong = u8 < t_s
    weak = u8 < t_w
    lab, n = ndimage.label(weak)
    if n == 0:
        return weak
    seeds = np.unique(lab[strong])
    seeds = seeds[seeds > 0]
    return np.isin(lab, seeds)


ALGOS = [
    ("A_skel_recon", a_skel_recon),
    ("B_cc_main", a_cc_main),
    ("C_cc_big", a_cc_big),
    ("D_open5", a_open5),
    ("E_median_bg", a_median_bg),
    ("F_hyst", a_hyst),
    ("G_topo", a_topo),
]

DIFF = {"easy": 10, "mid": 4, "hard": 1}


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="底图数")
    ap.add_argument("--blobs", type=int, default=25, help="每图注入噪点数")
    ap.add_argument("--vis", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    # 1. 挑干净底图 (OSIR 低)
    files = [f for f in os.listdir(IMG_ROOT) if f.endswith(".png")]
    rng0 = random.Random(0)
    rng0.shuffle(files)
    from tools.skel_recon_clean import osir
    bases = []
    for f in files:
        try:
            a = np.asarray(Image.open(os.path.join(IMG_ROOT, f)).convert("L"),
                           dtype=np.uint8)
            ink = a < 128
            if ink.sum() < 2000:
                continue
            v, _ = osir(ink)
            if v < 0.005:                 # 干净
                bases.append(a)
        except Exception:
            continue
        if len(bases) >= args.n:
            break
    print(f"[bench] 干净底图 {len(bases)} 张, 每图注入 {args.blobs} 噪点\n")

    res = {name: {d: {"nr": [], "sr": []} for d in DIFF} for name, _ in ALGOS}

    for bi, base in enumerate(bases):
        ink0 = base < 128
        for dname, dmin in DIFF.items():
            rng = np.random.default_rng(1000 + bi * 10 + dmin)
            noisy, noise = inject(base, ink0, rng, args.blobs, 1.5, 5.0, dmin)
            if noise.sum() == 0:
                continue
            for name, fn in ALGOS:
                try:
                    kept = fn(noisy)
                except Exception:
                    continue
                deleted = (noisy < 128) & ~kept
                nr = deleted[noise].sum() / noise.sum()
                sr = (kept & ink0).sum() / ink0.sum()
                res[name][dname]["nr"].append(nr)
                res[name][dname]["sr"].append(sr)
        print(f"  base {bi+1}/{len(bases)}", flush=True)

    print("\n" + "=" * 68)
    print(f"{'算法':<15s}{'难度':<7s}{'NR去噪↑':>10s}{'SR保真↑':>10s}{'误删↓':>10s}")
    print("=" * 68)
    for name, _ in ALGOS:
        for dname in DIFF:
            nr = np.mean(res[name][dname]["nr"])
            sr = np.mean(res[name][dname]["sr"])
            print(f"{name:<15s}{dname:<7s}{nr*100:9.1f}%{sr*100:9.1f}%"
                  f"{(1-sr)*100:9.1f}%")
        print("-" * 68)

    with open(os.path.join(OUT, "bench.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["algo", "diff", "NR", "SR"])
        for name, _ in ALGOS:
            for dname in DIFF:
                w.writerow([name, dname,
                            f"{np.mean(res[name][dname]['nr']):.4f}",
                            f"{np.mean(res[name][dname]['sr']):.4f}"])
    print(f"\n-> {OUT}/bench.csv")


if __name__ == "__main__":
    main()
