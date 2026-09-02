# -*- coding: utf-8 -*-
"""noise_clean_lab.py — 本地实验: 拓片噪点清洗算法对比.

输入样例: _sync_work/samples/{205890,148960,49482}_orig.png (篆书拓片, 高噪)
网格对比 6 种算法, 输出 grid PNG:
  [orig | A 去孤立斑 | B 形态学开 | C 亮度阈值+主域
   | D 斑纹背景估计-残差 | E 中值滤波背景减 | F 连通域+拓扑过滤]

关键认识 (v1 失败教训):
  拓片笔画会断裂成多个连通域, "最大连通域=主字"假设不成立.
  正确思路: 笔画是"细长结构", 墨斑是"团块/孤立点" ->
  用形态学/拓扑特征区分, 而不是连通域大小.
"""
import os
import sys
import glob
import numpy as np
from PIL import Image
from scipy import ndimage

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "_sync_work", "samples")
OUT = os.path.join(SAMPLES, "clean_lab")
os.makedirs(OUT, exist_ok=True)


# ---------- 工具 ----------
def to_ink(a):
    return a < 128


def ink_to_img(ink, orig):
    """修复后的 ink 掩码 -> 保留原灰度质感的图"""
    out = orig.copy()
    out[ink] = 255
    return out


# ---------- 算法 ----------
def algo_a_remove_isolated(orig):
    """A: 原v1思路(失败参照): 删非主连通域"""
    ink = to_ink(orig)
    lab, n = ndimage.label(ink)
    if n == 0:
        return orig
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    main = sizes.argmax()
    return ink_to_img(ink & (lab == main), orig)


def algo_b_open(orig):
    """B: 形态学开运算: 小结构(噪斑)被腐蚀掉, 笔画保留.
    斑点通常 2-6px, 用 5x5 开; 笔画宽 3-15px."""
    ink = to_ink(orig)
    st = np.ones((5, 5), bool)
    keep = ndimage.binary_opening(ink, structure=st)
    return ink_to_img(ink & ~keep, orig)  # 只删被开运算抹掉的部分


def algo_c_open_close(orig):
    """C: 开运算保留笔画骨架级结构, 再闭运算弥合笔画内断裂"""
    ink = to_ink(orig)
    st = np.ones((4, 4), bool)
    keep = ndimage.binary_opening(ink, structure=st)
    keep = ndimage.binary_closing(keep, structure=st)
    # 原图里属于 keep 的保留, 其余清白; 再把 keep 内的孔洞不填 (保真实质感)
    return ink_to_img(ink & ~keep, orig)


def algo_d_bg_subtract(orig):
    """D: 背景估计-残差: 大核最大值滤波估计斑纹背景亮度场,
    原图-背景 归一后重新阈值 -> 斑纹被压平, 笔画凸显."""
    # max filter ≈ 膨胀: 找局部最亮(纸面), 斑纹是暗点会被局部最亮覆盖
    bg = ndimage.maximum_filter(orig, size=31)
    # 归一化: orig/bg -> 斑纹区域变亮(接近纸面), 笔画保持暗
    norm = (orig.astype(np.float32) + 1) / (bg.astype(np.float32) + 1)
    ink2 = norm < 0.55  # 笔画(暗)且背景处(被提亮)不再触发
    ink2 = ndimage.binary_opening(ink2, structure=np.ones((2, 2), bool))
    return ink_to_img(ink2, orig)


def algo_e_median_bg(orig):
    """E: 中值背景估计 + 残差阈值 (更稳, 不放大孤立暗点)"""
    bg = ndimage.median_filter(orig, size=21)
    norm = orig.astype(np.float32) - bg.astype(np.float32)
    ink2 = norm > 40  # 显著比局部背景暗 = 笔画
    ink2 = ndimage.binary_closing(ink2, structure=np.ones((3, 3), bool))
    return ink_to_img(ink2, orig)


def algo_f_topo(orig):
    """F: 连通域 + 拓扑过滤: 保留 [面积大] 或 [细长比高] 的域;
    删 [小且圆] 的域 (墨斑特征). 细长比 = 面积/外接框面积 低 + 面积/周长 高"""
    ink = to_ink(orig)
    lab, n = ndimage.label(ink)
    if n == 0:
        return orig
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    keep_mask = np.zeros_like(ink)
    H, W = ink.shape
    for lb in range(1, n + 1):
        comp = lab == lb
        area = int(comp.sum())
        if area < 12:      # 极小点直接删
            continue
        ys, xs = np.where(comp)
        bh, bw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        fill = area / (bh * bw)          # 填充率: 团块高, 细长笔画低
        # 主字整体很大, 保留; 小而"方"的(填充率高)是斑纹团块
        if area > 0.02 * H * W:          # 大块 = 主字部件, 保留
            keep_mask |= comp
        elif fill < 0.45 and area > 40:  # 细长 = 笔画段
            keep_mask |= comp
        # 其余: 小而方 -> 删
    return ink_to_img(keep_mask, orig)


ALGOS = [
    ("A_isolated", algo_a_remove_isolated),
    ("B_open5", algo_b_open),
    ("C_open_close", algo_c_open_close),
    ("D_bgmax", algo_d_bg_subtract),
    ("E_median", algo_e_median_bg),
    ("F_topo", algo_f_topo),
    ("G_hyst100", lambda o: algo_g_hysteresis(o, 100, 128)),
    ("H_hyst110", lambda o: algo_g_hysteresis(o, 110, 140)),
    ("I_hyst+E", lambda o: with_edge_band(lambda x: algo_g_hysteresis(x, 100, 128), o)),
    ("J_topo+hyst", lambda o: with_edge_band(lambda x: algo_j_topo_hyst(x), o)),
    ("K_E+topo", lambda o: with_edge_band(lambda x: algo_k_median_topo(x), o)),
    ("L_res_hyst", lambda o: with_edge_band(lambda x: algo_l_residual_hyst(x), o)),
]


def algo_k_median_topo(orig):
    """K: 中值残差(40) + 拓扑过滤(删小而方的残斑)"""
    m = np.asarray(algo_e_median_bg(orig), dtype=np.uint8)
    return algo_f_topo(m)


def algo_l_residual_hyst(orig):
    """L: 残差图上做滞后阈值: strong=res>60 种子, weak=res>35 连通保留"""
    bg = ndimage.median_filter(orig, size=21)
    res = orig.astype(np.float32) - bg.astype(np.float32)
    strong = res > 60
    weak = res > 35
    lab, n = ndimage.label(weak)
    if n == 0:
        return orig
    seeds = np.unique(lab[strong])
    seeds = seeds[seeds > 0]
    keep = np.isin(lab, seeds)
    return ink_to_img(keep, orig)


def algo_g_hysteresis(orig, t_strong, t_weak):
    """G: 滞后阈值 (Canny 思路): 深墨=笔画种子, 中灰仅在与深墨连通时保留.
    斑纹(中灰/孤立)被剔, 断笔画(与深墨相连)保住."""
    strong = orig < t_strong
    weak = orig < t_weak
    lab, n = ndimage.label(weak)
    keep = np.zeros_like(weak)
    # 含 strong 像素的 weak 连通域整体保留
    seed_labels = np.unique(lab[strong])
    seed_labels = seed_labels[seed_labels > 0]
    keep = np.isin(lab, seed_labels)
    return ink_to_img(keep, orig)


def with_edge_band(fn, orig, band=10):
    """+ 边缘环带清除: 修复结果在边界 band px 内的前景全部清白(裁切残片)"""
    fixed = np.asarray(fn(orig), dtype=np.uint8)
    ink = fixed < 128
    b = band
    border = np.zeros_like(ink)
    border[:b, :] = True; border[-b:, :] = True
    border[:, :b] = True; border[:, -b:] = True
    fixed[ink & border] = 255
    return fixed


def algo_j_topo_hyst(orig):
    """J: 滞后阈值后再跑拓扑过滤(删小而方的残留斑)"""
    g = np.asarray(algo_g_hysteresis(orig, 100, 128), dtype=np.uint8)
    return algo_f_topo(g)


def main():
    cases = []
    for p in sorted(glob.glob(os.path.join(SAMPLES, "*_orig.png"))):
        iid = os.path.basename(p).replace("_orig.png", "")
        cases.append((iid, p))
    print(f"cases: {[c[0] for c in cases]}")

    for iid, path in cases:
        orig = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        tiles = [("orig", orig)]
        for name, fn in ALGOS:
            try:
                tiles.append((name, fn(orig)))
            except Exception as e:
                print(f"  {iid}/{name} FAIL: {e}")
        # grid
        cols = len(tiles)
        H, W = orig.shape
        sc = 0.5
        th, tw = int(H * sc), int(W * sc)
        canvas = Image.new("L", (tw * cols + 8 * (cols - 1), th + 20), 255)
        from PIL import ImageDraw
        d = ImageDraw.Draw(canvas)
        for i, (name, im) in enumerate(tiles):
            x = i * (tw + 8)
            d.text((x + 2, 2), name, fill=0)
            canvas.paste(Image.fromarray(im).resize((tw, th)), (x, 20))
        out = os.path.join(OUT, f"lab_{iid}.png")
        canvas.save(out)
        print(f"saved {out}")


if __name__ == "__main__":
    main()