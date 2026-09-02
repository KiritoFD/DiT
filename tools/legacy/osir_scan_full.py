# -*- coding: utf-8 -*-
"""osir_scan_full.py — 全量 OSIR 扫描 + 数据覆盖收缩分析.

回答三问:
  Q1 能否识别脏样本?        -> OSIR 分布区分度 + top 脏图可视化
  Q2 删掉会否剧烈收缩覆盖?  -> 按 (书体,字) 聚合, 算各阈值下归零对数
  Q3 可替代性?              -> 结合每对样本数, 分级: 多样本可剔, 单样本只能洗

注意 OSIR 只用于**筛选**, 不用于评估清洗效果 (自证循环, 见 skel_recon_clean.py).

用法:
  python tools/osir_scan_full.py scan        # 扫本地 final_imgs_256 (较慢, 多进程)
  python tools/osir_scan_full.py report      # 覆盖分析 (快, 需先 scan)
"""
import os
import sys
import csv
import glob
import argparse
import collections
import numpy as np
from PIL import Image, ImageDraw

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
IMG_ROOT = os.path.join(ROOT, "final_imgs_256")
CSV_IN = os.path.join(ROOT, "5script", "train_fame.csv")
OUT = os.path.join(ROOT, "_sync_work", "samples", "osir_scan")
OSIR_CSV = os.path.join(OUT, "osir_full.csv")


def _work(path):
    try:
        from tools.skel_recon_clean import osir, robust_skeleton, reconstruct
        a = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        ink = a < 128
        if ink.sum() < 100:
            return (os.path.basename(path)[:-4], -1.0)
        v, _ = osir(ink)
        return (os.path.basename(path)[:-4], float(v))
    except Exception:
        return (os.path.basename(path)[:-4], -1.0)


def cmd_scan(workers):
    from multiprocessing import Pool
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(IMG_ROOT, "*.png")))
    print(f"[scan] {len(files)} images, workers={workers}", flush=True)
    rows = []
    with Pool(workers) as p:
        for i, r in enumerate(p.imap_unordered(_work, files, chunksize=32)):
            if r[1] >= 0:
                rows.append(r)
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(files)}", flush=True)
    with open(OSIR_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["img_id", "osir"])
        for iid, v in sorted(rows):
            w.writerow([iid, f"{v:.6f}"])
    arr = np.array([v for _, v in rows])
    print(f"[scan] {len(rows)} -> {OSIR_CSV}")
    for q in (50, 75, 90, 95, 98, 99):
        print(f"  p{q:<3d} = {np.percentile(arr, q)*100:6.2f}%")
    print(f"  mean={arr.mean()*100:.2f}%  max={arr.max()*100:.2f}%")


def cmd_report():
    if not os.path.isfile(OSIR_CSV):
        print("! 先跑 scan")
        return
    osir = {}
    with open(OSIR_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            osir[r["img_id"]] = float(r["osir"])

    rows = list(csv.DictReader(open(CSV_IN, encoding="utf-8")))
    # pair -> [img_id...];  书家覆盖
    pair_imgs = collections.defaultdict(list)
    img_pair, img_callig = {}, {}
    for r in rows:
        iid = os.path.basename(r["image_path"])[:-4]
        key = r["script"] + "|" + r["character"]
        pair_imgs[key].append(iid)
        img_pair[iid] = key
        img_callig[iid] = r["calligrapher"]

    have = set(i for i in img_pair if i in osir)
    print(f"[data] 样本 {len(rows)}  对 {len(pair_imgs)}  "
          f"已扫 {len(have)}/{len(img_pair)} ({len(have)/len(img_pair)*100:.1f}%)")

    # 每对样本数分布
    cnt = np.array([len(v) for v in pair_imgs.values()])
    print(f"[覆盖] 每对样本数: mean={cnt.mean():.2f}  "
          f"==1: {(cnt==1).mean()*100:.1f}%  >=2: {(cnt>=2).mean()*100:.1f}%")

    arr = np.array(sorted(osir.values()))
    print("\n" + "=" * 78)
    print(f"{'剔除阈值':<12s}{'剔除张数':>9s}{'占比':>8s}{'归零对数':>10s}"
          f"{'覆盖损失':>10s}{'书家损失':>10s}")
    print("=" * 78)
    base_pairs = len(pair_imgs)
    base_calligs = len(set(img_callig.values()))
    for q in (90, 95, 98, 99):
        t = np.percentile(arr, q)
        drop = {i for i, v in osir.items() if v > t}
        kept_pairs, lost = 0, 0
        for key, imgs in pair_imgs.items():
            rem = [i for i in imgs if i not in drop]
            if rem:
                kept_pairs += 1
            else:
                lost += 1
        calligs = set(img_callig[i] for i in have if i not in drop)
        print(f"p{q} ({t*100:4.2f}%){len(drop):9d}{len(drop)/len(osir)*100:7.2f}%"
              f"{lost:10d}{lost/base_pairs*100:9.2f}%"
              f"{(base_calligs-len(calligs)):10d}")
    print("=" * 78)

    # 分级策略: 单样本对不可剔, 多样本对可剔
    t95 = np.percentile(arr, 95)
    dirty_1, dirty_n = 0, 0
    for key, imgs in pair_imgs.items():
        d = [i for i in imgs if osir.get(i, 0) > t95]
        if not d:
            continue
        if len(imgs) == 1:
            dirty_1 += 1          # 独苗且脏 -> 只能洗, 不能剔
        else:
            dirty_n += 1          # 有替代 -> 可剔
    print(f"\n[分级 @p95]  脏样本所在对:")
    print(f"  独苗对(1张, 只能清洗): {dirty_1}")
    print(f"  多样本对(有替代, 可剔): {dirty_n}")
    n1 = sum(1 for v in pair_imgs.values() if len(v) == 1)
    print(f"  -> 独苗对总数 {n1}, 其中脏的占 {dirty_1/max(n1,1)*100:.1f}%")

    # top 脏图可视化
    top = sorted(osir.items(), key=lambda x: -x[1])[:6]
    print(f"\n[top 脏图] {[(i, round(v*100,1)) for i, v in top]}")
    _vis(top, img_pair)


def _vis(top, img_pair):
    from tools.skel_recon_clean import robust_skeleton, reconstruct
    os.makedirs(OUT, exist_ok=True)
    for iid, v in top:
        p = os.path.join(IMG_ROOT, f"{iid}.png")
        if not os.path.isfile(p):
            continue
        orig = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
        ink = orig < 128
        _, sk = robust_skeleton(ink)
        R, _ = reconstruct(ink, sk)
        off = ink & ~R
        rgb = np.stack([orig] * 3, -1).copy()
        rgb[off] = [255, 0, 0]
        canvas = Image.new("RGB", (256 * 3 + 16, 256 + 22), (255,) * 3)
        d = ImageDraw.Draw(canvas)
        d.text((2, 3), f"{iid} OSIR={v*100:.2f}% ({img_pair.get(iid,'?')})",
               fill=(0, 0, 0))
        canvas.paste(Image.fromarray(np.stack([orig] * 3, -1)), (0, 22))
        canvas.paste(Image.fromarray(np.where(sk, 0, 255).astype(np.uint8)
                                    .repeat(3).reshape(256, 256, 3)), (272, 22))
        canvas.paste(Image.fromarray(rgb), (544, 22))
        canvas.save(os.path.join(OUT, f"dirty_{iid}.png"))
    print(f"  可视化 -> {OUT}/dirty_*.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan", "report"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "scan":
        cmd_scan(a.workers)
    else:
        cmd_report()


if __name__ == "__main__":
    main()
