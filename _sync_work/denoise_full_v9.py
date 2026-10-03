# -*- coding: utf-8 -*-
"""denoise_full_v9.py — denoise_v3.1 全量: data/imgs/final_imgs_fame_v8 (只读) → data/imgs/final_imgs_fame_v9 (新建)

v3.0 教训 (2026-09-03): "删贴边非主大块"把篆/隶/草的**分离部件**当装裱删了
  (774 张多部件字只剩单部件, 如 219580/151267/175633)。主 CC 保留率指标是假阴性。
v3.1 修复: 贴边块必须同时满足
    (a) 实心: fill = area/bbox_area > 0.70   (装裱黑边是实心矩形; 分离笔画是弯曲细条, fill 低)
    (b) 横贯一边: bbox 宽或高 >= 50% 图宽/高
  才判为装裱删除。并新增部件保留验证 (big_cc_before/after, 面积>=1%N 的 CC 数)。

安全约束:
  * 绝不写入 data/imgs/final_imgs_fame_v8 —— 全程只读
  * 输出全部落到 data/imgs/final_imgs_fame_v9 (新建目录)
  * 断言 IMG_ROOT != OUT_ROOT
  * 不缩放/不 letterbox —— 保持 256x256, 只"擦除"污染像素, 输出纯二值

算法 (denoise_v3.1):
  1) 极性归一
  2) 删孤立小点: 非主 CC 且 面积 < 0.05%N
  3) 删装裱黑边: 非主 CC 且 贴 8px 边带 且 面积 >= 0.1%N 且 fill>0.70 且 横贯一边
  4) 毛刺去除: skeletonize → 剪端点 PRUNE_L 次 → reconstruction (笔画宽度不变)

产物:
  data/imgs/final_imgs_fame_v9/*.png
  assets/clean_report_v9.csv        (含 big_cc_before/after 部件保留验证)
  assets/train_fame_clean_v9.csv

用法(远程):
  /opt/conda/bin/python _sync_work/denoise_full_v9.py --workers 8
"""
import os
import sys
import csv
import argparse
import numpy as np
from multiprocessing import Pool
from PIL import Image
from scipy import ndimage

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SMALL_FRAC = 0.0005
EDGE_THR_FRAC = 0.001
MID_FRAC = 0.01
PRUNE_L = 8
FILL_THR = 0.70      # 实心度阈值 (装裱黑边 >0.7; 分离笔画 <0.5)
PROTECT_LEN = 12     # v3.4: 骨架总长 <=12px 的部件(楷书"点"等短笔画)不剪
IMG_ROOT = "data/imgs/final_imgs_fame_v8"
OUT_ROOT = "data/imgs/final_imgs_fame_v9"

_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)


def _edge_mask(H, W, b=8):
    bm = np.zeros((H, W), bool)
    bm[:b, :] = True
    bm[-b:, :] = True
    bm[:, :b] = True
    bm[:, -b:] = True
    return bm


def _main_frac(lab, nlab, n_ink):
    if nlab == 0 or n_ink == 0:
        return 0.0
    return float(np.bincount(lab.ravel())[1:].max()) / float(n_ink)


def _big_cc_count(ink, N):
    """面积 >= 1%N 的连通域个数 (字形部件数, 验证部件不被误删)."""
    lab, n = ndimage.label(ink)
    if n == 0:
        return 0
    areas = ndimage.sum(ink, lab, range(1, n + 1))
    return int((areas >= MID_FRAC * N).sum())


def denoise_v3_mask(a):
    """返回 (clean_ink_bool, info, before_stats)。"""
    ink = a < 128
    inverted = bool(ink.mean() > 0.5)
    if inverted:
        ink = ~ink
    ink = ink.astype(bool)
    H, W = ink.shape
    N = float(H * W)

    lab0, n0 = ndimage.label(ink)
    before = (int(ink.sum()), int(n0), _main_frac(lab0, n0, int(ink.sum())))

    info = {"removed_small": 0, "removed_edge": 0, "prune_iters": 0,
            "aggressive": 0, "inverted": int(inverted)}
    if n0 == 0:
        return ink, info, before

    bm = _edge_mask(H, W)
    areas = ndimage.sum(ink, lab0, range(1, n0 + 1))
    main = int(np.argmax(areas)) + 1
    keep = (lab0 == main)
    small_thr = max(SMALL_FRAC * N, 8)
    edge_thr = max(EDGE_THR_FRAC * N, 24)
    # v3.3: 只有单部件字 (big_cc==1) 才删贴边实心大块; 多部件字绝不动块
    #   v3.0 教训: 直接删贴边大块 → 误删篆/隶分离部件 (774 张)
    #   v3.1 教训: fill+横贯判定 → 篆书粗横画仍中招 (211 张)
    #   v3.2 教训: 距离判定 → 楷书部件间隔与装裱间隔重叠, 更糟
    big_cnt = int((areas >= MID_FRAC * N).sum())
    aggressive = (big_cnt == 1)
    info["aggressive"] = int(aggressive)
    mount_thr = 1000      # 只删 >=1000px (~1.5%N) 的块, 小笔画绝不碰
    for ci in range(1, n0 + 1):
        if ci == main:
            continue
        comp = (lab0 == ci)
        area = areas[ci - 1]
        if area < small_thr:
            info["removed_small"] += 1
            continue
        if aggressive and area >= mount_thr and bool((comp & bm).any()):
            ys, xs = np.where(comp)
            bw = int(xs.max() - xs.min() + 1)
            bh = int(ys.max() - ys.min() + 1)
            fill = area / float(bw * bh)
            if fill > FILL_THR:
                info["removed_edge"] += 1
                continue
        keep |= comp
    ink2 = keep

    # v3.7: 温和形态学平滑 —— 2x2 opening 去除 <2px 毛刺/噪线 (保留 >=2px 笔画),
    #   2x2 closing 填补 1px 断缝。
    #   弃用骨架剪枝: v3.5 reconstruction 因连通性必然恢复毛刺 (99.97% 无变化);
    #   v3.6 有限膨胀普遍收缩笔画边缘 (颜体粗笔画半宽 > GROW)。
    if ink2.any():
        st = np.ones((2, 2), dtype=bool)
        ink3 = ndimage.binary_opening(ink2, structure=st)
        ink3 = ndimage.binary_closing(ink3, structure=st)
    else:
        ink3 = ink2
    return ink3, info, before


def work(rel):
    src = os.path.join(IMG_ROOT, rel)
    dst = os.path.join(OUT_ROOT, rel)
    iid = os.path.splitext(os.path.basename(rel))[0]
    try:
        a = np.asarray(Image.open(src).convert("L"), dtype=np.uint8)
    except Exception as e:
        return {"img_id": iid, "ok": 0, "err": str(e)[:60]}
    try:
        clean, info, before = denoise_v3_mask(a)
    except Exception as e:
        return {"img_id": iid, "ok": 0, "err": str(e)[:60]}

    out = np.where(clean, 0, 255).astype(np.uint8)
    d = os.path.dirname(dst)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    Image.fromarray(out).save(dst)

    # 部件保留验证 (v3.1 新增, 防 v3.0 式假阴性)
    N = float(a.shape[0] * a.shape[1])
    ink_b = a < 128
    if ink_b.mean() > 0.5:
        ink_b = ~ink_b
    big_b = _big_cc_count(ink_b, N)
    big_a = _big_cc_count(clean, N)

    lab1, n1 = ndimage.label(clean)
    n_ink1 = int(clean.sum())
    changed = int((out != a).sum())
    return {
        "img_id": iid, "ok": 1, "err": "",
        "ink_before": before[0], "ink_after": n_ink1,
        "ink_delta": round((n_ink1 - before[0]) / max(before[0], 1), 6),
        "n_cc_before": before[1], "n_cc_after": int(n1),
        "main_frac_before": round(before[2], 6),
        "main_frac_after": round(_main_frac(lab1, n1, n_ink1), 6),
        "big_cc_before": big_b, "big_cc_after": big_a,
        "removed_small": info["removed_small"],
        "removed_edge": info["removed_edge"],
        "prune_iters": info["prune_iters"],
        "aggressive": info["aggressive"],
        "inverted": info["inverted"],
        "changed_px": changed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="assets/train_fame_clean_v8.csv")
    ap.add_argument("--img-root", default=IMG_ROOT)
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--report", default="assets/clean_report_v9.csv")
    ap.add_argument("--out-csv", default="assets/train_fame_clean_v9.csv")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    assert os.path.abspath(args.img_root) != os.path.abspath(args.out_root), \
        "img_root 与 out_root 不能相同 (会覆盖原数据)"
    os.makedirs(args.out_root, exist_ok=True)

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    rels = sorted(set(os.path.basename(r["image_path"]) for r in rows))
    if args.limit:
        rels = rels[:args.limit]
    print(f"[v9] {len(rels)} images: {args.img_root} (只读) -> {args.out_root}",
          flush=True)

    FIELDS = ["img_id", "ok", "err", "ink_before", "ink_after", "ink_delta",
              "n_cc_before", "n_cc_after", "main_frac_before",
              "main_frac_after", "big_cc_before", "big_cc_after",
              "removed_small", "removed_edge", "prune_iters",
              "aggressive", "inverted", "changed_px"]

    out_rows = []
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(work, rels, chunksize=64)):
            out_rows.append(res)
            if (i + 1) % 5000 == 0:
                print(f"  {i+1}/{len(rels)}", flush=True)

    with open(args.report, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"[done] report -> {args.report} ({len(out_rows)} rows)", flush=True)

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        cols = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            r2 = dict(r)
            r2["image_path"] = os.path.join(args.out_root,
                                            os.path.basename(r["image_path"]))
            w.writerow(r2)
    print(f"[done] csv -> {args.out_csv}", flush=True)

    ok = [r for r in out_rows if r.get("ok") == 1]
    if ok:
        n = len(ok)
        print(f"\n=== 汇总 ({n} ok / {len(out_rows)}) ===")
        ink_d = np.array([r["ink_delta"] for r in ok], dtype=float)
        print(f"  ink_delta mean={ink_d.mean():.4f} median={np.median(ink_d):.4f}")
        print(f"  |ink_delta|>5%:  {int((np.abs(ink_d)>0.05).sum())} "
              f"({100*(np.abs(ink_d)>0.05).mean():.2f}%)")
        print(f"  ink_delta<-0.3:  {int((ink_d<-0.3).sum())} "
              f"({100*(ink_d<-0.3).mean():.2f}%)")
        print(f"  main_keep<0.95:  "
              f"{sum(1 for r in ok if r['main_frac_after'] < r['main_frac_before']*0.95)}")
        # 部件保留 (核心安全指标)
        lost = [r for r in ok if r["big_cc_after"] < r["big_cc_before"]]
        print(f"  big_cc 丢失 (部件被删): {len(lost)} ({100*len(lost)/n:.2f}%)  ← 必须为 0")
        for r in lost[:10]:
            print(f"    [!] {r['img_id']}: big_cc {r['big_cc_before']}→{r['big_cc_after']} "
                  f"ink_delta={r['ink_delta']:.3f}")
        print(f"  changed_px=0: {sum(1 for r in ok if r['changed_px']==0)} "
              f"({100*sum(1 for r in ok if r['changed_px']==0)/n:.2f}%)")
        print(f"  removed_edge 总数: {sum(r['removed_edge'] for r in ok)}")
        print(f"  removed_small 总数: {sum(r['removed_small'] for r in ok)}")
        aggr = sum(1 for r in ok if r.get("aggressive") == 1)
        print(f"  aggressive(单部件字, 允许删块): {aggr} ({100*aggr/n:.2f}%)")


if __name__ == "__main__":
    main()
