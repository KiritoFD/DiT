# -*- coding: utf-8 -*-
"""clean_v2.py — 多方案 GT 图像清洗 + 质量评估对比。

支持三种方案（见 docs/system/28_data_cleaning_survey.md）:
  A (基线/conservative): 非主CC且面积<0.05%全图 -> 删; 边缘环带非主墨 -> 删
  B (参考字形 bbox):     在 A 基础上, 用标准字形 kai 的墨 bbox 判定——
                         非主CC中心若在 std bbox(膨胀 margin) **之外** -> 删(脏污);
                         在 bbox **之内** -> 保留(正常飞白/连笔)
  C (空间拓扑):          在 A 基础上, 非主CC到主CC的归一化距离 > 阈值 -> 删

前置处理（所有方案）:
  - 反相检测: ink_ratio>0.5 且 n_cc<=3 -> 整图反相 (白字黑底 -> 黑字白底)
  - 黑边 crop: border_bar=1 (某边近似实心黑条) -> 裁掉黑边 + 白边 pad

评估指标（逐图输出）:
  - main_keep:      主连通域保留率 (clean_main_area / orig_main_area)
  - ink_change:     墨量变化率
  - frag:           碎片度变化 (n_cc_after / n_cc_before)
  - small_cc_left:  修复后残留小CC数 (area<0.05%)
  - outside_ink:    修复后 bbox(字区域) 外的墨占比
  - inverted/crop:  是否触发反相/crop

用法:
  python tools/clean_v2.py --csv 5script/train_fame.csv \
      --img-root final_imgs_256 --out-root clean_out_B --scheme B \
      --report report_B.csv [--workers 32] [--limit 0]
"""
import os
import sys
import csv
import json
import argparse
import numpy as np
from multiprocessing import Pool

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SMALL_FRAC = 0.0005   # 小CC阈值 (占全图)
EDGE_PX = 8           # 边缘环带
BBOX_MARGIN = 12      # std bbox 膨胀(pixel), 容忍真迹与标准字形的位置/大小差异
DIST_THR = 0.35       # 方案 C: 非主CC到主CC归一化距离阈值(相对图对角线)

_STD_BBOX = None
_CHAR2CP = None


def load_std_bbox(path="_sync_work/std_glyph_bbox.json"):
    global _STD_BBOX
    if _STD_BBOX is None:
        try:
            with open(path, encoding="utf-8") as f:
                _STD_BBOX = json.load(f)
        except Exception:
            _STD_BBOX = {}
    return _STD_BBOX


def cc_stats(ink):
    from scipy import ndimage
    lab, nlab = ndimage.label(ink)
    if nlab == 0:
        return None, 0, None, 0
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    main_label = int(sizes.argmax())
    return lab, nlab, sizes, main_label


def clean_one(task):
    src, dst, scheme, char_id = task
    from PIL import Image
    try:
        img = Image.open(src)
        a0 = np.asarray(img.convert("L"), dtype=np.uint8).copy()
    except Exception as e:
        return None

    H, W = a0.shape
    N = float(H * W)
    a = a0.copy()
    orig_ink = a < 128
    orig_n_ink = int(orig_ink.sum())
    if orig_n_ink == 0:
        return None

    lab0, nlab0, sizes0, main0 = cc_stats(orig_ink)
    orig_main_area = int(sizes0[main0]) if sizes0 is not None else orig_n_ink

    inverted = 0
    cropped = 0

    # ---- 前置 1: 反相检测 (白字黑底 -> 黑字白底) ----
    if (orig_n_ink / N) > 0.5 and nlab0 <= 3:
        a = 255 - a
        inverted = 1
        ink = a < 128
        lab, nlab, sizes, main_label = cc_stats(ink)
    else:
        ink = orig_ink
        lab, nlab, sizes, main_label = lab0, nlab0, sizes0, main0
    if lab is None:
        return None

    # ---- 前置 2: 黑边 crop (border_bar) ----
    b = EDGE_PX
    bar = 0
    for strip in (ink[:b, :], ink[-b:, :], ink[:, :b], ink[:, -b:]):
        if strip.size and (strip.sum() / float(strip.size)) > 0.90:
            bar = 1
            break
    if bar:
        # 找最大非边界区域: 用连通域, 排除接触边界且面积大的
        # 简化: 逐边裁掉连续全黑的行/列
        def is_dark_line(mask, axis, idx):
            return (mask[idx, :].mean() if axis == 0 else mask[:, idx].mean()) > 0.90
        top = 0
        while top < H // 4 and is_dark_line(ink, 0, top):
            top += 1
        bot = H - 1
        while bot > 3 * H // 4 and is_dark_line(ink, 0, bot):
            bot -= 1
        left = 0
        while left < W // 4 and is_dark_line(ink, 1, left):
            left += 1
        right = W - 1
        while right > 3 * W // 4 and is_dark_line(ink, 1, right):
            right -= 1
        if top > 0 or bot < H - 1 or left > 0 or right < W - 1:
            a[:(top), :] = 255
            a[(bot + 1):, :] = 255
            a[:, :(left)] = 255
            a[:, (right + 1):] = 255
            ink = a < 128
            lab, nlab, sizes, main_label = cc_stats(ink)
            cropped = 1
            if lab is None:
                return None

    foreign = ink & (lab != main_label)
    if not foreign.any():
        # 无需清理（也可能 clean 后无 foreign）
        pass

    # ---- 方案化清理 ----
    std_bbox = load_std_bbox()
    cp = None
    if scheme == "B" and char_id is not None:
        bbox = std_bbox.get(str(char_id))
        # char_id 是训练 csv 的 character_id, 不是 codepoint!
        # 需要 char_id -> char -> codepoint; 这里由调用方传 codepoint 更可靠,
        # 回退: 若 std_bbox 的 key 不是 codepoint 形式则跳过 bbox 判定
        if bbox is not None:
            cp = bbox

    remove = np.zeros_like(ink)
    if scheme == "B":
        # B (改进版): 
        #   - bbox **外**(膨胀 margin) 的非主墨 -> 删 (明确脏污/边界污染)
        #   - bbox **内** 的非主墨 且 面积很小 -> 也删 (字内噪点)
        #   - bbox **内** 的中等/大面积非主墨 -> 保留 (正常飞白/连笔/分离笔画)
        if cp is not None:
            y0, y1, x0, x1, w, h = cp
            pad = BBOX_MARGIN
            by0 = max(0, y0 - pad); by1 = min(H - 1, y1 + pad)
            bx0 = max(0, x0 - pad); bx1 = min(W - 1, x1 + pad)
            in_box = np.zeros_like(ink)
            in_box[by0:by1 + 1, bx0:bx1 + 1] = True
            # bbox 外: 全删
            remove |= (foreign & ~in_box)
            # bbox 内: 仅删面积很小的 (字内噪点), 保留中等以上(飞白/连笔)
            small_thresh = SMALL_FRAC * N
            for lb in range(1, len(sizes)):
                if lb == main_label:
                    continue
                m = (lab == lb)
                if int(sizes[lb]) < small_thresh and (m & in_box).any():
                    remove |= m
        else:
            # 无参考 bbox -> 退回 A
            remove |= _scheme_a_remove(foreign, lab, sizes, main_label, N, ink)
    elif scheme == "C":
        remove |= _scheme_a_remove(foreign, lab, sizes, main_label, N, ink)
        # 额外: 非主CC到主CC距离远 -> 删
        ys, xs = np.where(lab == main_label)
        if len(ys):
            cy, cx = ys.mean(), xs.mean()
            diag = float(np.hypot(H, W))
            for lb in range(1, nlab + 1):
                if lb == main_label:
                    continue
                m = (lab == lb)
                yy, xx = np.where(m)
                d = float(np.hypot(yy.mean() - cy, xx.mean() - cx)) / diag
                if d > DIST_THR:
                    remove |= m
    else:  # A
        remove |= _scheme_a_remove(foreign, lab, sizes, main_label, N, ink)

    if remove.any():
        a[remove] = 255

    # ---- 评估 ----
    new_ink = a < 128
    nlab_new = 0
    if new_ink.any():
        labn, nlab_new, sizesn, mainn = cc_stats(new_ink)
        new_main_area = int(sizesn[mainn]) if sizesn is not None else 0
        small_left = 0
        for lb in range(1, nlab_new + 1):
            if int(sizesn[lb]) < SMALL_FRAC * N:
                small_left += 1
    else:
        new_main_area = 0
        small_left = 0
        labn, nlab_new, sizesn, mainn = None, 0, None, 0

    clean_ink_n = int(new_ink.sum())
    d = {
        "img_id": os.path.splitext(os.path.basename(src))[0],
        "scheme": scheme,
        "inverted": inverted,
        "cropped": cropped,
        "orig_ink": orig_n_ink,
        "clean_ink": clean_ink_n,
        "ink_change": round((clean_ink_n - orig_n_ink) / max(orig_n_ink, 1), 6),
        "main_keep": round(new_main_area / max(orig_main_area, 1), 6),
        "n_cc_before": int(nlab),
        "n_cc_after": int(nlab_new),
        "small_cc_left": small_left,
    }
    # 只写**实际被修改**的图 (与 final_imgs_256_clean 管线一致, 节省磁盘)
    changed = bool(remove.any()) or bool(inverted) or bool(cropped)
    d["changed"] = int(changed)
    if changed:
        if img.mode == "L":
            Image.fromarray(a, "L").save(dst)
        else:
            rgb = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
            g = (a < 128)
            rgb[g] = 0
            rgb[~g] = 255
            Image.fromarray(rgb, "RGB").save(dst)
    return d


def _scheme_a_remove(foreign, lab, sizes, main_label, N, ink):
    """方案 A: 非主CC面积<阈值 -> 删; 边缘环带内非主墨 -> 删。"""
    remove = np.zeros_like(ink)
    small_thresh = SMALL_FRAC * N
    for lb in range(1, len(sizes)):
        if lb == main_label:
            continue
        if int(sizes[lb]) < small_thresh:
            remove |= (lab == lb)
    b = EDGE_PX
    H, W = ink.shape
    bm = np.zeros_like(ink)
    bm[:b, :] = True; bm[-b:, :] = True; bm[:, :b] = True; bm[:, -b:] = True
    remove |= (foreign & bm)
    return remove


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="5script/train_fame.csv")
    ap.add_argument("--img-root", default="final_imgs_256")
    ap.add_argument("--out-root", default="clean_out")
    ap.add_argument("--scheme", default="A", choices=["A", "B", "C"])
    ap.add_argument("--report", default="clean_report.csv")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    tasks = []
    for r in rows:
        p = r["image_path"]
        if not os.path.isabs(p):
            p = os.path.join(args.img_root, os.path.basename(p)) \
                if not os.path.isfile(p) else p
        if not os.path.isfile(p):
            continue
        # char_id 传过去用于查 std bbox (key 是 codepoint str, 需映射)
        # 这里先传 character 的 codepoint (若可得)
        ch = r.get("character", "")
        cp = ord(ch[0]) if ch and len(ch) == 1 else None
        dst = os.path.join(args.out_root, os.path.basename(p))
        tasks.append((p, dst, args.scheme, cp))
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"[clean_v2] scheme={args.scheme} {len(tasks)} images -> {args.out_root}", flush=True)

    FIELDS = ["img_id", "scheme", "inverted", "cropped", "changed",
              "orig_ink", "clean_ink",
              "ink_change", "main_keep", "n_cc_before", "n_cc_after", "small_cc_left"]
    out = []
    with Pool(args.workers) as pool:
        for i, d in enumerate(pool.imap_unordered(clean_one, tasks, chunksize=32)):
            if d is not None:
                out.append(d)
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(tasks)}", flush=True)

    with open(args.report, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for d in out:
            w.writerow(d)
    print(f"[done] {len(out)} rows -> {args.report}")

    if out:
        arr = {k: np.array([d[k] for d in out], dtype=float) for k in FIELDS
               if k not in ("img_id", "scheme")}
        print("\n=== 汇总 ===")
        for k in ["inverted", "cropped", "changed", "ink_change", "main_keep",
                  "n_cc_before", "n_cc_after", "small_cc_left"]:
            v = arr[k]
            print(f"  {k:<16} mean={v.mean():.5f} median={np.median(v):.5f} "
                  f"p95={np.percentile(v,95):.5f} max={v.max():.5f}")
        ch = arr["changed"]
        print(f"\n  >>> 实际修复(changed=1): {int(ch.sum())} / {len(out)} "
              f"({ch.mean()*100:.2f}%)")
        mk = arr["main_keep"]
        print(f"  main_keep<1 (笔画被删): {(mk<0.999).sum()} ({(mk<0.999).mean()*100:.2f}%)")
        ic = np.abs(arr["ink_change"])
        print(f"  |ink_change|>0.05 (墨量变化大): {(ic>0.05).sum()} ({(ic>0.05).mean()*100:.2f}%)")


if __name__ == "__main__":
    main()
