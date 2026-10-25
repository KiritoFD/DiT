"""eval_complexity_stratification.py — 复杂度分层测试 (20x256 vs 20x384 at matched exposure)."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt, label
from skimage.morphology import skeletonize


def ssim_slice(pred, gt):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(pred, gt, channel_axis=2 if pred.ndim==3 else None, data_range=1.0))


def analyze_image_pair(pred_path, gt_path):
    p_img = np.array(Image.open(pred_path).convert('L')) / 255.0
    g_img = np.array(Image.open(gt_path).convert('L')) / 255.0

    p_bin = p_img < 0.5
    g_bin = g_img < 0.5

    gt_mass = float(g_bin.mean())
    gl, gn = label(g_bin, np.ones((3, 3)))
    pl, pn = label(p_bin, np.ones((3, 3)))

    # Missing components
    missing = sum(int((gl == i).sum()) >= 5 and float(p_bin[gl == i].mean()) < 0.25 for i in range(1, gn + 1))

    # Merged components (错误粘连)
    merged = 0
    for i in range(1, pn + 1):
        ids, counts = np.unique(gl[pl == i], return_counts=True)
        merged += int(((ids > 0) & (counts >= 5)).sum() > 1)

    # Skeleton recall (内容准确率 / 笔画拓扑召回)
    skel = skeletonize(g_bin)
    thin = skel & (distance_transform_edt(g_bin) <= 2.5)
    thin_recall = float(binary_dilation(p_bin)[thin].mean()) if thin.any() else 1.0

    # SSIM & L1
    ssim_val = ssim_slice(p_img, g_img)
    l1_val = float(np.abs(p_img - g_img).mean())

    # Ink IoU
    inter = (p_bin & g_bin).sum()
    union = (p_bin | g_bin).sum()
    ink_iou = float(inter / max(union, 1))

    return {
        "gt_mass": gt_mass,
        "gt_components": int(gn),
        "thin_skeleton_recall": thin_recall,
        "missing_components": int(missing),
        "merged_components": int(merged),
        "ssim": ssim_val,
        "l1": l1_val,
        "ink_iou": ink_iou
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True, help="label=dir")
    p.add_argument("--n", type=int, default=187)
    p.add_argument("--out", default="assets/capacity_audit_20261017/complexity_stratification.json")
    args = p.parse_args()

    models = {}
    for item in args.model:
        lbl, d = item.split("=", 1)
        models[lbl] = Path(d)

    # Analyze first model's GT to establish complexity grouping
    first_dir = next(iter(models.values()))
    gt_stats = []
    for i in range(args.n):
        gt_path = first_dir / f"gt{i}.png"
        g_img = np.array(Image.open(gt_path).convert('L')) / 255.0
        g_bin = g_img < 0.5
        gl, gn = label(g_bin, np.ones((3, 3)))
        gt_stats.append({
            "idx": i,
            "gt_mass": float(g_bin.mean()),
            "gt_components": int(gn)
        })

    # Combined complexity score = rank(mass) + rank(components)
    masses = np.array([x["gt_mass"] for x in gt_stats])
    comps = np.array([x["gt_components"] for x in gt_stats])
    rank_m = np.argsort(np.argsort(masses))
    rank_c = np.argsort(np.argsort(comps))
    score = rank_m + rank_c

    # Divide into 3 tiers: Simple (bottom 33%), Medium (middle 33%), Complex (top 33%)
    t1_bound = np.quantile(score, 0.33)
    t2_bound = np.quantile(score, 0.67)

    tiers = {}
    for i in range(args.n):
        if score[i] <= t1_bound:
            tiers[i] = "Simple (低复杂度)"
        elif score[i] <= t2_bound:
            tiers[i] = "Medium (中复杂度)"
        else:
            tiers[i] = "Complex (高复杂度)"

    tier_counts = {t: sum(1 for v in tiers.values() if v == t) for t in set(tiers.values())}
    print("Tier counts:", tier_counts)

    results = {}
    for lbl, mdir in models.items():
        records = []
        for i in range(args.n):
            pred_path = mdir / f"g{i}.png"
            gt_path = mdir / f"gt{i}.png"
            stat = analyze_image_pair(pred_path, gt_path)
            stat["tier"] = tiers[i]
            records.append(stat)

        by_tier = {}
        for t in ["Simple (低复杂度)", "Medium (中复杂度)", "Complex (高复杂度)"]:
            sub = [r for r in records if r["tier"] == t]
            by_tier[t] = {
                "count": len(sub),
                "ssim": float(np.mean([r["ssim"] for r in sub])),
                "l1": float(np.mean([r["l1"] for r in sub])),
                "ink_iou": float(np.mean([r["ink_iou"] for r in sub])),
                "thin_skeleton_recall": float(np.mean([r["thin_skeleton_recall"] for r in sub])),
                "merged_components": float(np.mean([r["merged_components"] for r in sub])),
                "missing_components": float(np.mean([r["missing_components"] for r in sub])),
            }
        results[lbl] = {
            "overall": {
                "ssim": float(np.mean([r["ssim"] for r in records])),
                "l1": float(np.mean([r["l1"] for r in records])),
                "ink_iou": float(np.mean([r["ink_iou"] for r in records])),
                "thin_skeleton_recall": float(np.mean([r["thin_skeleton_recall"] for r in records])),
                "merged_components": float(np.mean([r["merged_components"] for r in records])),
                "missing_components": float(np.mean([r["missing_components"] for r in records])),
            },
            "by_tier": by_tier
        }

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Stratification results saved to {out_p}")


if __name__ == "__main__":
    main()
