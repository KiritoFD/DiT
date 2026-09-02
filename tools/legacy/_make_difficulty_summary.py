# -*- coding: utf-8 -*-
"""
_make_difficulty_summary.py — 汇总「数据难度 vs 生成指标」的关系。

核心问题：SSIM 跨实验可比吗？
把 47 个实验按训练集书家数分组，看 SSIM 是否随难度单调下降。
若成立，则历史上"0.73 → 0.47"不是退步，而是评测集变难，
此前的跨实验对比（以及基于它的结论）都需要重新审视。

产出 5script/difficulty_summary.csv（本地处理 experiments_enriched.csv）
"""
import csv, collections, statistics as st


def main():
    rows = list(csv.DictReader(open("5script/experiments_enriched.csv",
                                    encoding="utf-8")))
    groups = collections.defaultdict(list)
    for r in rows:
        try:
            cal = int(r["data_cal"])
        except (TypeError, ValueError):
            continue
        try:
            ssim = float(r["best_ssim"])
        except (TypeError, ValueError):
            continue
        if ssim <= 0.1:          # 过滤掉明显训崩/早期中断的
            continue
        groups[cal].append({
            "series": r["series"], "ssim": ssim,
            "step": r["best_step"], "task": r["task"],
            "data_char": r["data_char"], "data_n": r["data_n"],
            "diffusion": r["diffusion_cfg"],
        })

    out = []
    for cal in sorted(groups):
        g = groups[cal]
        ss = [x["ssim"] for x in g]
        out.append({
            "n_calligraphers": cal,
            "n_experiments": len(g),
            "ssim_max": round(max(ss), 4),
            "ssim_mean": round(st.mean(ss), 4),
            "ssim_median": round(st.median(ss), 4),
            "ssim_min": round(min(ss), 4),
            "data_char": g[0]["data_char"],
            "data_n": g[0]["data_n"],
            "best_series": max(g, key=lambda x: x["ssim"])["series"],
            "all_series": "; ".join(sorted({x["series"] for x in g})),
        })

    cols = ["n_calligraphers", "n_experiments", "ssim_max", "ssim_mean",
            "ssim_median", "ssim_min", "data_char", "data_n",
            "best_series", "all_series"]
    with open("5script/difficulty_summary.csv", "w", encoding="utf-8",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)

    print(f"# wrote 5script/difficulty_summary.csv ({len(out)} groups)\n")
    print(f"  {'书家数':>7}{'实验数':>6}{'SSIM_max':>10}{'SSIM_mean':>11}"
          f"{'SSIM_med':>10}{'字数':>8}   最佳系列")
    for r in out:
        print(f"  {r['n_calligraphers']:>7}{r['n_experiments']:>6}"
              f"{r['ssim_max']:>10.4f}{r['ssim_mean']:>11.4f}"
              f"{r['ssim_median']:>10.4f}{str(r['data_char']):>8}   "
              f"{r['best_series']}")

    # 单调性检验
    print("\n# 单调性：按书家数升序，看 ssim_max 是否递减")
    prev = None
    for r in out:
        arrow = "" if prev is None else ("↓" if r["ssim_max"] < prev else "↑")
        print(f"    cal={r['n_calligraphers']:<5} ssim_max={r['ssim_max']:.4f} {arrow}")
        prev = r["ssim_max"]


if __name__ == "__main__":
    main()
