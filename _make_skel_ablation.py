# -*- coding: utf-8 -*-
"""
_make_skel_ablation.py — 骨架条件消融对照：1px vs 3px vs std-skel。

直接从 daemon 日志提取 base/ctrl 双臂指标，按 step 对齐，产出
5script/skel_ablation.csv。

科学问题
--------
3px 骨架是 GT 图经 skeletonize 后 8 邻域膨胀 ×3 得到的。它在 256px 图上
占据可观面积，其 VAE latent 与真实书法图相当接近 —— 因此怀疑
fame-ctrl 的 0.8045 中有一部分来自「条件本身泄露」（给的条件已经很像目标图），
而非真正的结构控制能力。

1px（细线）是更干净的纯结构条件。判读：
  - 1px ≈ 3px（甚至更好）→ 泄露不是主因，模型确实在学结构
  - 1px 明显更差        → 3px 高分确实含泄露成分，需要重新审视该指标
"""
import os, sys, csv, re, glob, collections
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ANSI = re.compile(r"\x1b\[[0-9;]*m")
# [ctrl-metrics] step 2500: base  MSE=0.91634 SSIM=0.4977 SkelIoU=0.0115
P = re.compile(r"\[ctrl-metrics\] step (\d+): (base|ctrl)\s+"
               r"MSE=([\d.]+) SSIM=([\d.]+) SkelIoU=([\d.]+)")
# [ctrl-metrics] step 2500: LPIPS=... (若存在)
PL = re.compile(r"\[ctrl-metrics\] step (\d+): (base|ctrl)\s+.*LPIPS=([\d.]+)")

# 实验 -> (标签, 骨架类型, 日志)
RUNS = {
    "3px (fame-ctrl)": ("3px", ["/tmp/ctrl_daemon.log"]),
    "1px (fame-ctrl-1pix)": ("1px", ["/tmp/1pix_daemon.log"]),
    "std-skel (fame-ctrl-std)": ("std-skel", ["/tmp/std_daemon.log"]),
}


def parse(paths):
    """log -> {(step, arm): metrics}"""
    out = {}
    for p in paths:
        if not os.path.isfile(p):
            continue
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                s = ANSI.sub("", line)
                m = P.search(s)
                if not m:
                    continue
                step, arm = int(m.group(1)), m.group(2)
                d = out.setdefault((step, arm), {})
                d["mse"] = float(m.group(3))
                d["ssim"] = float(m.group(4))
                d["skel_iou"] = float(m.group(5))
                ml = PL.search(s)
                if ml:
                    d["lpips"] = float(ml.group(3))
    return out


def main():
    data = {}
    for label, (_, logs) in RUNS.items():
        d = parse(logs)
        if d:
            data[label] = d
            print(f"# {label}: {len(d)//2} steps from {logs}")

    if not data:
        print("# no ctrl-metrics logs found")
        return

    steps = sorted({s for d in data.values() for (s, a) in d})
    rows = []
    for st in steps:
        row = {"step": st}
        for label, d in data.items():
            b, c = d.get((st, "base")), d.get((st, "ctrl"))
            if not c:
                continue
            tag = RUNS[label][0]
            row[f"{tag}_ctrl_ssim"] = c.get("ssim")
            row[f"{tag}_ctrl_iou"] = c.get("skel_iou")
            row[f"{tag}_ctrl_mse"] = c.get("mse")
            if b:
                row[f"{tag}_base_ssim"] = b.get("ssim")
                row[f"{tag}_delta_ssim"] = round(c["ssim"] - b["ssim"], 4)
                row[f"{tag}_delta_iou"] = round(
                    c["skel_iou"] - b["skel_iou"], 4) if "skel_iou" in b else None
        rows.append(row)

    cols = ["step"]
    for tag in ("3px", "1px", "std-skel"):
        cols += [f"{tag}_base_ssim", f"{tag}_ctrl_ssim", f"{tag}_delta_ssim",
                 f"{tag}_ctrl_iou", f"{tag}_delta_iou", f"{tag}_ctrl_mse"]
    cols = [c for c in cols if any(c in r for r in rows)]

    os.makedirs("5script", exist_ok=True)
    with open("5script/skel_ablation.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n# wrote 5script/skel_ablation.csv ({len(rows)} steps)")

    print("\n# 1px vs 3px 对照（ctrl 臂）:")
    print(f"  {'step':>7}{'1px_ssim':>10}{'3px_ssim':>10}{'Δ(1px-3px)':>12}"
          f"{'1px_iou':>9}{'3px_iou':>9}")
    for r in rows:
        a, b = r.get("1px_ctrl_ssim"), r.get("3px_ctrl_ssim")
        ia, ib = r.get("1px_ctrl_iou"), r.get("3px_ctrl_iou")
        if a is None or b is None:
            continue
        print(f"  {r['step']:>7}{a:>10.4f}{b:>10.4f}{a-b:>+12.4f}"
              f"{(ia or 0):>9.4f}{(ib or 0):>9.4f}")


if __name__ == "__main__":
    main()
