"""从 eval_auto_*.json + 日志重建训练曲线 (绕开 P0-1 被孤儿备份吞掉的 summary)。

为什么需要: in_mem_eval 的 summary 写入曾有 bug (已修, 见 _sync_work/fix_summary_patch.py),
历史 run 的曲线散在 `eval_stdskel_summary.csv.bak_oldcols*` 和 `eval_auto_<step>.json` 里。
本脚本把三者合起来输出**完整曲线** (CSV + 终端表格), 不依赖任何外部库。

用法:
  python tools/plot_curve.py --run-dir exp-std/runs_AB/20261003-212654-v50-A-... 
  python tools/plot_curve.py --run-dir <dir> --out curve.csv
"""
import argparse
import csv
import glob
import json
import os
import sys

KEYS = ["ssim", "mse", "lpips", "ink_ssim", "ink_iou", "skel_iou", "frag",
        "hole", "nn_ssim", "nn_mean", "tgt_spec"]


def load_auto(run_dir):
    """eval_auto_<step>.json -> {step: dict}"""
    out = {}
    pats = [os.path.join(run_dir, "checkpoints", "eval_auto_*.json"),
            os.path.join(run_dir, "eval_auto_*.json"),
            os.path.join(run_dir, "**", "eval_auto_*.json")]
    seen = set()
    for p in pats:
        for f in glob.glob(p, recursive=True):
            if f in seen:
                continue
            seen.add(f)
            try:
                st = int(os.path.basename(f).split("_")[-1].split(".")[0])
            except ValueError:
                continue
            try:
                out[st] = json.load(open(f, encoding="utf-8"))
            except Exception:                                     # noqa: BLE001
                pass
    return out


def load_bak(run_dir):
    """散在 .bak_oldcols* 里的 summary 行 -> {step: dict} (按表头对齐)。"""
    out = {}
    for f in sorted(glob.glob(os.path.join(run_dir, "**", "*bak_oldcols*"), recursive=True)
                    + glob.glob(os.path.join(run_dir, "*bak_oldcols*"))):
        try:
            with open(f, encoding="utf-8") as fh:
                rd = csv.DictReader(fh)
                for r in rd:
                    try:
                        st = int(r.get("step", ""))
                    except ValueError:
                        continue
                    out.setdefault(st, r)
        except Exception:                                         # noqa: BLE001
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = a.run_dir
    if not os.path.isdir(d):
        raise SystemExit(f"没有目录: {d}")

    auto = load_auto(d)
    bak = load_bak(d)
    steps = sorted(set(auto) | set(bak))
    if not steps:
        raise SystemExit(f"{d} 里既没有 eval_auto_*.json 也没有 bak_oldcols 行")

    rows = []
    for st in steps:
        r = {"step": st}
        if st in auto:
            src = auto[st]
            for k in KEYS:
                if k in src:
                    r[k] = src[k]
            r["_src"] = "auto"
        if st in bak:
            for k in ("ssim_mean", "lpips_mean", "ink_iou_mean", "skel_iou_mean",
                      "frag_ratio", "ssim_med"):
                if k in bak[st] and k not in r:
                    r[k] = bak[st][k]
            r["_src"] = (r.get("_src", "") + "+bak").strip("+")
        rows.append(r)

    print(f"\n曲线来源: eval_auto {len(auto)} 个点, bak_oldcols {len(bak)} 个点 -> 共 {len(rows)} 点")
    print(f"{'step':>7} {'ssim':>8} {'ssim_med':>9} {'lpips':>8} {'ink_ssim':>9} "
          f"{'ink_iou':>8} {'frag':>7} {'nn':>7} {'tgt_spec':>9}")
    print("-" * 90)

    def g(r, *names):
        for n in names:
            if n in r and r[n] not in ("", None):
                try:
                    return float(r[n])
                except (TypeError, ValueError):
                    continue
        return None

    for r in rows:
        f = lambda *n: (lambda v: "-" if v is None else f"{v:.4f}")(g(r, *n))  # noqa: E731
        print(f"{r['step']:>7} {f('ssim'):>8} {f('ssim_med'):>9} {f('lpips'):>8} "
              f"{f('ink_ssim', 'ink_ssim_mean'):>9} {f('ink_iou', 'ink_iou_mean'):>8} "
              f"{f('frag', 'frag_ratio'):>7} {f('nn_ssim'):>7} {f('tgt_spec'):>9}")

    if a.out:
        cols = ["step"] + KEYS + ["ssim_mean", "lpips_mean", "ink_iou_mean",
                                  "skel_iou_mean", "frag_ratio", "ssim_med", "_src"]
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\n[out] {a.out}  ({len(rows)} 行)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
