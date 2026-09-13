#!/usr/bin/env python
"""collect_follow_matrix.py — 汇总 char-null 协议 (开集新字正确评估) 的 metrics.json"""
import json, glob, sys, csv
import statistics

paths = sorted(glob.glob("/root/Workspace/xy/DiT/assets/results/skel_follow_gpu/*_charnull/metrics.json"))
rows = []
for p in paths:
    tag = p.split("/")[-2]
    rows.append((tag, json.load(open(p))))

if not rows:
    print("NO charnull RESULTS FOUND"); sys.exit(1)

def iou3list(d): return [r["iou3"] for r in d.get("rows", [])]
def iou1list(d): return [r["iou"] for r in d.get("rows", [])]

print("### char-null 协议 (开集新字正确评估) — n=30, 字库骨架 latent 条件, callig=1 ###")
print(f"{'tag':<24}{'n':>4}{'iou3_mean':>10}{'iou3_med':>10}{'iou3_std':>10}"
      f"{'iou1_mean':>10}{'iou1_med':>10}")
out_rows = []
for tag, d in rows:
    x3 = iou3list(d); x1 = iou1list(d)
    m3 = d.get("iou3_mean", statistics.mean(x3)) if x3 else float("nan")
    md3 = d.get("iou3_median", statistics.median(x3)) if x3 else float("nan")
    sd3 = statistics.stdev(x3) if len(x3) > 1 else 0.0
    m1 = d.get("iou_mean", statistics.mean(x1)) if x1 else float("nan")
    md1 = statistics.median(x1) if x1 else float("nan")
    n = d.get("n", len(x3))
    print(f"{tag:<24}{n:>4}{m3:>10.3f}{md3:>10.3f}{sd3:>10.3f}{m1:>10.3f}{md1:>10.3f}")
    out_rows.append((tag, n, round(m3,4), round(md3,4), round(sd3,4), round(m1,4), round(md1,4)))

out = "/root/Workspace/xy/DiT/assets/results/skel_follow_gpu/summary_charnull.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tag", "n", "iou3_mean", "iou3_median", "iou3_std", "iou1_mean", "iou1_median"])
    w.writerows(out_rows)
print("\nCSV ->", out)