import csv
import collections

p = "/root/Workspace/xy/DiT/assets/results/v66_tables_condroute2456/20261006-024102-v66-tables-condroute2456-adaLN/eval_stdskel_batch.csv"

rows = list(csv.DictReader(open(p, encoding="utf-8")))
steps = collections.defaultdict(lambda: collections.defaultdict(list))

for r in rows:
    st = int(r["step"])
    sname = r["set"]
    try:
        ssim = float(r["ssim"])
        lpips = float(r["lpips"])
        skel = float(r["skel_iou"])
        mse = float(r["mse"])
        steps[st][sname].append((ssim, lpips, skel, mse))
    except Exception:
        pass

print("=" * 80)
print(f"【v66 实验阶段评测全生命周期指标 (eval200_fixed, N=187)】")
print("=" * 80)
print(f"{'Step':<8} | {'Strict SSIM':<12} | {'Strict LPIPS':<12} | {'Seen SSIM':<12} | {'Skel IoU':<10} | {'MSE':<8}")
print("-" * 80)

for st in sorted(steps.keys()):
    d = steps[st]
    strict = d.get("eval200fix", [])
    seen = d.get("seen", [])
    if not strict:
        continue
    s_ssim = sum(x[0] for x in strict) / len(strict)
    s_lpips = sum(x[1] for x in strict) / len(strict)
    s_skel = sum(x[2] for x in strict) / len(strict)
    s_mse = sum(x[3] for x in strict) / len(strict)
    seen_ssim = sum(x[0] for x in seen) / len(seen) if seen else 0.0
    print(f"{st:<8d} | {s_ssim:<12.4f} | {s_lpips:<12.4f} | {seen_ssim:<12.4f} | {s_skel:<10.4f} | {s_mse:<8.4f}")
print("=" * 80)
