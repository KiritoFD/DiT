"""给 batch_eval.py 的 CSV header 加 3 个墨迹指标列。"""
import io
import os
import re

os.chdir("/root/Workspace/xy/DiT")
p = "src/eval/batch_eval.py"
s = io.open(p, encoding="utf-8").read()
orig = s

# summary header
s = s.replace(
    '"ssim_med", "ssim_q3", "ssim_p90", "mse_mean", "lpips_mean"])',
    '"ssim_med", "ssim_q3", "ssim_p90", "mse_mean", "lpips_mean",\n'
    '                        "ink_ssim_mean", "ink_iou_mean", "skel_iou_mean"])')
# raw header
s = s.replace(
    '"mse", "ssim", "lpips"])',
    '"mse", "ssim", "lpips",\n'
    '                        "ink_ssim", "ink_iou", "skel_iou"])')

if s == orig:
    print("  ⚠ 没有任何替换生效，打印实际代码：")
    for i, ln in enumerate(io.open(p, encoding="utf-8").read().splitlines()):
        if "ssim_med" in ln or "lpips_mean" in ln or '"mse", "ssim"' in ln:
            print(f"    {i+1}: {ln!r}")
else:
    io.open(p, "w", encoding="utf-8").write(s)
    print("  ✓ header 已更新")
    import ast
    ast.parse(s)
    print("  SYNTAX OK")

# 复验
s2 = io.open(p, encoding="utf-8").read()
for k in ("ink_ssim_mean", "ink_ssim\""):
    print(f"  {k!r} 出现 {s2.count(k)} 次")
