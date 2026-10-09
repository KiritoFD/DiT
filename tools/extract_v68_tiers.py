import os
import pandas as pd
import csv

p = "/root/Workspace/xy/DiT/assets/results/v68_aug_sp_c2ot/20261006-192021-v68-aug-sp-c2ot/eval_stdskel_batch.csv"
rows = []
with open(p, "r", encoding="utf-8", errors="ignore") as f:
    reader = csv.DictReader(f)
    for r in reader:
        if r.get("set") == "eval200fix" and r.get("step") == "200000":
            rows.append(r)

print(f"找到 v68 step 200000 在 eval200fix 上的样本数: {len(rows)}")

df = pd.DataFrame(rows)
df["ssim"] = df["ssim"].astype(float)
df["mse"] = df["mse"].astype(float)
df["idx"] = df["idx"].astype(int)

# 按 ssim 降序排序
df_sorted = df.sort_values(by="ssim", ascending=False).reset_index(drop=True)

print("\n=== Top 10 (最高 SSIM) ===")
top10 = df_sorted.head(10)
for i, r in top10.iterrows():
    print(f"  [{i+1}] idx={r['idx']:03d} | char={r['char']} | callig={r['calligrapher']} | script={r['script']} | ssim={r['ssim']:.4f} | mse={r['mse']:.4f}")

# 中位数附近 10 个 (中间点为 187 // 2 = 93，取 89..98)
mid_start = (len(df_sorted) - 10) // 2
mid10 = df_sorted.iloc[mid_start:mid_start+10]
print(f"\n=== Mid 10 (中位数区域，排名 {mid_start+1}..{mid_start+10}) ===")
for i, r in mid10.reset_index(drop=True).iterrows():
    print(f"  [{i+1}] idx={r['idx']:03d} | char={r['char']} | callig={r['calligrapher']} | script={r['script']} | ssim={r['ssim']:.4f} | mse={r['mse']:.4f}")

# 最差 10 个 (Lowest SSIM)
print("\n=== Worst 10 (最低 SSIM) ===")
worst10 = df_sorted.tail(10).iloc[::-1]
for i, r in worst10.reset_index(drop=True).iterrows():
    print(f"  [{i+1}] idx={r['idx']:03d} | char={r['char']} | callig={r['calligrapher']} | script={r['script']} | ssim={r['ssim']:.4f} | mse={r['mse']:.4f}")
