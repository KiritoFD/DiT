#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import sys

sys.stdout.reconfigure(encoding="utf-8")

df = pd.read_csv("assets/train_50k_v2_fixed.csv")
cs_counts = df.groupby(["calligrapher", "script"]).size().reset_index(name="count")

print(f"总计 (书家 x 书体) 对数量: {len(cs_counts)}")
print(f"平均每个书家拥有书体数: {len(cs_counts) / 45:.2f} 种")

# 统计每个书家各书体的倾斜度
print("\n样本极度稀疏的 (书家 x 书体) 对 (样本数 < 30):")
sparse = cs_counts[cs_counts["count"] < 30].sort_values(by="count")
print(sparse.to_string(index=False))

print("\n样本数在 30 ~ 100 的对:")
semi_sparse = cs_counts[(cs_counts["count"] >= 30) & (cs_counts["count"] < 100)].sort_values(by="count")
print(semi_sparse.to_string(index=False))

print("\n每个书家的主次书体分布样例 (Top 10 书家):")
for c in list(df["calligrapher"].value_counts().index[:10]):
    sub = cs_counts[cs_counts["calligrapher"] == c]
    d_str = ", ".join([f"{r['script']}:{r['count']}" for _, r in sub.iterrows()])
    print(f"  - {c:<8}: {d_str}")
