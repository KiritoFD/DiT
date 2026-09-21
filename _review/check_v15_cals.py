"""查 v15 的 87 个书家里，我挑的 few-shot 候选是否已被包含。"""
import csv
import json
import os

import torch

os.chdir("/root/Workspace/xy/DiT")

d = torch.load("assets/multistyle_k4_pretrained.pt", map_location="cpu",
               weights_only=False)
print(f"  embedding {tuple(d['embedding'].shape)}, "
      f"centroids {tuple(d['centroids'].shape)}, "
      f"k_clusters={d.get('k_clusters')}, dim={d.get('dim')}")
print(f"  map_path = {d.get('map_path')}")
print(f"  npz_path = {d.get('npz_path')}")

targets = ["怀素", "伊秉绶", "徐渭", "沈周", "文征明", "文徵明",
           "宋高宗", "傅山", "李阳冰", "张瑞图", "宋徽宗", "成亲王",
           "陈道复", "王福庵"]

# 从 v15 训练数据里找书家名
found = {}
for cfg in ("v15a_multistyle_k4_pool", "v15b_multistyle_k4_ca",
            "v15_multistyle_k4"):
    p = f"src/train/configs/{cfg}.json"
    if not os.path.exists(p):
        continue
    c = json.load(open(p, encoding="utf-8"))
    dc = c.get("data_csv")
    if not dc or not os.path.exists(dc):
        continue
    rows = list(csv.DictReader(open(dc, encoding="utf-8")))
    cals = sorted(set(r["calligrapher"] for r in rows))
    print(f"\n  [{cfg}] data_csv={dc}")
    print(f"    {len(cals)} 个书家")
    print(f"    前几个: {cals[:8]}")
    for t in targets:
        if t in cals:
            found[t] = True
    break

print("\n  === few-shot 候选是否已在 v15 训练数据里 ===")
for t in targets:
    mark = "★已在（不能当'新书家'）" if found.get(t) else "不在（可用）"
    print(f"    {t:<8} {mark}")

# 尝试从 map_path 拿名字
mp = d.get("map_path")
if mp and os.path.exists(str(mp)):
    try:
        m = json.load(open(mp, encoding="utf-8"))
        print(f"\n  map_path 内容键: {list(m)[:6] if isinstance(m, dict) else type(m)}")
    except Exception as e:
        print(f"  读 map_path 失败: {e}")
