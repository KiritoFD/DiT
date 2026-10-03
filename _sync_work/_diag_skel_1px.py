# -*- coding: utf-8 -*-
"""诊断 1px 重做所需数据: 训练/评估 集 id 覆盖, skel1 PNG 可用性."""
import csv, glob, os, re
import numpy as np

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)

def read_ids(csv_path):
    ids = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = re.search(r"(\d+)\.png", row["image_path"])
            if m:
                ids.append(int(m.group(1)))
    return sorted(set(ids))

def skel1_avail(ids, d="data/skel/final_skel1"):
    ex = set()
    # 抽样检查, 不每次全局 ls
    for i in ids:
        if os.path.exists(os.path.join(d, f"{i}.png")):
            ex.add(i)
    return len(ex), len(ids)

def shard_ids(npat):
    ids = []
    for f in sorted(glob.glob(npat)):
        with np.load(f) as d:
            ids.extend(int(x) for x in d["img_ids"])
    return sorted(set(ids))

# 训练集
train_ids = read_ids("assets/train_mid_common.csv")
print(f"[train_mid_common] id total = {len(train_ids)}")
a, t = skel1_avail(train_ids, "data/skel/final_skel1")
print(f"[train_mid_common] skel1 png available = {a}/{t}")

mid_common_lat_ids = shard_ids("data/skel/final_skel_latents_mid_common/shard_*.npz")
print(f"[mid_common latent] id total = {len(mid_common_lat_ids)}")

# 评估集
eval_ids = read_ids("assets/eval_strict_midclean.csv")
print(f"[eval_strict_midclean] id total = {len(eval_ids)}")
a, t = skel1_avail(eval_ids, "data/skel/final_skel1")
print(f"[eval] skel1 png available = {a}/{t}")

eval_v2_ids = shard_ids("data/skel/final_skel_latents_eval_v2/shard_*.npz")
print(f"[eval_v2 latent] id total = {len(eval_v2_ids)}")
missing_in_v2 = [i for i in eval_ids if i not in set(eval_v2_ids)]
print(f"[eval_v2] missing eval ids = {len(missing_in_v2)}, sample = {missing_in_v2[:10]}")

# 检查 1px 骨架是不是真的细 (对比 3px 的激活像素占比)
from PIL import Image
for iid in eval_ids[:200]:
    p1 = f"data/skel/final_skel1/{iid}.png"
    if os.path.exists(p1):
        a1 = (np.asarray(Image.open(p1).convert("L")) < 127).sum()
        p3 = f"data/skel/final_skel3/{iid}.png"
        a3 = (np.asarray(Image.open(p3).convert("L")) < 127).sum() if os.path.exists(p3) else -1
        print(f"[px] id={iid} skel1_px={a1} skel3_px={a3} ratio={(a3/a1 if a1 else -1):.2f}")
        if iid >= 0 and len([x for x in eval_ids[:200] if x <= iid]) > 8:
            break
print("[diag] DONE")
