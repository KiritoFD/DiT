"""去掉沈周 + 核验最终 CSV 可用性。"""
import csv, collections, os, json
ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)

SRC = "assets/train_hcsu_kxl.csv"
DST = "assets/train_hcsu_kxl_final.csv"
EXCLUDE = {"沈周"}

rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
print(f"原始 {len(rows)} 行")
keep = [r for r in rows if r["calligrapher"] not in EXCLUDE]
dropped = len(rows) - len(keep)
print(f"剔除 {EXCLUDE}: {dropped} 行 -> 保留 {len(keep)}")

# 核验: 每条记录的 image_path / std_path 都存在
miss_i = sum(1 for r in keep if not os.path.exists(r["image_path"]))
miss_s = sum(1 for r in keep if not os.path.exists(r["std_path"]))
print(f"缺失 image_path: {miss_i}   std_path: {miss_s}")
if miss_i or miss_s:
    keep = [r for r in keep if os.path.exists(r["image_path"]) and os.path.exists(r["std_path"])]
    print(f"  -> 过滤后 {len(keep)}")

# 核验: 图与 std 都是 256x256 二值
from PIL import Image
import numpy as np
import random
random.seed(5)
smp = random.sample(keep, 400)
bad_sz, bad_bin = 0, 0
for r in smp:
    for k in ("image_path", "std_path"):
        a = np.asarray(Image.open(r[k]).convert("L"))
        if a.shape != (256, 256):
            bad_sz += 1
        if len(np.unique(a)) > 2:
            bad_bin += 1
print(f"抽样 400 条 x2: 尺寸非256x256 {bad_sz}   非二值 {bad_bin}")

# 统计
print(f"\n书家 {len(set(r['calligrapher'] for r in keep))}  字 {len(set(r['character'] for r in keep))}")
print("书体:", dict(collections.Counter(r["script"] for r in keep)))
print("来源:", dict(collections.Counter(r["source"] for r in keep)))

# 与现有数据合并视角
old = list(csv.DictReader(open("assets/train_fame-kxl-tj-px60.csv", encoding="utf-8")))
oc = {r["calligrapher"] for r in old}
print(f"\n现有 {len(old)} 行 / {len(oc)} 书家")
print(f"HCSU 新书家(合并后新增) {sorted({r['calligrapher'] for r in keep} - oc)}")
print(f"合并总计 {len(old)+len(keep)} 行 / "
      f"{len(oc | {r['calligrapher'] for r in keep})} 书家")

with open(DST, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(keep[0].keys()))
    w.writeheader(); w.writerows(keep)
print(f"\n写出 {DST}")
print("表头:", list(keep[0].keys()))
print("样例:", keep[0])
