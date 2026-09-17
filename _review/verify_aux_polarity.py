"""验证新生成的 aux 与旧 aux 对**同一张原图**是否逐像素一致（极性对不对）。

50k 的 csv 有 `src_image_path` 指回旧图，所以能一一对应上。
"""
import csv
import os
import re
import sys

import numpy as np
from PIL import Image

os.chdir("/root/Workspace/xy/DiT")
sys.path.insert(0, "/root/Workspace/xy/DiT")
from tools.gen_aux_50k import gen_one, SKEL_OUT, CANNY_OUT  # noqa: E402

os.makedirs(SKEL_OUT, exist_ok=True)
os.makedirs(CANNY_OUT, exist_ok=True)


def old_iid(p):
    m = re.search(r"(\d+)\.png$", str(p))
    return int(m.group(1)) if m else None


rows = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))
# 找 src_image_path 指向 fame-kxl-tj-px60 的（那些才有旧 aux 可比）
cand = [r for r in rows if "fame-kxl-tj-px60" in r.get("src_image_path", "")]
print(f"  有旧图对应关系的行: {len(cand)}")

n_ok = n_bad = n_missing = 0
for r in cand[:20]:
    iid = old_iid(r["image_path"])
    oiid = old_iid(r["src_image_path"])
    new_id, ok = gen_one((iid, r["image_path"]))
    for tag, new_dir, old_dir in (
            ("skel3", SKEL_OUT, "data/skel/final_skel3_base"),
            ("canny", CANNY_OUT, "data/aux/final_canny_base")):
        old_p = f"{old_dir}/{oiid}.png"
        new_p = f"{new_dir}/{iid:06d}.png"
        if not os.path.exists(old_p):
            n_missing += 1
            continue
        a = np.asarray(Image.open(old_p).convert("L"), dtype=np.int16)
        # canny 旧 PNG 是黑底白线（doc58 修复前遗留），新的是白底 -> 比之前先反转
        if tag == "canny":
            a = 255 - a
        b = np.asarray(Image.open(new_p).convert("L"), dtype=np.int16)
        d = np.abs(a - b)
        same = (d > 0).mean()
        if same < 0.02:
            n_ok += 1
        else:
            n_bad += 1
            if n_bad <= 4:
                print(f"  ✗ {tag} {oiid}->{iid}: 差异像素占比 {same:.3f}  "
                      f"旧背景众数={int(np.bincount(a.ravel().clip(0,255)).argmax())} "
                      f"新背景众数={int(np.bincount(b.ravel().clip(0,255)).argmax())}")

print()
print(f"  一致(<2% 差异): {n_ok}   不一致: {n_bad}   旧文件缺: {n_missing}")
print("  (canny 已把旧 PNG 反转后再比 —— 因为新规范是白底，旧的是黑底)")
print("  -> " + ("**极性与旧 aux 等价** ✓" if n_bad == 0
                 else "**有差异，需查** ✗"))
