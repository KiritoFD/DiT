# -*- coding: utf-8 -*-
"""导出各类污染的代表性样本图，用于人工确认污染类型（避免指标误判）。"""
import os, sys, csv, shutil
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")
import numpy as np

SRC = "data/imgs/final_imgs_256"
OUT = "_sync_work/pollution_samples"
os.makedirs(OUT, exist_ok=True)

rows = list(csv.DictReader(open("assets/scan_train_pollution.csv", encoding="utf-8")))


def col(k):
    return np.array([float(r[k]) for r in rows], dtype=float)


foreign = col("foreign_area_ratio")
mainf = col("main_frac")
edge = col("edge_ink_ratio")
smalla = col("small_cc_area_ratio")
bar = col("border_bar")
inv = col("inverted")
ncc = col("n_cc")


def save(indices, tag, n=8):
    sub = os.path.join(OUT, tag)
    os.makedirs(sub, exist_ok=True)
    for i in indices[:n]:
        r = rows[i]
        iid = r["img_id"]
        # 找原图: train_fame.csv 里的 image_path
        src = os.path.join(SRC, f"{iid}.png")
        if not os.path.isfile(src):
            continue
        dst = os.path.join(sub, f"{iid}_{tag}.png")
        shutil.copy(src, dst)
    print(f"  {tag}: saved {min(n, len(indices))} samples")


# 按指标排序取最极端的样本
order_f = np.argsort(-foreign)
order_m = np.argsort(mainf)
order_e = np.argsort(-edge)
order_s = np.argsort(-smalla)
order_n = np.argsort(-ncc)

print("exporting samples ...")
save(list(order_f[:8]), "01_high_foreign")      # 非主连通域最多(疑似脏污或正常多笔画)
save(list(order_m[:8]), "02_low_mainfrac")      # 主连通域占比最低(碎片)
save(list(order_e[:8]), "03_high_edge")         # 边界墨最多(黑框/边缘污染)
save(list(order_s[:8]), "04_high_smallnoise")   # 小噪点最多
save(list(np.where(bar == 1)[0][:8]), "05_border_bar")  # 实心黑条
save(list(np.where(inv == 1)[0][:8]), "06_inverted")    # 反相
save(list(order_n[:8]), "07_many_cc")           # 连通域极多

print(f"\ndone -> {OUT}")
# 同时打印这些样本的指标值
print("\n=== 样本指标 ===")
for tag, idxs in [("high_foreign", order_f[:5]), ("low_mainfrac", order_m[:5]),
                  ("high_edge", order_e[:5]), ("high_smallnoise", order_s[:5]),
                  ("border_bar", np.where(bar == 1)[0][:5]),
                  ("inverted", np.where(inv == 1)[0][:5])]:
    print(f"\n[{tag}]")
    for i in idxs:
        r = rows[i]
        print(f"  {r['img_id']}: ink={r['ink_ratio']} n_cc={r['n_cc']} "
              f"main_frac={r['main_frac']} small_a={r['small_cc_area_ratio']} "
              f"foreign={r['foreign_area_ratio']} edge={r['edge_ink_ratio']} "
              f"bar={r['border_bar']} inv={r['inverted']}")
