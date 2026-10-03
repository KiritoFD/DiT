"""证据链：few-shot 的 img_id (0..149) 与 50k 的 img_id 撞号 -> g(骨架条件) 张冠李戴。

skel_latent_shards_dir 指向 50k 的 shards_std，而 shard 查表**只认 img_id**。
few-shot CSV 的 img_id 是 0..149（该主题内部编号），于是训练/评测拿到的是
"50k 里第 0..149 号样本"的骨架，而不是新书家那个字的骨架。
"""
import csv
import os

os.chdir("/root/Workspace/xy/DiT")

k50 = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))
print("50k 列:", [c for c in k50[0].keys()])
by_id = {}
for i, r in enumerate(k50):
    key = r.get("img_id") or str(i)
    by_id.setdefault(key, (r["character"], r["calligrapher"], os.path.basename(r["std_path"])))

for cal in ("沈周", "伊秉绶"):
    print(f"\n===== {cal} =====")
    for nm in ("train", "eval"):
        rows = list(csv.DictReader(open(f"assets/fs50_{cal}_{nm}.csv", encoding="utf-8")))
        print(f"-- {nm} 前 5 行: img_id | 本行该写的字 | shards_std 里这个 id 实际的字/书家/std")
        for r in rows[:5]:
            iid = r["img_id"]
            got = by_id.get(iid, ("<无>", "?", "?"))
            print(f"   {iid:>4s} | {r['character']}({os.path.basename(r['std_path'])}) "
                  f"| {got[0]} {got[1]} {got[2]}")
        break

# 直接查 std shard 里 id 0..4 的真实来源
import numpy as np
d = np.load("data/50k/shards_std/shard_00000.npz")
print("\nshards_std/shard_00000 img_ids[:8]:", d["img_ids"][:8], "dtype", d["img_ids"].dtype)
print("shards_fs50_沈周 img_ids[:8]:",
      np.load("data/50k/shards_fs50_沈周/shard_00000.npz")["img_ids"][:8])
