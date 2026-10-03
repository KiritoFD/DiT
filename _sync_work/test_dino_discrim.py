"""决定性测试：DINO CLS 特征本身区分不区分「书家×书体」pair？

如果不区分（pair 间余弦高），那 v15 的 K-Means 表和 SupCon 都救不了，
必须先换特征。如果区分，那 v15 表的坏是 K-Means 实现/用法的问题，SupCon 可修。
"""
import csv
import json
import os

import numpy as np

os.chdir("/root/Workspace/xy/DiT")

NPZ = "assets/dino_cls_50k.npz"
MAP = "assets/callig_script_id_map.json"
CSV = "assets/train_50k_v2.csv"

z = np.load(NPZ)
print(f"  npz keys: {list(z.files)[:5]}")
feat = z["feat"] if "feat" in z.files else z[z.files[0]]
ids = z["img_id"] if "img_id" in z.files else None
print(f"  feat {feat.shape} {feat.dtype}  ids={None if ids is None else ids.shape}")

m = json.load(open(MAP, encoding="utf-8"))
pair_map = m["pair_map"]          # "callig:script" -> pair_id
print(f"  pair_map {len(pair_map)} 条")

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
print(f"  csv {len(rows)} 行")

# 建 img_id -> pair_id：用行序对齐（dino 提取按 csv 顺序）
# 先试 npz 里的 ids 能否对上
def norm(s):
    return os.path.basename(str(s))

csv_keys = [norm(r.get("image_path", "")) for r in rows]
npz_keys = [norm(x) for x in ids] if ids is not None else None

pair_of = []
if npz_keys is not None and len(npz_keys) == len(rows):
    # 按顺序假设对齐，但校验一下
    same = sum(1 for a, b in zip(csv_keys, npz_keys) if a == b)
    print(f"  按行序对齐校验: {same}/{len(rows)} 文件名一致")
    use_order = same > len(rows) * 0.5
else:
    use_order = False

if use_order or ids is None:
    for r in rows:
        k = f"{r['calligrapher_id']}:{r['script_id']}"
        pair_of.append(pair_map.get(k, -1))
else:
    idx = {k: i for i, k in enumerate(npz_keys)}
    pair_of = [-1] * len(rows)
    for i, r in enumerate(rows):
        j = idx.get(norm(r.get("image_path", "")), -1)
        if j >= 0:
            k = f"{rows[j]['calligrapher_id']}:{rows[j]['script_id']}"
            pair_of[i] = pair_map.get(k, -1)

pair_of = np.array(pair_of)
ok = pair_of >= 0
print(f"  能映射的样本: {ok.sum()}/{len(pair_of)}")

f = feat.astype(np.float64)
fn = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-12)
f = f[ok]
fn = fn[ok]
p = pair_of[ok]

n_pairs = int(m["num_pairs"])
print(f"\n  === 结果 ===")

# 1) 所有样本两两余弦的总体水平
rng = np.random.default_rng(0)
ix = rng.choice(len(fn), 4000, replace=False)
sub = fn[ix]
C = sub @ sub.T
off = C[~np.eye(len(sub), dtype=bool)]
print(f"  样本间余弦(随机4000): mean={off.mean():.4f} max={off.max():.4f}")

# 2) 同 pair 内 vs 不同 pair 间
cent = np.zeros((n_pairs, fn.shape[1]), dtype=np.float64)
cnt = np.zeros(n_pairs)
np.add.at(cent, p, fn)
np.add.at(cnt, p, 1.0)
good = cnt >= 20
cent[good] /= cnt[good][:, None]
cn = cent[good] / (np.linalg.norm(cent[good], axis=1, keepdims=True) + 1e-12)
CC = cn @ cn.T
offc = CC[~np.eye(len(cn), dtype=bool)]
print(f"  pair 质心间余弦({good.sum()} 个有效pair): mean={offc.mean():.4f} "
      f"max={offc.max():.4f} min={offc.min():.4f}")

# 3) 线性探测：用质心做 1-NN 分类，看准确率
print(f"\n  判读:")
if offc.mean() > 0.7:
    print(f"    ✗ DINO 特征**几乎不区分** pair（质心余弦 {offc.mean():.3f}）")
    print(f"      -> v15 表的坏是**特征本身**的问题，换 K-Means/SupCon 都无效")
    print(f"      -> 必须先换风格特征（或用端到端+强约束让它自己学开）")
else:
    print(f"    ✓ DINO 特征**能区分** pair（质心余弦 {offc.mean():.3f}）")
    print(f"      -> v15 表的坏是 K-Means 用法的问题，SupCon 可修")
