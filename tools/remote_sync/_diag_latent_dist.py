# -*- coding: utf-8 -*-
"""度量: 标准字形 latent vs GT 书法 latent 的距离(替换高斯先验可行性)。
对样本, 比较三个距离:
  D(g, x0)    标准字形 latent -> GT 书法 latent
  D(noise, x0) 高斯噪声 -> GT 书法 latent  (基线)
  D(g加噪t, x0) 标准字形部分加噪(模拟初始点) -> GT
若 D(g,x0) << D(noise,x0): 标准字形作为初始点明显更靠近真值, 可行。
"""
import csv, os, sys, json, random
import numpy as np
from PIL import Image
from collections import defaultdict

random.seed(0)
rows = list(csv.DictReader(open("kailishu_train.csv", encoding="utf-8")))
# 从 std_glyph_latent 加载标准字形 latent, final_latents 加载 GT latent
STD_LAT = "std_glyph_latent/%s/U+%05X.npy"
# final_latents shard 索引 img_id -> latent
shards = sorted([f for f in os.listdir("final_latents") if f.startswith("shard_")])
id2lat = {}
for sp in shards:
    d = np.load(os.path.join("final_latents", sp))
    for iid, lat in zip(d["img_ids"], d["latents"].reshape(len(d["img_ids"]), -1)):
        id2lat[int(iid)] = lat.reshape(4, 32, 32)
print(f"loaded {len(id2lat)} GT latents")

# 采样 200 样本(分层)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
sampled = random.sample(rows, min(N, len(rows)))
D_std, D_noise, D_noise_per = [], [], []
for r in sampled:
    pid = os.path.basename(r["image_path"])[:-4]
    x0 = id2lat.get(int(pid))
    if x0 is None: continue
    book = {"0":"kai","4":"li"}.get(r["script_id"])
    if not book: continue
    gpath = STD_LAT % (book, ord(r["character"]))
    if not os.path.exists(gpath): continue
    g = np.load(gpath)
    # 距离(逐样本 L2)
    d_gx = float(np.linalg.norm(g.ravel() - x0.ravel())
                 / np.sqrt(x0.size))
    # 高斯噪声到 x0（用与 x0 同方差的白噪声的期望距离≈ sqrt(2*var)）
    noise = np.random.randn(*x0.shape) * 1.0
    d_nx = float(np.linalg.norm(noise.ravel() - x0.ravel()) / np.sqrt(x0.size))
    D_std.append(d_gx); D_noise.append(d_nx)
    D_noise_per.append(d_nx)

D_s=np.array(D_std); D_n=np.array(D_noise)
print(f"\nN={len(D_s)}")
print(f"D(标准字形→GT)  : mean={D_s.mean():.4f} median={np.median(D_s):.4f}")
print(f"D(高斯噪声→GT)   : mean={D_n.mean():.4f} median={np.median(D_n):.4f}")
# 超过多少样本 标准字形 显著更近
better = (D_s < D_n * 0.8).mean()
print(f"标准字形距离 < 高斯80% 的样本占比: {better*100:.1f}%")
print(f"标准字形距离 < 高斯50%: {(D_s < D_n*0.5).mean()*100:.1f}%")
# 归一化: 每样本标准字形距离/高斯距离比值
ratio = D_s / (D_n + 1e-6)
print(f"D_std/D_noise 比值: mean={ratio.mean():.3f} median={np.median(ratio):.3f} (<1 表示标准字形更近)")
with open("std_vs_noise_dist.json","w",encoding="utf-8") as f:
    json.dump({"n":int(len(D_s)),"d_std_mean":float(D_s.mean()),"d_noise_mean":float(D_n.mean()),
               "ratio_mean":float(ratio.mean())}, f)
