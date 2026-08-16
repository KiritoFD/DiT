# -*- coding: utf-8 -*-
"""排查 '昌' 为何生成不对：验证训练数据里 callig/glyph/标准字形 latent 是否正确对应。
对 kailishu_train.csv 里 CALLIG=39(于右任) 的 '昌' 样本(若有), 打印：
  - y_char(glyph)=2377? char='昌'?
  - dataset 返回的 g(latent) 是否 = std_glyph_latent/kai/U+0660C.npy
  - 该样本的 GT latent(x0, from final_latents) 与非 target 的差异
"""
import csv, os, glob, numpy as np, torch
from latent_dataset import MCCDLatentDataset

rows = list(csv.DictReader(open("kailishu_train.csv", encoding="utf-8")))
# 找 '昌'
picked = [r for r in rows if r["character"] == "昌"]
print(f"'昌' 样本: {len(picked)}")
for r in picked[:3]:
    print(f"  callig={r['calligrapher']}({r['calligrapher_id']}) script={r['script']} "
          f"char={r['character']} glyph={r['glyph_id']} img={r['image_path']} key={r['std_glyph_key']}")

# 用 dataset 验证一个样本
ds = MCCDLatentDataset(csv_file="kailishu_train.csv", latent_shards_dir="final_latents",
                       img_root="final_imgs_256", image_size=256,
                       load_canny=False, load_skel=False, load_image=True,
                       preload=False, structure_size=256, use_glyph_cond=True)
# 找第一个昌的 index
idx = next(i for i, r in enumerate(ds.samples) if r["character"] == "昌")
b = ds[idx]
print(f"\ndataset[{idx}] char={ds.samples[idx]['character']} glyph={ds.samples[idx]['glyph_id']}")
print(f"  y_callig={b['y_callig'].item()}, y_char={b['y_char'].item()}, g.shape={tuple(b['g'].shape)}")
# g 是否=昌的标准字形latent
ref = np.load("std_glyph_latent/kai/U+0660C.npy")  # 楷昌
print(f"  b['g'] == 楷昌 latent ref?  diff = {float((b['g'].numpy()-ref).sum())}")
# 用另一个字(鼎 U+9F0E)的 ref 对比
ref_ding = np.load("std_glyph_latent/kai/U+09F0E.npy")
print(f"  b['g'] vs 楷鼎 ref diff = {float((b['g'].numpy()-ref_ding).sum())} (应更大, 证明 g 确实是昌)")
