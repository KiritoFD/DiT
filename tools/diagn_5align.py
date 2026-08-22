#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全 5 样本对齐诊断: 对 poster 的 show5 5 个字, 验证
  sample{i}(生成) <-> gt{i}(GT原图) <-> remote_gt/{id} 是否逐字对应。
输出: 每个样本的字/图片id, 以及生成骨架 vs GT骨架的覆盖率。
"""
import os, csv, numpy as np
from PIL import Image

HERE=os.path.dirname(os.path.abspath(__file__))
rows=list(csv.DictReader(open(os.path.join(HERE,"show5_eval.csv"),encoding="utf-8")))[:5]
ids=[os.path.basename(r["image_path"])[:-4] for r in rows]
chars=[r["character"] for r in rows]

def load(p):
    return None if not os.path.exists(p) else np.asarray(Image.open(p).convert("L"),dtype=np.float32)

def cov(s,t):
    sa=s>64; ta=t>64
    inter=np.logical_and(sa,ta).sum()
    return inter/max(ta.sum(),1)

step=os.path.join(HERE,"remote_eval_samples","step0005000")
print("=== show5 5个字: id / char / glyph确认 ===")
for i,(id_,ch) in enumerate(zip(ids,chars)):
    s4=load(os.path.join(step,f"sample{i}.png"))
    g4=load(os.path.join(step,f"gt{i}.png"))
    rsk=load(os.path.join(HERE,"remote_gt","skel",f"{id_}.png"))
    c_sg = cov(s4,g4) if (s4 is not None and g4 is not None) else -1
    c_rt = cov(s4,rsk) if (s4 is not None and rsk is not None) else -1
    print(f"  [{i}] id={id_} char={ch} | 生成骨架覆盖GT={c_sg:.3f} 生成骨架覆盖remote_gt_skel={c_rt:.3f}")
print("\n若某样本 c_sg 低(~0) = 生成的不是 GT 那个字; 若 c_rt 低但 c_sg 高 = remote_gt skel 不对应")
