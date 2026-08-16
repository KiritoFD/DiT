#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 标准skel vs GT骨架 对比图: 左右并排 + 叠加, 供人工目检是否同字/拓扑相近。"""
import os, numpy as np
from PIL import Image, ImageDraw

HERE=os.path.dirname(os.path.abspath(__file__))
STD=os.path.join(HERE,"_fonttest","std_skel")
GTD=os.path.join(HERE,"_fonttest","std_gt")
OUT=os.path.join(HERE,"_fonttest","compare_view")

def main():
    os.makedirs(OUT, exist_ok=True)
    for book in ["kai","li"]:
        st,gt=os.path.join(STD,book),os.path.join(GTD,book)
        fs=sorted(f for f in os.listdir(gt) if f.startswith("U+") and os.path.exists(os.path.join(st,f)))
        for f in fs[:8]:
            s=np.asarray(Image.open(os.path.join(st,f)).convert("L"))
            g=np.asarray(Image.open(os.path.join(gt,f)).convert("L"))
            # 白底展示: GT 是黑底白线 => 反白显示为 白底黑线
            s_rgb=np.stack([s]*3,-1); g_rgb=np.stack([g]*3,-1)
            canvas=Image.new("RGB",(256*3,256), (20,20,20))
            canvas.paste(Image.fromarray(255-s_rgb),(0,0))       # 标准(黑线)
            canvas.paste(Image.fromarray(255-g_rgb),(256,0))     # GT(黑线)
            # 叠加: 标准红, GT绿
            comp=np.zeros((256,256,3),np.uint8)
            comp[:,:,0]=(s>128)*255
            comp[:,:,1]=(g>128)*255
            canvas.paste(Image.fromarray(comp),(512,0))
            fp=os.path.join(OUT,f"{book}_{f}")
            canvas.save(fp)
            print(f"  {fp}  (标准|GT|叠加[红=标准,绿=GT])")

if __name__=="__main__":
    main()
