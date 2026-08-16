#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""XL + 甲2 glyph_cond 结构 smoke: 验证 DiT-2Cond-XL + use_glyph_cond 能构建 + 前向。
不加载 XL 权重(本地无), 只验证结构与维度。"""
import os, sys
import torch
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE,"remote_sync"))
from models import DiT_2Cond_models

def main():
    torch.manual_seed(0)
    model = DiT_2Cond_models["DiT-2Cond-XL/2"](
        input_size=32, num_calligraphers=1011, num_characters=35130,
        use_checkpoint=False, condition_fusion="xl_highdim",
        callig_embed_dim=384, char_embed_dim=768,
        cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
        use_glyph_cond=True)
    print("[xl-smoke] XL + glyph_cond built")
    print("[xl-smoke] glyph_embedder:", model.glyph_embedder)
    print("[xl-smoke] glyph_scale:", model.glyph_scale.item())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).train()
    x=torch.randn(1,4,32,32,device=dev); t=torch.randint(0,1000,(1,),device=dev)
    yc=torch.randint(0,1011,(1,),device=dev); yg=torch.randint(0,35130,(1,),device=dev)
    g=torch.randn(1,4,32,32,device=dev)
    out=model(x,t,yc,yg,g=g)
    print("[xl-smoke] out:", tuple(out.shape))
    assert tuple(out.shape)==(1,8,32,32)
    # CFG
    outf=model.forward_with_cfg(x,t,yc,yg,cfg_scale=4.0,g=g)
    assert tuple(outf.shape)==(1,8,32,32)
    print("[xl-smoke] CFG out:", tuple(outf.shape))
    # backward (glyph_embedder grad)
    loss=out[:,:4].sum()
    loss.backward()
    ge=model.glyph_embedder.weight.grad
    print("[xl-smoke] glyph_embedder.grad abs sum:", float(ge.abs().sum().item()) if ge is not None else None)
    print("[xl-smoke] ALL PASS")

if __name__=="__main__":
    main()
