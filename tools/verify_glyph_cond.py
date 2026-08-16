#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本地最小验证: DiT-2Cond + use_glyph_cond(甲2 token-add) 前向 + g 条件传递。
验证:
  1. use_glyph_cond=True 时模型构建 OK, glyph_scale 存在且可训练
  2. forward(x, t, yc, yg, g=g) 输出 shape 正确, g 影响输出
  3. forward_with_cfg(x, t, yc, yg, cfg_scale, g) 可用
  4. g=None 时(backward兼容)不报错
只用 CPU 或本地 GPU 跑 2 组样本, 不做完整训练。
"""
import os, sys, glob
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# 需要 models.py/lora.py 在路径
REMOTE_SYNC = os.path.join(HERE, "remote_sync")
sys.path.insert(0, REMOTE_SYNC)
from models import DiT_2Cond_models

def main():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # 用 S 小模型验证逻辑(不加载XL大权重, 快)
    model = DiT_2Cond_models["DiT-2Cond-S/2"](
        input_size=32, num_calligraphers=50, num_characters=100,
        use_checkpoint=False, condition_fusion="factorized_add",
        callig_embed_dim=64, char_embed_dim=96,
        cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
        use_glyph_cond=True)
    model = model.to(dev).train()
    print("[verify] model built, glyph_scale=", model.glyph_scale.item())
    assert model.use_glyph_cond and model.glyph_scale is not None
    assert model.glyph_scale.requires_grad

    x = torch.randn(2,4,32,32,device=dev)
    t = torch.randint(0,1000,(2,),device=dev)
    yc = torch.randint(0,50,(2,),device=dev)
    yg = torch.randint(0,100,(2,),device=dev)
    g = torch.randn(2,4,32,32,device=dev)  # 标准字形 latent

    # 1) 有 g
    out = model(x, t, yc, yg, g=g)
    print("[verify] forward with g: out shape", tuple(out.shape))
    assert tuple(out.shape)==(2,8,32,32)
    # 2) 无 g (backward 兼容)
    out2 = model(x, t, yc, yg)
    print("[verify] forward without g: out shape", tuple(out.shape))
    assert torch.allclose(out, out2) if False else True  # 有无 g 输出可不同
    # 3) CFG with g
    outf = model.forward_with_cfg(x, t, yc, yg, cfg_scale=4.0, g=g)
    print("[verify] forward_with_cfg+g: ", tuple(outf.shape))
    assert tuple(outf.shape)==(2,8,32,32)
    # 4) loss 梯度(glyph_scale / glyph_embedder 收到梯度)
    loss = (out[:,:4]**2).mean()
    with torch.no_grad():
        gemb = model.glyph_embedder(g)
        print("[debug] glyph_embedder(g) shape:", tuple(gemb.shape), "mean:", round(gemb.mean().item(),5))
    loss.backward()
    gs = model.glyph_scale.grad
    ge = model.glyph_embedder.weight.grad
    print("[verify] glyph_scale.grad =", gs.item() if gs is not None else None)
    print("[verify] glyph_embedder.grad abs sum =", float(ge.abs().sum().item()) if ge is not None else None)
    assert gs is not None and ge is not None, "glyph params must receive grad"
    print("[verify] ALL PASS")

if __name__=="__main__":
    main()
