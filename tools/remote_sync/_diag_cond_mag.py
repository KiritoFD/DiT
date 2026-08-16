# -*- coding: utf-8 -*-
"""诊断：对比"预训练 XL 的条件向量(c=t_emb+y_emb)"与"我们的随机条件向量"幅度。
判断条件错位程度。"""
import torch, numpy as np

HERE_arg = __file__
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import DiT_2Cond_models
from download import find_model

torch.manual_seed(0)
model = DiT_2Cond_models["DiT-2Cond-XL/2"](
    input_size=32, num_calligraphers=1011, num_characters=35130,
    use_checkpoint=False, condition_fusion="xl_highdim",
    callig_embed_dim=384, char_embed_dim=768,
    cond_drop_all_prob=0.10, cond_drop_one_prob=0.30)
# 加载预训练权重到 body
pre = find_model("pretrained_models/DiT-XL-2-256x256.pt")
pre2 = {k: v for k, v in pre.items()
        if not k.startswith(("y_embedder", "cond_fusion", "callig_proj",
                             "char_proj", "y_callig", "y_char"))}
model.load_state_dict(pre2, strict=False)

# 对比 t_emb 幅度 vs 我们的随机 cond y_emb 幅度
model.eval()
with torch.no_grad():
    t = torch.tensor([500, 900, 100, 700])
    t_emb = model.t_embedder(t)  # (4, 1152)
    yc = torch.tensor([0, 1, 5, 100])
    yg = torch.tensor([0, 100, 500, 30000])
    e_c = model.y_callig_embedder(yc, False)
    e_g = model.y_char_embedder(yg, False)
    y_emb_raw = model.cond_fusion(torch.cat([e_c, e_g], dim=-1))
    y_emb = y_emb_raw / (torch.norm(y_emb_raw, dim=-1, keepdim=True) + 1e-6) * model.y_scale

    print("t_emb  norm per sample:", torch.norm(t_emb, dim=1).tolist())
    print("t_emb  mean abs:", t_emb.abs().mean().item())
    print("y_emb(缩放后) norm per sample:", torch.norm(y_emb, dim=1).tolist())
    print("y_scale =", model.y_scale.item())
    print("c = t_emb + y_emb norm:", torch.norm(t_emb + y_emb, dim=1).tolist())

    # 预训练官方：y_embedder(1001,1152) 的标准 norm 作为参照
    yw = pre["y_embedder.embedding_table.weight"]
    ynorms = torch.norm(yw, dim=1)
    print("\n官方 y_embedder row norm: mean", ynorms.mean().item(), "std", ynorms.std().item(),
          "min", ynorms.min().item(), "max", ynorms.max().item())
    print("官方 t_emb(固定) norm 参考上面；官方 c 的 y 部分 norm ~%.1f" % ynorms.mean().item())
