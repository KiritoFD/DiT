# -*- coding: utf-8 -*-
"""diag_repa_dev.py - reproduce proj device mismatch."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.loss.repa import build_repa_module
from src.utils.dino_cache import DinoFeatureCache

cache = DinoFeatureCache("data/dino_cache/v11_sym_full")
repa = build_repa_module(student_dim=432, layers=(8,),
                         teacher_ckpt="data/pretrained/dinov2_vits14_pretrain.safetensors",
                         w_repa=0.03, warmup_steps=0, device="cuda",
                         feature_cache=cache)
print("proj device:", next(repa.losses[0].proj.parameters()).device)
print("teacher:", repa.losses[0].teacher)
sf = torch.randn(2, 256, 432, device="cuda")
img = torch.rand(2, 3, 256, 256, device="cuda")
ids = torch.tensor([76, 77])
out = repa({8: sf}, img, step=0, img_ids=ids)
print("loss ok:", out.item())
