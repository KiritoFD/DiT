# -*- coding: utf-8 -*-
"""_chk_resume_compat.py — 结构改了(新增风格token/换GlyphStyleCrossAttn)能否 resume 旧 ckpt?

检查三件事:
  1) key 层面: 旧 ckpt 与新模型的 missing / unexpected / 形状不匹配
  2) 语义层面: load 后 out_proj 是否被旧值覆盖为非零 (新风格token会否污染输出)
  3) 结论: 表面兼容 vs 真正安全
远程跑: /opt/conda/envs/cu121/bin/python _sync_work/_chk_resume_compat.py
"""
import os
import sys
import glob
import numpy as np
import torch

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
from src.model.dit import DiT_2Cond_models  # noqa: E402

CKPT = (sorted(glob.glob(
    "/root/Workspace/xy/DiT/assets/results/v10b_stdskel_fame3_c41x/*/checkpoints/*.pt"))
    or sorted(glob.glob(
    "/root/Workspace/xy/DiT/assets/results/*c41x*/*/checkpoints/*.pt")))
if not CKPT:
    print("no c41x ckpt found")
    raise SystemExit(0)
ck = CKPT[-1]
print("旧 ckpt:", ck)


def build(n_style):
    return DiT_2Cond_models["DiT-2Cond-Sp/2"](
        input_size=32, in_channels=4,
        num_calligraphers=41, num_characters=35130,
        condition_fusion="factorized_add",
        callig_embed_dim=128, char_embed_dim=384,
        use_glyph_cond=True, glyph_inject_layers=12,
        glyph_inject_mode="xattn", glyph_embedder_depth=2,
        style_token_n=n_style, use_char_cond=False,
        learn_sigma=False,         # 与旧 ckpt 一致 (final_layer 输出 16=2*2*4)
    )


# 新结构模型
m = build(32)
sd_new = m.state_dict()
d = torch.load(ck, map_location="cpu")
old = d.get("delta", d)
# 旧 ckpt 带 _orig_mod. 前缀(compile) -> 统一去掉
old = {k.replace("_orig_mod.", ""): v for k, v in old.items() if k != "args"}

newk, oldk = set(sd_new.keys()), set(old.keys())
missing = sorted(k for k in newk - oldk if "pos_embed" not in k)
unexpected = sorted(oldk - newk)
shape_bad = [k for k in (newk & oldk)
             if hasattr(old[k], "shape") and tuple(old[k].shape) != tuple(sd_new[k].shape)]

print(f"\n=== 1) key 兼容性 ===")
print(f"  新模型参数: {len(newk)}   旧ckpt: {len(oldk)}")
print(f"  missing(新模型有, ckpt无): {len(missing)}")
for k in missing[:12]:
    print(f"      - {k}")
print(f"  unexpected(ckpt有, 新模型无): {len(unexpected)}")
for k in unexpected[:12]:
    print(f"      - {k}")
print(f"  形状不匹配: {len(shape_bad)}  {shape_bad[:5]}")

print(f"\n=== 2) 语义安全性: load 后 out_proj 是否为零 ===")
# 模拟: 构建(零初始化) -> load
m2 = build(32)
before = float(m2.glyph_injections[0].out_proj.weight.abs().mean())
res = m2.load_state_dict(old, strict=False)
after = float(m2.glyph_injections[0].out_proj.weight.abs().mean())
print(f"  构建后 out_proj |mean| = {before:.3e}  (zero-init)")
print(f"  load 后 out_proj |mean| = {after:.3e}")
print(f"  -> {'⚠ 已被旧值覆盖为非零' if after > 1e-8 else '✓ 仍为零(安全)'}")
inj_keys = [k for k in old if "injections" in k and "out_proj" in k]
print(f"  旧ckpt含 injections 参数: {len(inj_keys)} 个 (会被直接载入)")

print(f"\n=== 3) 结论 ===")
if not shape_bad and len(unexpected) == 0:
    print("  表面兼容: key 可载入 (两类的子模块命名/形状一致)")
else:
    print("  不兼容: 存在形状/多余 key")
if after > 1e-8:
    print("  ⚠ 语义不安全: load 后 out_proj 非零, 而 style token 是随机新参数")
    print("     -> 随机风格向量会经已训练的 out_proj 注入残差流 = 污染已训模型")
    print("     -> 若要 resume, 必须在 load 后显式重新 zero-init injections.out_proj")
else:
    print("  语义安全")
