"""用**真实的 v13 ckpt** 测 4ch->12ch 扩展（纯 CPU，不占 GPU）。

验证 train.py 里那段接线的核心逻辑：
  读 ckpt -> 扩展 3 个张量 -> load 进 12ch 模型 -> missing/unexpected 应为 0
"""
import glob
import os
import sys

import torch

os.chdir("/root/Workspace/xy/DiT")
sys.path.insert(0, "/root/Workspace/xy/DiT")
from src.utils.channel_expand import expand_ckpt_4ch_to_12ch  # noqa: E402
from src.model import DiT_2Cond_models  # noqa: E402

CKPTS = sorted(glob.glob("assets/results/v13_base_50k/*/checkpoints/*.pt"))
print(f"  v13 ckpt: {[os.path.basename(c) for c in CKPTS]}")
ck_path = CKPTS[-1]
print(f"  用: {ck_path}")

C = dict(
    num_calligraphers=45, num_characters=7765, condition_fusion="factorized_cat",
    callig_embed_dim=128, char_embed_dim=384, use_char_cond=False,
    use_glyph_cond=True, glyph_scale_init=0.6, glyph_embedder_depth=2,
    glyph_inject_layers=4, glyph_inject_mode="adaln", glyph_vec_cond=True,
    glyph_vec_dim=128, norm_type="rms", mlp_type="swiglu", qk_norm=1, rope=1,
    attn_impl="sdpa", input_size=32, learn_sigma=False,
)

ck = torch.load(ck_path, map_location="cpu", weights_only=False)
print(f"  ckpt keys: {list(ck.keys())}")
sd = ck.get("delta", ck.get("model", ck))
print(f"  权重张量: {len(sd)}")

N_AUX = 2
ck2 = expand_ckpt_4ch_to_12ch(ck, N_AUX)
rep = ck2.get("_expand_report", {})
print(f"  扩展报告: { {k: len(v) for k, v in rep.items()} }")

# 建 12ch 模型并 load
m12 = DiT_2Cond_models["DiT-2Cond-S/2"](in_channels=4 + 4 * N_AUX,
                                        image_channels=4, **C)
sd12 = ck2.get("delta", ck2.get("model", ck2))
miss, unexp = m12.load_state_dict(sd12, strict=False)
print()
print(f"  in_channels = {m12.in_channels}  (4 + 4*{N_AUX})")
print(f"  load_state_dict: missing={len(miss)}  unexpected={len(unexp)}")
if miss:
    print(f"    missing: {miss[:8]}")
if unexp:
    print(f"    unexpected: {unexp[:8]}")

# 关键：扩展后那 3 个张量应该是"前 4 通道 = 原值，后 8 通道 = 0"
xw = sd12["x_embedder.proj.weight"]
fw = sd12["final_layer.linear.weight"]
print()
print(f"  x_embedder.proj.weight      {tuple(xw.shape)}  "
      f"新通道是否全零: {bool((xw[:, 4:] == 0).all())}")
print(f"  final_layer.linear.weight   {tuple(fw.shape)}")
h = fw.shape[0] // 12
print(f"    新通道是否全零: {bool((fw.view(h, 12, -1)[:, 4:] == 0).all())}")
print(f"  final_layer.linear.bias     {tuple(sd12['final_layer.linear.bias'].shape)}")
print(f"    新通道是否全零: "
      f"{bool((sd12['final_layer.linear.bias'].view(h, 12)[:, 4:] == 0).all())}")

# 与 4ch ckpt 的前 4 通道比
sd4 = ck.get("delta", ck.get("model", ck))
# ckpt 键带 _orig_mod. 前缀 -> 剥掉再查（与 channel_expand 里的处理一致）
sd4 = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
       for k, v in sd4.items()}
print(f"  剥前缀后键数: {len(sd4)}")
xw4 = sd4["x_embedder.proj.weight"]
fw4 = sd4["final_layer.linear.weight"]
print()
print(f"  x_embedder 前 4 通道 == ckpt 原值: "
      f"{bool(torch.equal(xw[:, :4], xw4))}")
print(f"  final_layer 前 4 通道 == ckpt 原值: "
      f"{bool(torch.equal(fw.view(h, 12, -1)[:, :4], fw4.view(h, 4, -1)))}")
print()
print("  -> " + ("**扩展接线正确，可上 12ch 后训练** ✓"
                 if len(miss) == 0 and len(unexp) == 0 else "**有缺/多余，需查** ✗"))
