"""端到端验证 4ch -> 12ch 通道扩展：扩展后的 12ch 模型在前 4 通道上应与 4ch 逐位相同。"""
import os
import sys

import torch

os.chdir("/root/Workspace/xy/DiT")
sys.path.insert(0, "/root/Workspace/xy/DiT")
from src.utils.channel_expand import expand_state_dict_4ch_to_12ch  # noqa: E402
from src.model import DiT_2Cond_models  # noqa: E402

C = dict(
    num_calligraphers=45, num_characters=7765, condition_fusion="factorized_cat",
    callig_embed_dim=128, char_embed_dim=384, use_char_cond=False,
    use_glyph_cond=True, glyph_scale_init=0.6, glyph_embedder_depth=2,
    glyph_inject_layers=4, glyph_inject_mode="adaln", glyph_vec_cond=True,
    glyph_vec_dim=128, norm_type="rms", mlp_type="swiglu", qk_norm=1, rope=1,
    attn_impl="sdpa", input_size=32, learn_sigma=False,
)

torch.manual_seed(0)
m4 = DiT_2Cond_models["DiT-2Cond-S/2"](in_channels=4, image_channels=4, **C).eval()
m12 = DiT_2Cond_models["DiT-2Cond-S/2"](in_channels=12, image_channels=4, **C).eval()

sd, exp = expand_state_dict_4ch_to_12ch(m4.state_dict(), n_aux_groups=2)
print("  扩展的键:")
for k in exp:
    print("   ", k)

miss, unexp = m12.load_state_dict(sd, strict=False)
print(f"  load_state_dict: missing={len(miss)} unexpected={len(unexp)}")
if miss:
    print("   missing:", miss[:5])
if unexp:
    print("   unexpected:", unexp[:5])

B = 2
x4 = torch.randn(B, 4, 32, 32)
x12 = torch.cat([x4, torch.randn(B, 8, 32, 32)], 1)
t = torch.rand(B)
yc = torch.randint(0, 45, (B,))
yh = torch.zeros(B, dtype=torch.long)
g = torch.randn(B, 4, 32, 32)
with torch.no_grad():
    o4 = m4(x4, t, yc, yh, g=g)
    o12 = m12(x12, t, yc, yh, g=g)

d = (o4 - o12[:, :4]).abs().max().item()
print()
print(f"  4ch 输出 vs 12ch 前 4 通道: max|diff| = {d:.3e}")
print(f"  新通道输出: {float(o12[:, 4:].abs().max()):.3e}")
print("  -> " + ("**逐位无损** ✓" if d < 1e-6 else "**有差异** ✗"))
