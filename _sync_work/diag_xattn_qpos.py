"""诊断: q_pos 为何不生效 —— 查 ctx_pos 形状 + 注入是否真的被调用 + 分支是否进入。"""
import os
import sys

import torch

os.chdir("/root/Workspace/xy/DiT")
sys.path.insert(0, "/root/Workspace/xy/DiT")
from src.model import DiT_2Cond_models  # noqa: E402

COMMON = dict(
    num_calligraphers=64, num_characters=7765, condition_fusion="factorized_cat",
    callig_embed_dim=128, char_embed_dim=384, use_char_cond=False,
    use_glyph_cond=True, glyph_scale_init=0.6, glyph_embedder_depth=2,
    glyph_inject_layers=4, glyph_inject_mode="xattn", glyph_vec_cond=True,
    glyph_vec_dim=128, norm_type="rms", mlp_type="swiglu", qk_norm=1, rope=1,
    attn_impl="sdpa", input_size=32, in_channels=4, image_channels=4,
    learn_sigma=False,
)

torch.manual_seed(0)
m = DiT_2Cond_models["DiT-2Cond-S/2"](xattn_q_pos=True, **COMMON).eval()
inj = m.glyph_injections[0]
print(f"  x_embedder.num_patches = {m.x_embedder.num_patches}")
print(f"  inj.q_pos              = {inj.q_pos}")
print(f"  inj.ctx_pos.shape      = {tuple(inj.ctx_pos.shape)}   (需要 >= 256 才能进 q_pos 分支)")

# 注入是否真的被调用 + 分支是否进入
called = {"n": 0, "branch": 0}
_orig = inj._inject


def spy(x, context):
    called["n"] += 1
    if inj.q_pos and x.shape[1] <= inj.ctx_pos.shape[1]:
        called["branch"] += 1
    return _orig(x, context)


inj._inject = spy

x = torch.randn(2, 4, 32, 32)
t = torch.rand(2)
yc = torch.randint(0, 64, (2,))
yh = torch.zeros(2, dtype=torch.long)
g = torch.randn(2, 4, 32, 32)
with torch.no_grad():
    m(x, t, yc, yh, g=g)

print(f"  注入被调用次数         = {called['n']}")
print(f"  其中进入 q_pos 分支    = {called['branch']}")
print()
if called["branch"] == 0:
    print("  -> **q_pos 分支从未进入** -> 这就是开关不生效的原因")
else:
    print("  -> 分支进入了，但输出仍无差异 => 需查别处")
