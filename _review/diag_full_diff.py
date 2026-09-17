"""定位: 全模型里 q_pos 的差异为何消失。

逐步核对:
  ① 两个模型的权重是否真的逐位相同（除 q_pos 标志）
  ② 输入是否相同
  ③ 中间层残差流是否出现差异
  ④ 最终输出是否出现差异
"""
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


def build(qp):
    torch.manual_seed(1234)
    m = DiT_2Cond_models["DiT-2Cond-S/2"](xattn_q_pos=qp, **COMMON).eval()
    with torch.no_grad():
        for inj in m.glyph_injections:
            inj.out_proj.weight.normal_(0, 0.02)
            inj.out_proj.bias.normal_(0, 0.02)
    return m


ma, mb = build(False), build(True)

# ① 权重是否一致
sa, sb = ma.state_dict(), mb.state_dict()
diff_keys = [k for k in sa if not torch.equal(sa[k], sb[k])]
print(f"  ① 权重差异键数: {len(diff_keys)}  {diff_keys[:3]}")
print(f"     inj[0].q_pos: {ma.glyph_injections[0].q_pos} vs {mb.glyph_injections[0].q_pos}")
print(f"     inj[0].out_proj 非零: {bool((ma.glyph_injections[0].out_proj.weight != 0).any())}")

# ② 相同输入
torch.manual_seed(7)
x = torch.randn(2, 4, 32, 32)
t = torch.rand(2)
yc = torch.randint(0, 64, (2,))
yh = torch.zeros(2, dtype=torch.long)
g = torch.randn(2, 4, 32, 32)

# ③ 抓中间残差流
hooks = {}


def mk_hook(tag, model_key):
    def h(mod, inp, out):
        hooks.setdefault(tag, {})[model_key] = (
            out[0].detach() if isinstance(out, tuple) else out.detach())
    return h


for i, blk in enumerate(ma.blocks):
    blk.register_forward_hook(mk_hook(f"blk{i}", "a"))
for i, blk in enumerate(mb.blocks):
    blk.register_forward_hook(mk_hook(f"blk{i}", "b"))

with torch.no_grad():
    oa = ma(x, t, yc, yh, g=g)
    ob = mb(x, t, yc, yh, g=g)

print()
print("  ③ 各 block 输出的差异（第一个出现非零差异的 block 就是注入点）:")
first = None
for i in range(len(ma.blocks)):
    ta, tb = hooks[f"blk{i}"]["a"], hooks[f"blk{i}"]["b"]
    d = (ta - tb).abs().max().item()
    if d > 0 and first is None:
        first = i
    if i <= 13:
        print(f"     blk{i:>2}: max|diff| = {d:.3e}" + ("   <- 首次出现差异" if i == first else ""))

print()
print(f"  ④ 最终输出 max|diff| = {(oa - ob).abs().max().item():.3e}")
print(f"     注入层在: {ma.glyph_inject_at}")
