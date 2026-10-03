"""对照实验: 分离"逻辑错"与"fp32 batch 归约噪声"。

思路: 把**完全相同**的样本拼成 batch 4 跑一次, 再单独跑一次,
如果两者也有 ~1e-3 的差异, 那 C2/C5 的残差就纯粹是数值噪声。
"""
import os, sys
import torch

os.chdir("/root/Workspace/xy/DiT")
sys.path.insert(0, "/root/Workspace/xy/DiT")
from src.model import DiT_2Cond_models

dev = "cuda"
torch.manual_seed(0)
m = DiT_2Cond_models["DiT-2Cond-XS/2"](
    input_size=32, in_channels=4, num_calligraphers=8, callig_embed_dim=64,
    use_char_cond=False, use_glyph_cond=True, glyph_scale_init=0.6,
    glyph_embedder_depth=0, glyph_inject_layers=2, glyph_inject_mode="xattn",
    condition_fusion="factorized_cat", glyph_vec_cond=True,
    norm_type="rms", mlp_type="swiglu", qk_norm=True, rope=True,
    attn_impl="sdpa", image_channels=4, learn_sigma=False, use_checkpoint=False,
    num_characters=100, char_embed_dim=64, xattn_q_pos=True,
).to(dev).eval()
with torch.no_grad():
    m.final_layer.linear.weight.normal_(0, 0.05)
    m.final_layer.adaLN_modulation[-1].weight.normal_(0, 0.05)
    m.final_layer.adaLN_modulation[-1].bias.normal_(0, 0.05)
    for inj in m.glyph_injections:
        inj.out_proj.weight.normal_(0, 0.02)

B = 3
x = torch.randn(B, 4, 32, 32, device=dev)
t = torch.rand(B, device=dev)
yc = torch.randint(0, 8, (B,), device=dev)
yh = torch.zeros(B, dtype=torch.long, device=dev)
g = torch.randn(B, 4, 32, 32, device=dev)

with torch.no_grad():
    single = m.forward(x, t, yc, yh, g=g)
    if isinstance(single, tuple):
        single = single[0]
    # 把同样的样本重复 4 次拼成 batch=4B, 取前 B
    quad = m.forward(x.repeat(4, 1, 1, 1), t.repeat(4), yc.repeat(4), yh.repeat(4),
                     g=g.repeat(4, 1, 1, 1))
    if isinstance(quad, tuple):
        quad = quad[0]

d = (quad[:B] - single).abs().max().item()
scale = single.abs().max().item()
print(f"输出量级 max|out| = {scale:.3f}")
print(f"对照: 同一样本 batch=1 vs batch=12 的差异 = {d:.3e}  (相对 {d/scale:.2e})")
print("-> " + ("说明 ~1e-3 是 fp32 batch 归约噪声, 不是逻辑错"
              if d > 1e-4 else "说明 batching 几乎无噪声, 那 C2 的残差需要另找原因"))

print()
print("=== 用相对容差重判 C2 ===")
with torch.no_grad():
    o_id = m.forward_with_cfg(x, t, yc, yh, cfg_scale=1.0, g=g, cfg_glyph=1.0, w_inter=1.0)
    o_full = m.forward(x, t, yc, yh, g=g)
    if isinstance(o_full, tuple):
        o_full = o_full[0]
d2 = (o_id - o_full).abs().max().item()
rel = d2 / o_full.abs().max().item()
print(f"|2axis(1,1,1) - full|_max = {d2:.3e}  相对 {rel:.2e}")
print("-> " + ("PASS (在 batch 噪声量级内)" if d2 <= d * 3 + 1e-5 else "FAIL (超出噪声)"))
