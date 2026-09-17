"""验证双轴 CFG 实现。

C1  cfg_glyph=None -> 走经典 2 路, 输出 shape 与旧行为一致 (向后兼容)
C2  代数恒等式: cfg_callig=1, cfg_glyph=1, w_inter=1 -> eps 必须精确等于 full pass
C3  cfg_callig=0, cfg_glyph=1, w_inter=0 -> 内容轴单独生效 (旧实现下这会是 no-op)
C4  g=None 不崩
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

# 打破 zero-init, 否则输出恒为 0 (第一版测试就栽在这)
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

print("=" * 70)
print("C1  向后兼容: cfg_glyph=None 走经典 2 路")
with torch.no_grad():
    o = m.forward_with_cfg(x, t, yc, yh, cfg_scale=0.7, g=g)
print(f"   shape={tuple(o.shape)} (期望 ({B}, 4, 32, 32))  -> " +
      ("PASS" if tuple(o.shape) == (B, 4, 32, 32) else "FAIL"))

print("=" * 70)
print("C2  代数恒等式: cfg_callig=1, cfg_glyph=1, w_inter=1 应 == full pass")
with torch.no_grad():
    o_id = m.forward_with_cfg(x, t, yc, yh, cfg_scale=1.0, g=g,
                              cfg_glyph=1.0, w_inter=1.0)
    o_full = m.forward(x, t, yc, yh, g=g)
    if isinstance(o_full, tuple):
        o_full = o_full[0]
d = (o_id - o_full).abs().max().item()
print(f"   |2axis(1,1,1) - full|_max = {d:.3e}  -> " + ("PASS" if d < 1e-4 else "FAIL"))

print("=" * 70)
print("C3  内容轴单独生效: cfg_callig=0, cfg_glyph=1 应与 callig=0,glyph=0 不同")
with torch.no_grad():
    o_c = m.forward_with_cfg(x, t, yc, yh, cfg_scale=0.0, g=g, cfg_glyph=1.0)
    o_u = m.forward_with_cfg(x, t, yc, yh, cfg_scale=0.0, g=g, cfg_glyph=0.0)
d = (o_c - o_u).abs().max().item()
print(f"   |content-only - uncond|_max = {d:.3e}  -> " + ("PASS (内容轴有效)" if d > 1e-4 else "FAIL"))

print("=" * 70)
print("C4  g=None 不崩")
try:
    with torch.no_grad():
        o = m.forward_with_cfg(x, t, yc, yh, cfg_scale=1.5, g=None, cfg_glyph=2.0)
    print(f"   shape={tuple(o.shape)}  -> PASS")
except Exception as e:
    print(f"   FAIL: {type(e).__name__}: {e}")

print("=" * 70)
print("C5  12ch 场景 (in_channels=8, image_channels=4) 的 rest 通道处理")
m2 = DiT_2Cond_models["DiT-2Cond-XS/2"](
    input_size=32, in_channels=8, num_calligraphers=8, callig_embed_dim=64,
    use_char_cond=False, use_glyph_cond=True, glyph_scale_init=0.6,
    glyph_embedder_depth=0, glyph_inject_layers=2, glyph_inject_mode="adaln",
    condition_fusion="factorized_cat", glyph_vec_cond=True,
    norm_type="rms", mlp_type="swiglu", qk_norm=True, rope=True,
    attn_impl="sdpa", image_channels=4, learn_sigma=False, use_checkpoint=False,
    num_characters=100, char_embed_dim=64,
).to(dev).eval()
with torch.no_grad():
    m2.final_layer.linear.weight.normal_(0, 0.05)
    m2.final_layer.adaLN_modulation[-1].weight.normal_(0, 0.05)
    m2.final_layer.adaLN_modulation[-1].bias.normal_(0, 0.05)
x8 = torch.randn(B, 8, 32, 32, device=dev)
with torch.no_grad():
    o = m2.forward_with_cfg(x8, t, yc, yh, cfg_scale=1.0, g=g, cfg_glyph=1.0, w_inter=1.0)
    f = m2.forward(x8, t, yc, yh, g=g)
    if isinstance(f, tuple):
        f = f[0]
print(f"   shape={tuple(o.shape)} (期望 ({B}, 8, 32, 32))  "
      f"|2axis(1,1,1)-full|_max={(o-f).abs().max().item():.3e}")
print(f"   aux 通道(4:8) 是否保持 full 值: "
      f"{torch.allclose(o[:, 4:], f[:, 4:], atol=1e-4)}  -> " +
      ("PASS" if torch.allclose(o[:, 4:], f[:, 4:], atol=1e-4) else "FAIL"))
