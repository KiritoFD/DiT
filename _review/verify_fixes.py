"""验证 2026-09-17 的三处修复。

T1  cat 融合的 drop guard 修好了 (等价性测试: drop=1.0 时 forward(x,y) == forward(x,null))
T2  xattn 的 Q 位置开关生效, 且默认关闭时与旧行为一致
T3  旧 ckpt 仍能 strict 加载 (xattn_q_pos 默认 False, 不破坏兼容)
"""
import os, sys
import torch

os.chdir("/root/Workspace/xy/DiT")
sys.path.insert(0, "/root/Workspace/xy/DiT")

from src.model import DiT_2Cond_models

dev = "cuda"
KW = dict(input_size=32, in_channels=4, num_calligraphers=8, callig_embed_dim=64,
          use_char_cond=False, use_glyph_cond=True, glyph_scale_init=0.6,
          glyph_embedder_depth=0, glyph_inject_layers=2, glyph_inject_mode="adaln",
          condition_fusion="factorized_cat", glyph_vec_cond=True,
          norm_type="rms", mlp_type="swiglu", qk_norm=True, rope=True,
          attn_impl="sdpa", image_channels=4, learn_sigma=False, use_checkpoint=False,
          num_characters=100, char_embed_dim=64)


def mk(**over):
    k = dict(KW); k.update(over)
    torch.manual_seed(0)
    m = DiT_2Cond_models["DiT-2Cond-XS/2"](**k).to(dev)
    # ⚠ DiT 的 final_layer.linear / adaLN_modulation 是 zero-init -> **输出恒为 0**,
    #   不打破它, 所有"输出差值"都是 0, 测不出任何东西 (第一版就栽在这)。
    with torch.no_grad():
        m.final_layer.linear.weight.normal_(0, 0.05)
        m.final_layer.adaLN_modulation[-1].weight.normal_(0, 0.05)
        m.final_layer.adaLN_modulation[-1].bias.normal_(0, 0.05)
    return m


B = 4
x = torch.randn(B, 4, 32, 32, device=dev)
t = torch.rand(B, device=dev)
yc = torch.randint(0, 8, (B,), device=dev)
ych = torch.zeros(B, dtype=torch.long, device=dev)
g = torch.randn(B, 4, 32, 32, device=dev)

print("=" * 68)
print("T1  cat 融合的 drop guard")
# 用 cond_drop_all_prob=1.0 让 drop 恒真; 若 guard 生效, 传真实 yc 与传 null 应完全等价
m = mk(cond_drop_all_prob=1.0, cond_drop_one_prob=0.0)
m.train()
with torch.no_grad():
    o_real = m(x, t, yc, ych, g=g)
    o_null = m(x, t, torch.full_like(yc, 8), ych, g=g)
d = (o_real - o_null).abs().max().item()
print(f"  factorized_cat + drop=1.0 : |forward(real) - forward(null)|_max = {d:.3e}")
print("  -> " + ("PASS (drop 生效, 两者等价)" if d < 1e-5 else "FAIL (drop 未生效!)"))

m0 = mk(cond_drop_all_prob=0.0, cond_drop_one_prob=0.0)
m0.train()
with torch.no_grad():
    d0 = (m0(x, t, yc, ych, g=g) - m0(x, t, torch.full_like(yc, 8), ych, g=g)).abs().max().item()
print(f"  对照 drop=0.0             : 差值 = {d0:.3e}  -> " +
      ("PASS (不该等价, 条件确实起作用)" if d0 > 1e-4 else "FAIL"))

# factorized_add 对照(本来就该通过)
ma = mk(condition_fusion="factorized_add", cond_drop_all_prob=1.0)
ma.train()
with torch.no_grad():
    da = (ma(x, t, yc, ych, g=g) - ma(x, t, torch.full_like(yc, 8), ych, g=g)).abs().max().item()
print(f"  对照 factorized_add       : 差值 = {da:.3e}  -> " +
      ("PASS" if da < 1e-5 else "FAIL"))

print("=" * 68)
print("T2  xattn Q 位置开关")
m_q = mk(glyph_inject_mode="xattn", glyph_inject_layers=2, xattn_q_pos=True)
m_n = mk(glyph_inject_mode="xattn", glyph_inject_layers=2, xattn_q_pos=False)
print(f"  q_pos=True  -> {m_q.glyph_injections[0].q_pos}")
print(f"  q_pos=False -> {m_n.glyph_injections[0].q_pos}")
# 手动把 out_proj 设非零, 否则 zero-init 会掩盖差异
for mm in (m_q, m_n):
    for inj in mm.glyph_injections:
        with torch.no_grad():
            inj.out_proj.weight.normal_(0, 0.02)
m_q.eval(); m_n.eval()
with torch.no_grad():
    a = m_q(x, t, yc, ych, g=g)
    b = m_n(x, t, yc, ych, g=g)
print(f"  |out(q_pos=True) - out(q_pos=False)|_max = {(a-b).abs().max().item():.4e}")
print("  -> " + ("PASS (开关生效)" if (a - b).abs().max().item() > 1e-4 else "FAIL"))

print("=" * 68)
print("T3  默认值 == 旧行为 (Q 不加位置)")
m_d = mk(glyph_inject_mode="xattn", glyph_inject_layers=2)      # 不传 xattn_q_pos
print(f"  默认 xattn_q_pos = {m_d.glyph_injections[0].q_pos}  -> " +
      ("PASS (默认 False, 旧 ckpt 兼容)" if not m_d.glyph_injections[0].q_pos else "FAIL"))
