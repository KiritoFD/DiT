# -*- coding: utf-8 -*-
"""诊断：g 是否真正进入前向；glyph_injections 是否 zero-init。"""
import os, sys, inspect, torch
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.model import DiT_2Cond_models

m = DiT_2Cond_models["DiT-2Cond-S/2"](
    input_size=32, in_channels=4,
    num_calligraphers=1013, num_characters=35130,
    use_checkpoint=False, learn_sigma=False,
    condition_fusion="factorized_add",
    callig_embed_dim=128, char_embed_dim=384,
    cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
    cond_drop_which_glyph_prob=0.75,
    skel_head_enabled=False, use_glyph_cond=True,
    glyph_scale_init=0.4, glyph_inject_layers=6,
    char_proj_mode="mlp", freeze_char_table=True,
    norm_type="rms", mlp_type="swiglu", qk_norm=True,
    rope=True, rope_theta=100.0, attn_impl="sdpa",
).cuda()

print("=== forward 签名 ===")
print(inspect.signature(m.forward))

print("\n=== glyph 相关属性 ===")
print("use_glyph_cond     :", m.use_glyph_cond)
print("glyph_embedder     :", type(m.glyph_embedder).__name__ if m.glyph_embedder else None)
print("glyph_inject_layers:", m.glyph_inject_layers)
print("glyph_injections   :", None if m.glyph_injections is None else f"{len(m.glyph_injections)} 个")
print("glyph_inject_at    :", getattr(m, "glyph_inject_at", None))
if m.glyph_injections is not None:
    p = m.glyph_injections[0].proj
    print("inj[0] type        :", type(m.glyph_injections[0]).__name__)
    print("inj[0].proj        :", p)
    print("inj[0] W max|.|    :", p.weight.abs().max().item())
    print("inj[0] b max|.|    :", p.bias.abs().max().item())
print("glyph_scale        :", m.glyph_scale.item())

print("\n=== 前向：g 是否影响输出 ===")
torch.manual_seed(0)
x = torch.randn(2, 4, 32, 32).cuda()
g1 = torch.randn(2, 4, 32, 32).cuda()
g2 = torch.randn(2, 4, 32, 32).cuda()
t = torch.rand(2).cuda()
yc = torch.randint(0, 1013, (2,)).cuda()
yh = torch.randint(0, 35130, (2,)).cuda()

m.eval()
with torch.no_grad():
    o0 = m(x, t, y_callig=yc, y_char=yh)              # 不给 g
    o1 = m(x, t, y_callig=yc, y_char=yh, g=g1)
    o2 = m(x, t, y_callig=yc, y_char=yh, g=g2)
print("out type           :", type(o0))
o0 = o0[0] if isinstance(o0, tuple) else o0
o1 = o1[0] if isinstance(o1, tuple) else o1
o2 = o2[0] if isinstance(o2, tuple) else o2
print("||o1 - o0|| (给g vs 不给g) :", (o1 - o0).norm().item())
print("||o2 - o0|| (换g)          :", (o2 - o0).norm().item())

print("\n=== 反向：关键参数梯度 ===")
m.train()
out = m(x, t, y_callig=yc, y_char=yh, g=g1)
out = out[0] if isinstance(out, tuple) else out
loss = out.float().pow(2).mean()
m.zero_grad(set_to_none=True)
loss.backward()

def gn(p, label):
    v = 0.0 if (p is None or p.grad is None) else p.grad.norm().item()
    print(f"  {label:<34} {v:.8f}")
    return v

print("  glyph_embedder.weight :", end=" ")
gn(m.glyph_embedder.weight, "glyph_embedder.weight")
print("  glyph_scale           :", end=" ")
gn(m.glyph_scale, "glyph_scale")
if m.glyph_injections is not None:
    for k in (0, len(m.glyph_injections) - 1):
        gn(m.glyph_injections[k].proj.weight, f"inj[{k}].proj.weight")
        gn(m.glyph_injections[k].proj.bias, f"inj[{k}].proj.bias")

print("\n=== 对照：blocks 最后一层是否有梯度（确认反向本身正常）===")
gn(m.blocks[-1].attention.qkv.weight if hasattr(m.blocks[-1], "attention") else None,
   "blocks[-1].attn.qkv.weight")
gn(m.final_layer.linear.weight if hasattr(m.final_layer, "linear") else None,
   "final_layer.linear.weight")
