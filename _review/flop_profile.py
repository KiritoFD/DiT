"""用 torch FlopCounterMode 精确统计各模块 FLOPs (S/2)。
CPU + B=1 (FLOPs per-sample, 与 batch 无关), 避免与训练争显存。
"""
import sys, torch
sys.path.insert(0, "/root/Workspace/xy/DiT")
from src.model.dit import DiT_2Cond_models
from torch.utils.flop_counter import FlopCounterMode

COMMON = dict(input_size=32, in_channels=4, num_calligraphers=52, callig_embed_dim=128,
              use_char_cond=False, use_glyph_cond=True, glyph_scale_init=0.6,
              glyph_embedder_depth=2, glyph_inject_layers=4, glyph_inject_mode="adaln",
              condition_fusion="factorized_cat", glyph_vec_cond=True,
              norm_type="rms", mlp_type="swiglu", qk_norm=True, rope=True,
              attn_impl="eager", image_channels=4, learn_sigma=False)


def measure(tag, show=True, **over):
    kw = dict(COMMON); kw.update(over)
    m = DiT_2Cond_models["DiT-2Cond-S/2"](**kw).eval()
    x = torch.randn(1, 4, 32, 32); t = torch.rand(1)
    yc = torch.randint(0, 52, (1,)); ych = torch.randint(0, 100, (1,))
    g = torch.randn(1, 4, 32, 32)
    with torch.no_grad():
        with FlopCounterMode(display=False) as fc:
            m(x, t, yc, ych, g=g)
    tot = fc.get_total_flops()
    per = fc.get_flop_counts()
    agg = {}
    for mod, ops in per.items():
        s = sum(ops.values())
        if s > 0:
            nm = mod.split(".")[-1] if "." in mod else mod
            agg[nm] = agg.get(nm, 0) + s
    if show:
        print(f"\n{'='*74}\n{tag}   total = {tot/1e9:.3f} GFLOPs/sample")
        for nm, s in sorted(agg.items(), key=lambda kv: -kv[1])[:12]:
            print(f"    {nm:26s} {s/1e9:8.3f} G  {100*s/tot:6.2f}%")
    return tot, agg


t_ad, a_ad = measure("A. adaln 注入 (v12 现状)")
t_xa, a_xa = measure("B. xattn 注入", glyph_inject_mode="xattn")
print(f"\n  xattn vs adaln: {(t_xa-t_ad)/t_ad*100:+.1f}%")

t_d0, a_d0 = measure("C. glyph_embedder_depth=0 (单层 Conv)", glyph_embedder_depth=0)
print(f"  depth 2->0: {(t_d0-t_ad)/t_ad*100:+.1f}%")

# 逐项拆解: 关掉 glyph 通路看 block 本体
t_nog, _ = measure("D. 无 g 通路 (只留 block 本体)", show=False,
                   use_glyph_cond=False, glyph_vec_cond=False, glyph_inject_layers=0)
print(f"\n  D(仅 block 本体) = {t_nog/1e9:.3f} G")
print(f"  A - D (g 通路总开销) = {(t_ad-t_nog)/1e9:.3f} G = {100*(t_ad-t_nog)/t_ad:.1f}%")
print(f"    其中 glyph_embedder(depth2) = {(a_ad.get('glyph_embedder',0))/1e9:.3f} G "
      f"= {100*a_ad.get('glyph_embedder',0)/t_ad:.1f}%")
print(f"    其中 glyph_injections(adaln) = {(a_ad.get('glyph_injections',0))/1e9:.3f} G "
      f"= {100*a_ad.get('glyph_injections',0)/t_ad:.1f}%")
print(f"    其中 cond_fusion/glyph_vec_proj = "
      f"{(a_ad.get('cond_fusion',0)+a_ad.get('glyph_vec_proj',0))/1e9:.3f} G")

print(f"\n{'='*74}")
print(f"参考: 4090 bf16 dense 峰值 ~82.6 TFLOPs/s (fp16/bf16 with fp32 accum)")
for nm, t in [("adaln(v12)", t_ad), ("xattn", t_xa), ("depth0", t_d0)]:
    step_fwd = t * 240
    step_total = step_fwd * 3          # fwd + bwd ~ 3x
    print(f"  {nm:12s} {t/1e9:6.3f} G/sample -> step(B=240) fwd {step_fwd/1e12:.3f} TFLOPs, "
          f"fwd+bwd {step_total/1e12:.3f} TFLOPs -> @82.6T 理论 {82.6e12/step_total:.2f} step/s")
