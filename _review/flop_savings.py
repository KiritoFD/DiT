"""全面 FLOP 拆解: 逐个候选优化实测 (S/2, B=1, CPU)。"""
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


def measure(tag, **over):
    kw = dict(COMMON); kw.update(over)
    m = DiT_2Cond_models["DiT-2Cond-S/2"](**kw).eval()
    n = sum(p.numel() for p in m.parameters())
    x = torch.randn(1, 4, 32, 32); t = torch.rand(1)
    yc = torch.randint(0, 52, (1,)); ych = torch.randint(0, 100, (1,))
    g = torch.randn(1, 4, 32, 32)
    with torch.no_grad():
        with FlopCounterMode(display=False) as fc:
            m(x, t, yc, ych, g=g)
    return fc.get_total_flops(), n


BASE, NBASE = measure("base")
print(f"基线 (v12 现状): {BASE/1e9:.3f} GFLOPs/sample, {NBASE/1e6:.2f}M params\n")

CASES = [
    ("glyph_embedder_sep=True        ", dict(glyph_embedder_sep=True)),
    ("glyph_embedder_depth=1         ", dict(glyph_embedder_depth=1)),
    ("glyph_embedder_depth=0         ", dict(glyph_embedder_depth=0)),
    ("glyph_inject_mode=xattn        ", dict(glyph_inject_mode="xattn")),
    ("mlp_ratio=3                    ", dict(mlp_ratio=3.0)),
    ("mlp_ratio=3 + sep              ", dict(mlp_ratio=3.0, glyph_embedder_sep=True)),
    ("mlp_ratio=3 + depth1           ", dict(mlp_ratio=3.0, glyph_embedder_depth=1)),
    ("glyph_inject_layers=2          ", dict(glyph_inject_layers=2)),
    ("无 g 通路 (block 本体)         ", dict(use_glyph_cond=False, glyph_vec_cond=False,
                                             glyph_inject_layers=0)),
]
print(f"{'改动':34s} {'GFLOPs':>8} {'Δ':>8} {'params':>9} {'Δparams':>9}")
print("-" * 74)
for tag, over in CASES:
    t, n = measure(tag, **over)
    print(f"{tag:34s} {t/1e9:8.3f} {100*(t-BASE)/BASE:+7.1f}% "
          f"{n/1e6:8.2f}M {100*(n-NBASE)/NBASE:+8.1f}%")

# 组合最优
t, n = measure("combo", mlp_ratio=3.0, glyph_embedder_sep=True)
print("-" * 74)
print(f"{'组合: mlp_ratio=3 + sep':34s} {t/1e9:8.3f} {100*(t-BASE)/BASE:+7.1f}% "
      f"{n/1e6:8.2f}M {100*(n-NBASE)/NBASE:+8.1f}%")

print(f"\n{'='*74}")
print("MFU 核算 (4090 bf16 dense 峰值 ~82.6 TFLOPs/s, B=240):")
for nm, tt in [("基线", BASE), ("mlp3+sep", t)]:
    step = tt * 240 * 3
    print(f"  {nm:10s} fwd+bwd {step/1e12:6.3f} TFLOPs/step -> 理论上限 "
          f"{82.6e12/step:5.2f} step/s")
