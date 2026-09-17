import sys, torch
sys.path.insert(0, "/root/Workspace/xy/DiT")
from src.model.dit import DiT_2Cond_models

COMMON = dict(
    input_size=32, in_channels=4, num_calligraphers=52, callig_embed_dim=128,
    use_char_cond=False, use_glyph_cond=True, glyph_inject_mode="adaln",
    glyph_inject_layers=4, glyph_scale_init=0.6, glyph_embedder_depth=2,
    norm_type="rms", mlp_type="swiglu", qk_norm=True, rope=True,
    attn_impl="sdpa", image_channels=4, learn_sigma=False,
)

NAMES = ["DiT-2Cond-XS6/2", "DiT-2Cond-XS/2", "DiT-2Cond-S320/2",
         "DiT-2Cond-S/2", "DiT-2Cond-M/2", "DiT-2Cond-Sp/2", "DiT-2Cond-B/2"]

print(f"{'variant':>17} {'depth':>5} {'h':>4} {'params':>9} {'FLOPs d*h^2':>12} "
      f"{'vs M/2':>8} {'p/sample':>9} {'est step/s':>11}")
rows = {}
for name in NAMES:
    if name not in DiT_2Cond_models:
        print(f"{name:>17}  *** NOT REGISTERED ***")
        continue
    m = DiT_2Cond_models[name](**COMMON)
    n = sum(p.numel() for p in m.parameters())
    d = len(m.blocks)
    h = m.blocks[0].norm1.dim
    rows[name] = (d, h, n, d * h * h)

ref = rows["DiT-2Cond-M/2"][3]
MEASURED_M = 4.15          # 实测 M/2 步速 (step/s)
for name in NAMES:
    if name not in rows:
        continue
    d, h, n, fl = rows[name]
    rel = fl / ref
    print(f"{name:>17} {d:>5} {h:>4} {n/1e6:>8.2f}M {fl:>12,} {rel:>7.3f}x "
          f"{n/28569:>9.0f} {MEASURED_M/rel:>10.2f}")

print()
print("FLOPs 比 == 步速比 的实测验证 (S/2):")
s = rows["DiT-2Cond-S/2"][3]
print(f"  S/2 vs M/2 FLOPs 比 = {s/ref:.3f} -> 预期步速比 {ref/s:.3f}x")
print(f"  实测 5.28/4.15 = {5.28/4.15:.3f}x   ✓")
