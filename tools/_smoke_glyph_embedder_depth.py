"""冒烟验证 glyph_embedder_depth 开关: depth=0 现状 vs depth>0 增强."""
import sys, torch
sys.path.insert(0, ".")
from src.model.dit import DiT_2Cond

for depth in [0, 1, 2, 3]:
    m = DiT_2Cond(input_size=32, in_channels=4, hidden_size=384, depth=4, num_heads=6,
                  patch_size=2, learn_sigma=False, condition_fusion="factorized_add",
                  use_glyph_cond=True, glyph_embedder_depth=depth)
    ge = m.glyph_embedder
    n = sum(p.numel() for p in ge.parameters())
    if depth == 0:
        ok = isinstance(ge, torch.nn.Conv2d)
        nlayers = 1
    else:
        ok = isinstance(ge, torch.nn.Sequential) and len(ge) == 1 + depth * 2
        nlayers = len(ge)
    x = torch.randn(2, 4, 32, 32)
    y = ge(x)
    shape_ok = y.shape == (2, 384, 16, 16)
    print(f"depth={depth}: type={type(ge).__name__}, nlayers={nlayers}, params={n}, out_shape={tuple(y.shape)}, ok={ok and shape_ok}")

n0 = sum(p.numel() for p in m.glyph_embedder.parameters())
print(f"depth=0 params={n0}, expected=6144, match={n0 == 6144}")
print("SMOKE OK")
