import sys, os
sys.path.insert(0, "/root/Workspace/xy/DiT")
import torch as th
from src.model.dit import DiT_2Cond_B_2, DiT_2Cond_Sp_2, DiT_2Cond_S_2

dev = th.device("cuda" if th.cuda.is_available() else "cpu")

kwargs = dict(
    callig_embed_dim=128,
    glyph_vec_cond=True,
    glyph_vec_dim=128,
    condition_fusion="factorized_cat",
    cond_fusion_norm="split",
    glyph_inject_layers=4,
    glyph_embedder_depth=2,
    num_calligraphers=23
)

m_s = DiT_2Cond_S_2(**kwargs)
m_sp = DiT_2Cond_Sp_2(**kwargs)
m_b = DiT_2Cond_B_2(**kwargs)

p_s = sum(p.numel() for p in m_s.parameters())
p_sp = sum(p.numel() for p in m_sp.parameters())
p_b = sum(p.numel() for p in m_b.parameters())

print(f"S/2  params : {p_s:,} (~{p_s/1e6:.1f}M) [原版基模]")
print(f"Sp/2 params : {p_sp:,} (~{p_sp/1e6:.1f}M) [1.8x 中大模]")
print(f"B/2  params : {p_b:,} (~{p_b/1e6:.1f}M) [3.7x 旗舰基模]")

# Test DiT-B/2 on GPU with batch 128 and batch 192
m_b = m_b.to(dev).train()
for b in [64, 128, 192]:
    th.cuda.empty_cache()
    th.cuda.reset_peak_memory_stats()
    z = th.randn(b, 4, 32, 32, device=dev)
    t = th.full((b,), 500.0, device=dev)
    y = th.zeros(b, dtype=th.long, device=dev)
    y_char = th.zeros(b, dtype=th.long, device=dev)
    g = th.randn(b, 4, 32, 32, device=dev)
    
    with th.autocast("cuda", dtype=th.bfloat16):
        out = m_b(z, t, y_callig=y, y_char=y_char, g=g)
        loss = out.mean()
    loss.backward()
    vram = th.cuda.max_memory_allocated() / (1024**3)
    print(f"  DiT-B/2 (133M) Batch={b:3d} Peak VRAM: {vram:5.2f} GB on {dev}")
