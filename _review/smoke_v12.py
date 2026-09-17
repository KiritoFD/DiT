import sys, torch
sys.path.insert(0, "/root/Workspace/xy/DiT")
from src.model.dit import DiT_2Cond_models

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV, "torch:", torch.__version__)

COMMON = dict(
    input_size=32, in_channels=4,
    num_calligraphers=52,
    callig_embed_dim=128,
    use_char_cond=False,          # no_char_cond=True
    use_glyph_cond=True,
    glyph_inject_mode="adaln",
    glyph_inject_layers=4,
    glyph_scale_init=0.6,
    glyph_embedder_depth=2,
    norm_type="rms", mlp_type="swiglu", qk_norm=True, rope=True,
    attn_impl="sdpa", image_channels=4,
)


def build(fusion, glyph_vec):
    kw = dict(COMMON)
    kw["condition_fusion"] = fusion
    kw["glyph_vec_cond"] = glyph_vec
    return DiT_2Cond_models["DiT-2Cond-S/2"](**kw).to(DEV)


def run(tag, fusion, glyph_vec):
    print("\n" + "=" * 64)
    print(f"{tag}: fusion={fusion}  glyph_vec_cond={glyph_vec}")
    m = build(fusion, glyph_vec)
    n_param = sum(p.numel() for p in m.parameters())
    print(f"  params: {n_param/1e6:.4f}M")
    print(f"  cond_fusion    -> {m.cond_fusion if m.cond_fusion is None else m.cond_fusion[1]}")
    print(f"  glyph_vec_proj -> {'None' if m.glyph_vec_proj is None else m.glyph_vec_proj[1]}")
    print(f"  glyph_vec_out  -> {'None' if m.glyph_vec_out is None else m.glyph_vec_out[1]}")

    B = 2
    x = torch.randn(B, 4, 32, 32, device=DEV)
    t = torch.rand(B, device=DEV)
    yc = torch.randint(0, 52, (B,), device=DEV)
    ych = torch.randint(0, 100, (B,), device=DEV)
    g = torch.randn(B, 4, 32, 32, device=DEV)

    # forward 两种 g 情形都要过 (g=None 时 glyph_vec 走零向量, 保 DDP 参数参与)
    m.train()
    o1 = m(x, t, yc, ych, g=g)
    o2 = m(x, t, yc, ych, g=None)
    print(f"  forward(g) ok {tuple(o1.shape)}   forward(g=None) ok {tuple(o2.shape)}")

    tgt = torch.randn(B, 8, 32, 32, device=DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    for step in range(6):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(m(x, t, yc, ych, g=g), tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    print(f"  after 6 steps loss={loss.item():.4f}")

    if fusion == "factorized_cat":
        gn = m.cond_fusion[1].weight.grad.norm().item()
        print(f"  cond_fusion.grad_norm = {gn:.6e}")
        assert gn > 0, "cond_fusion got NO gradient"
    else:
        gn = m.callig_proj[1].weight.grad.norm().item()
        print(f"  callig_proj.grad_norm = {gn:.6e}")
        assert gn > 0
    if glyph_vec:
        gn2 = m.glyph_vec_proj[1].weight.grad.norm().item()
        print(f"  glyph_vec_proj.grad_norm = {gn2:.6e}")
        assert gn2 > 0, "glyph_vec_proj got NO gradient"

    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    print(f"  params w/o grad (DDP hazard): {len(missing)}")
    for n in missing[:8]:
        print(f"     - {n}")
    assert not missing, "unused params -> DDP will error"

    m.eval()
    with torch.no_grad():
        oc = m.forward_with_cfg(x, t, yc, ych, cfg_scale=0.7, g=g)
    print(f"  forward_with_cfg ok: {tuple(oc.shape)}")
    return n_param


a = run("A. v12 target", "factorized_cat", True)
b = run("B. fusion control (add)", "factorized_add", True)
c = run("C. cat WITHOUT glyph vec (degenerate)", "factorized_cat", False)
d = run("D. add WITHOUT glyph vec (baseline)", "factorized_add", False)

print("\n" + "=" * 64)
print(f"A cat+glyphvec = {a/1e6:.4f}M")
print(f"B add+glyphvec = {b/1e6:.4f}M   (A-B = {(a-b)/1e6:+.4f}M)")
print(f"C cat only     = {c/1e6:.4f}M")
print(f"D add only     = {d/1e6:.4f}M   (C-D = {(c-d)/1e6:+.4f}M)  <- 应约为 0: 单因子退化")
print("\nALL SMOKE TESTS PASSED")
