import os, sys, glob
import torch as th

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")

dev = th.device("cuda")

ckpt_path = "exp/v37_skelnet_sp/20261002-123800-v37-sp-real26k/checkpoints/0002500.pt"
d = th.load(ckpt_path, map_location="cpu", weights_only=False)

from src.model.dit import DiT_2Cond_Sp_2
model = DiT_2Cond_Sp_2(
    callig_embed_dim=128,
    glyph_vec_cond=True,
    glyph_vec_dim=128,
    condition_fusion="factorized_cat",
    cond_fusion_norm="split",
    glyph_inject_layers=4,
    glyph_embedder_depth=2,
    num_calligraphers=23,
    use_glyph_cond=True,
    use_char_cond=False,
    learn_sigma=False,
    glyph_scale_init=0.6,
    norm_type="rms",
    mlp_type="swiglu",
    qk_norm=1,
    rope=1,
    rope_theta=100.0
).to(dev).eval()

cache = th.load("data/top10_style23/eval_real200_cache.pt", weights_only=False)
noise = cache["noise"][:4].to(dev)
conds = cache["conds"][:4]
std_lats = cache["std_lats"][:4].to(dev)
gt_lats = cache["gt_lats"][:4].to(dev)

# Compare model vs ema
for key_name in ["model", "ema"]:
    print(f"\n=== Testing {key_name} weights ===")
    sd = d[key_name]
    sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd)

    with th.no_grad():
        t_1 = th.full((4,), 1000.0, device=dev)
        yc = th.tensor([c[0] for c in conds], device=dev, dtype=th.long)
        v_1 = model(noise, t_1, y_callig=yc, y_char=None, g=std_lats)
        print("v_1 norm  :", v_1.flatten(1).norm(dim=1))
        print("gt_lats norm:", gt_lats.flatten(1).norm(dim=1))
        print("v_1 stats: mean=", v_1.mean().item(), "std=", v_1.std().item())
        g_pred = noise - v_1
        print("cos(g_pred, gt_lats):", th.cosine_similarity(g_pred.flatten(1), gt_lats.flatten(1), dim=1))
