"""量化 glyph_embedder 里两层 3x3 conv 的实际贡献 (不需重训)。

做法: 在同一个 ckpt 上对比三次前向
  A. 完整编码器 (depth=2)
  B. 旁路掉两层 3x3 conv (只留首层 Conv, 即 depth=0 的行为)
  C. g=None (完全去掉 g 条件)
指标: ||out_B - out_A|| / ||out_A||   —— 3x3 层的"实际作用"
      ||out_C - out_A|| / ||out_A||   —— 整个 g 通路的作用 (对照)
若 B 的差异远小于 C, 说明那两层对输出几乎无影响 -> 可安全精简。
"""
import sys, os, json, glob
import torch
import torch.nn as nn

sys.path.insert(0, "/root/Workspace/xy/DiT")
from src.model.dit import DiT_2Cond_models

RUN = "assets/results/v12_pretrain_S_cat_fame_kxl_tj_px60"
ck_dirs = sorted(glob.glob(f"{RUN}/*/checkpoints/*.pt"))
ck_dirs = [p for p in ck_dirs if os.path.getsize(p) > 1e6]
ck_path = ck_dirs[-1]
print("ckpt:", ck_path)
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
a = ck.get("args", {})
if not isinstance(a, dict):
    a = vars(a)

MODEL_KW = ["input_size", "num_calligraphers", "num_characters",
            "condition_fusion", "callig_embed_dim", "char_embed_dim",
            "cond_drop_all_prob", "cond_drop_one_prob", "cond_drop_which_glyph_prob",
            "use_char_cond", "glyph_scale_init", "glyph_drop_prob",
            "glyph_inject_layers", "glyph_inject_mode", "glyph_embedder_depth",
            "style_token_n", "style_role_init", "glyph_in_channels",
            "image_channels", "norm_type", "mlp_type", "qk_norm",
            "rope", "rope_theta", "attn_impl", "callig_style_attn", "callig_n_style",
            "callig_spatial", "callig_spatial_rank", "callig_proj_mode",
            "callig_scale_init", "char_proj_mode", "freeze_char_table",
            "glyph_vec_cond", "glyph_vec_dim", "glyph_vec_pool"]
kw = {k: a[k] for k in MODEL_KW if k in a and a[k] is not None}
kw["use_checkpoint"] = False
# use_char_cond 是派生的: train.py 里 = not no_char_cond
kw["use_char_cond"] = not bool(a.get("no_char_cond", False))
# 这两个在 train.py 里是派生出来的, 不在 args 里:
#   in_channels = latent_channels + 4 * len(aux_latent_shards_dirs)
_aux = [s for s in str(a.get("aux_latent_shards_dirs", "") or "").split(",") if s.strip()]
kw["in_channels"] = int(a.get("latent_channels", 4)) + 4 * len(_aux)
#   use_glyph_cond = w_glyph_cond>0 or skel_as_glyph_cond
kw["use_glyph_cond"] = bool(a.get("w_glyph_cond", 0)) or bool(a.get("skel_as_glyph_cond", False))
#   learn_sigma: flow 下 train.py 自动置 False
kw["learn_sigma"] = bool(a.get("learn_sigma", False)) if a.get("learn_sigma") is not None \
    else str(a.get("diffusion_type", "flow")).lower() not in ("flow", "flow_matching", "fm")
name = a.get("model") or "DiT-2Cond-S/2"
print("model:", name, " in_channels:", kw["in_channels"],
      " use_glyph_cond:", kw["use_glyph_cond"], " learn_sigma:", kw["learn_sigma"])
print("glyph_embedder_depth =", kw.get("glyph_embedder_depth"),
      " glyph_inject_layers =", kw.get("glyph_inject_layers"),
      " fusion =", kw.get("condition_fusion"), " gv =", kw.get("glyph_vec_cond"))

dev = torch.device("cuda")
model = DiT_2Cond_models[name](**kw)
sd = ck.get("ema") or ck.get("model")
sd = sd.state_dict() if hasattr(sd, "state_dict") else sd
# torch.compile 会给 state_dict 键加 "_orig_mod." 前缀 -> 必须剥掉
sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"load: missing={len(missing)} unexpected={len(unexpected)}")
if missing:
    print("  missing sample:", missing[:5])
model = model.to(dev).eval()
print("encoder:", type(model.glyph_embedder).__name__,
      "len=", len(model.glyph_embedder) if isinstance(model.glyph_embedder, nn.Sequential) else 1)

B = 4
torch.manual_seed(0)
x = torch.randn(B, kw["in_channels"], 32, 32, device=dev)
t = torch.rand(B, device=dev)
yc = torch.randint(0, kw["num_calligraphers"], (B,), device=dev)
g = torch.randn(B, 4, 32, 32, device=dev)

with torch.no_grad():
    oA = model(x, t, yc, torch.zeros_like(yc), g=g).float()
    oC = model(x, t, yc, torch.zeros_like(yc), g=None).float()
    # 旁路: 用首层 Conv 的输出直接当编码器输出 (等价 depth=0 的行为)
    full = model.glyph_embedder
    model.glyph_embedder = full[0]
    oB = model(x, t, yc, torch.zeros_like(yc), g=g).float()
    model.glyph_embedder = full

rel_B = ((oB - oA).norm() / oA.norm().clamp_min(1e-8)).item()
rel_C = ((oC - oA).norm() / oA.norm().clamp_min(1e-8)).item()
print(f"\n[B] 旁路两层 3x3 conv 的输出变化 = {rel_B:.5f}")
print(f"[C] 完全去掉 g 的输出变化        = {rel_C:.5f}")
print(f"\n比值 B/C = {rel_B/max(rel_C,1e-9):.3f}")
print("  解读: 若 B/C 很小 (<0.2), 那两层 3x3 对输出几乎无影响 -> 可精简;")
print("        若 B/C 接近或超过 1, 那两层在实质改变输出 -> 动它有风险。")
