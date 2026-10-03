"""冒烟验证 callig_style_attn (书家风格 cross-attn) 开关."""
import sys, torch, glob, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
from src.model import DiT_2Cond_models

def build(**kw):
    base = dict(num_calligraphers=41, num_characters=35130,
                condition_fusion="factorized_add", callig_embed_dim=128,
                char_embed_dim=384, char_proj_mode="mlp", callig_proj_mode="mlp",
                use_glyph_cond=True, use_char_cond=False,
                glyph_inject_layers=4, glyph_inject_mode="xattn",
                glyph_embedder_depth=2, learn_sigma=False)
    base.update(kw)
    return DiT_2Cond_models["DiT-2Cond-Sp/2"](**base)

m0 = build(callig_style_attn=False)
m1 = build(callig_style_attn=True, callig_n_style=8)
print(f"[1] callig_style_attn=False -> {m0.callig_style_ca is None}")
print(f"    callig_style_attn=True  -> {m1.callig_style_ca is not None}")

ca = m1.callig_style_ca
nparam = sum(p.numel() for p in ca.parameters())
print(f"[2] cross-attn 参数: {nparam/1e3:.1f}K, n_style={ca.n_style}")
print(f"    out_proj zero-init: weight_abs_sum={ca.out_proj.weight.abs().sum().item():.2e} "
      f"bias_abs_sum={ca.out_proj.bias.abs().sum().item():.2e}")

m0.eval(); m1.eval()
x = torch.randn(2, 4, 32, 32)
t = torch.rand(2) * 1000
y_callig = torch.randint(0, 41, (2,))
g = torch.randn(2, 4, 32, 32)
with torch.no_grad():
    o0 = m0(x, t, y_callig, None, g=g)
    o1 = m1(x, t, y_callig, None, g=g)
print(f"[3] forward 输出形状: {tuple(o1.shape)}")
print(f"[4] 开/关输出差异 (zero-init 应≈0): {(o0-o1).abs().max().item():.2e}")

# 风格可控性: 同 g 不同 callig, 输出应不同 (zero-init 时为 0, 训后应 >0)
y2 = torch.randint(0, 41, (2,))
with torch.no_grad():
    o2 = m1(x, t, y2, None, g=g)
print(f"[5] 同 g 换书家输出差异 (zero-init 应为 0): {(o1-o2).abs().max().item():.2e}")

# resume 兼容
ck = sorted(glob.glob("assets/results/v10b_stdskel_fame3_c41x_cos/20260909-123903-*/checkpoints/*.pt"),
            key=lambda p: int(os.path.basename(p).split(".")[0]))[-1]
sd_full = torch.load(ck, map_location="cpu", weights_only=False)
sd = sd_full.get("ema") or sd_full.get("model") or sd_full
sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
miss, unexp = m1.load_state_dict(sd, strict=False)
ca_miss = [k for k in miss if "callig_style_ca" in k]
print(f"[6] resume {os.path.basename(ck)}: missing={len(miss)} (callig_style_ca={len(ca_miss)})")
print(f"    例: {ca_miss[:3]}")
print("SMOKE OK")