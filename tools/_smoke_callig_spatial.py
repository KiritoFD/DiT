"""冒烟验证 callig_spatial 开关: 构建/forward/resume 兼容/zero-init."""
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

# 1. 关闭/开启开关
m0 = build(callig_spatial=False)
m1 = build(callig_spatial=True)
print(f"[1] callig_spatial=False -> has net: {m0.callig_spatial_net is not None}")
print(f"    callig_spatial=True  -> has net: {m1.callig_spatial_net is not None}")

# 2. zero-init 检查
if m1.callig_spatial_net is not None:
    last = m1.callig_spatial_net[-1]
    print(f"[2] to_spatial zero-init: weight_abs_sum={last.weight.abs().sum().item():.2e} "
          f"bias_abs_sum={last.bias.abs().sum().item():.2e}")

# 3. forward 形状
m1.eval()
x = torch.randn(2, 4, 32, 32)
t = torch.rand(2) * 1000
y_callig = torch.randint(0, 41, (2,))
g = torch.randn(2, 4, 32, 32)
with torch.no_grad():
    o = m1(x, t, y_callig, None, g=g)
print(f"[3] forward 输出形状: {tuple(o.shape)} (期望 (2,4,32,32))")

# 4. callig_spatial 是否真的改变 g_tok (对比开/关)
m0.eval()
with torch.no_grad():
    o0 = m0(x, t, y_callig, None, g=g)
    o1 = m1(x, t, y_callig, None, g=g)
print(f"[4] 开/关输出差异 (zero-init 时应≈0): {(o0-o1).abs().max().item():.2e}")

# 5. resume 兼容: 加载现有 c41x_cos ckpt (无 callig_spatial_net)
ck = sorted(glob.glob("assets/results/v10b_stdskel_fame3_c41x_cos/20260909-123903-*/checkpoints/*.pt"),
            key=lambda p: int(os.path.basename(p).split(".")[0]))[-1]
sd_full = torch.load(ck, map_location="cpu", weights_only=False)
sd = sd_full.get("ema") or sd_full.get("model") or sd_full
sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
miss, unexp = m1.load_state_dict(sd, strict=False)
spatial_miss = [k for k in miss if "callig_spatial" in k]
print(f"[5] resume ckpt {os.path.basename(ck)}: missing={len(miss)} unexp={len(unexp)}")
print(f"    callig_spatial 相关 missing keys = {len(spatial_miss)} (应为全部新增)") 
print(f"    例: {spatial_miss[:3]}")
print("SMOKE OK")