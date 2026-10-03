"""判定性冒烟: callig_style_attn 的 drop 语义修复.

注意: fresh DiT 的 final_layer.linear 零初始化 -> 输出恒 0, 直接比输出是空验证。
先给 final 层灌非零权重, 使输出携带 x/g/c 的信息。

T1: glyph 全 drop + style out_proj 非零 -> g 路径贡献必须精确为 0 (与 g=None 同 x 输出相同)
T2: glyph 不 drop -> g 路径贡献非零 (对照)
T3: callig 全 drop -> 换 y_callig 输出必须完全不变 (adaLN 与 style 共用同一 drop mask)
T4: callig 不 drop -> 换 y_callig 输出必须变化 (风格生效, 对照)
"""
import sys, torch
import torch.nn as nn
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "/root/Workspace/xy/DiT")
from src.model import DiT_2Cond_models

def build(**kw):
    base = dict(num_calligraphers=41, num_characters=35130,
                condition_fusion="factorized_add", callig_embed_dim=128,
                char_embed_dim=384, char_proj_mode="mlp", callig_proj_mode="mlp",
                use_glyph_cond=True, use_char_cond=False,
                glyph_inject_layers=4, glyph_inject_mode="xattn",
                glyph_embedder_depth=2, learn_sigma=False,
                glyph_drop_prob=0.0, cond_drop_all_prob=0.0, cond_drop_one_prob=0.0,
                cond_drop_which_glyph_prob=0.5)
    base.update(kw)
    torch.manual_seed(0)
    m = DiT_2Cond_models["DiT-2Cond-Sp/2"](**base)
    # fresh 模型输出恒 0 (final_layer 零初始化) -> 灌非零权重让输出携带信息
    with torch.no_grad():
        nn.init.normal_(m.final_layer.linear.weight, std=0.02)
        nn.init.normal_(m.final_layer.linear.bias, std=0.02)
        for blk in m.blocks:
            nn.init.normal_(blk.adaLN_modulation[-1].weight, std=0.02)
            nn.init.normal_(blk.adaLN_modulation[-1].bias, std=0.02)
    return m

def fwd(m, y, g, x=None, t=None):
    if x is None:
        x = torch.randn(2, 4, 32, 32)
        t = torch.rand(2) * 1000
    with torch.no_grad():
        return m(x, t, y, None, g=g)

m = build(callig_style_attn=True)
ca = m.callig_style_ca
with torch.no_grad():
    ca.out_proj.weight.normal_(0, 0.05)   # 模拟训后非零注入
    ca.out_proj.bias.normal_(0, 0.05)
m.train()

x = torch.randn(2, 4, 32, 32)
t = torch.rand(2) * 1000
g = torch.randn(2, 4, 32, 32)
y = torch.tensor([3, 17])

# T1: glyph 全 drop
m.glyph_drop_prob = 1.0
d = (fwd(m, y, g, x, t) - fwd(m, y, None, x, t)).abs().max().item()
assert d == 0.0, f"T1 FAIL: drop 后 g 贡献非零 ({d:.3e})"
print(f"[T1] glyph 全 drop -> g 贡献精确为 0 (diff={d:.2e})  OK")

# T2: glyph 不 drop
m.glyph_drop_prob = 0.0
d = (fwd(m, y, g, x, t) - fwd(m, y, None, x, t)).abs().max().item()
assert d > 1e-5, f"T2 FAIL: g 应有贡献 ({d:.3e})"
print(f"[T2] glyph 保留 -> g 有贡献 (diff={d:.2e})  OK")

# T3: callig 全 drop -> 换书家输出不变
m.cond_drop_all_prob = 1.0
d = (fwd(m, y, g, x, t) - fwd(m, torch.tensor([5, 9]), g, x, t)).abs().max().item()
assert d == 0.0, f"T3 FAIL: drop-callig 分支泄漏风格 ({d:.3e})"
print(f"[T3] callig 全 drop -> 换 y_callig 输出不变 (diff={d:.2e})  OK")

# T4: callig 不 drop -> 换书家输出变
m.cond_drop_all_prob = 0.0
d = (fwd(m, y, g, x, t) - fwd(m, torch.tensor([5, 9]), g, x, t)).abs().max().item()
assert d > 1e-5, f"T4 FAIL: 风格未生效 ({d:.3e})"
print(f"[T4] callig 保留 -> 换 y_callig 输出变 (diff={d:.2e})  OK")

nparam = sum(p.numel() for p in ca.parameters())
print(f"[info] callig_style_ca 参数: {nparam/1e6:.2f}M, n_style={ca.n_style}")
print("SMOKE OK")
