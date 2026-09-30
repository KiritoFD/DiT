#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/diag_glyph_wiring.py — 直查 g 是否真的影响输出 (CPU only)。

section [3] 的 "换 g 输出变化=0" 需要定位: 到底是探针伪影还是真实接线缺口。
本脚本直接打进 forward 内部, 打印每一段中间张量的幅度:
  A. glyph_embedder(g) 输出幅度  (g_tok 是否非零)
  B. _gate 是否为 None
  C. 输入层加法 x = x + glyph_scale*g_tok 之后的 g_tok 幅度
  D. 逐层注入器 proj 权重是否全零 (zero-init)
  E. 直接手工重放输入层加法, 确认 g 的敏感性
"""
import os, sys, json
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

CFG = "src/train/configs/v25_stdskel.json"
cfg = json.load(open(CFG, encoding="utf-8"))

from src.model.dit import DiT_2Cond

kw = dict(
    input_size=32, patch_size=2, in_channels=4, hidden_size=384, depth=12,
    num_heads=6, num_calligraphers=cfg["num_calligraphers"],
    callig_embed_dim=cfg["callig_embed_dim"],
    condition_fusion=cfg["condition_fusion"],
    cond_fusion_norm=cfg["cond_fusion_norm"],
    use_char_cond=not cfg["no_char_cond"],
    use_glyph_cond=(int(cfg.get("w_glyph_cond", 0) or 0) > 0 or bool(cfg["skel_as_glyph_cond"])),
    glyph_scale_init=float(cfg["glyph_scale_init"]),
    glyph_inject_layers=int(cfg["glyph_inject_layers"]),
    glyph_inject_mode=str(cfg["glyph_inject_mode"]),
    glyph_embedder_depth=int(cfg["glyph_embedder_depth"]),
    glyph_drop_prob=float(cfg["glyph_drop_prob"]),
    deform_skel=int(cfg["deform_skel"]),
    deform_ckpt=str(cfg["deform_ckpt"]),
)
model = DiT_2Cond(**kw).eval()

print("=" * 78)
print("[A] glyph_embedder(g) 输出幅度")
print("=" * 78)
g1 = torch.randn(2, 4, 32, 32)
g2 = torch.randn(2, 4, 32, 32)
with torch.no_grad():
    t1 = model.glyph_embedder(g1)
    t2 = model.glyph_embedder(g2)
print("  glyph_embedder type      :", type(model.glyph_embedder).__name__)
print("  |g_tok(g1)| mean/min/max : %.6f / %.6f / %.6f" % (
    t1.abs().mean(), t1.abs().min(), t1.abs().max()))
print("  |g_tok(g2)| mean         : %.6f" % t2.abs().mean())
print("  |t1 - t2| mean           : %.6f   (<- g 编码器确实对 g 敏感)" % (t1 - t2).abs().mean())

print()
print("=" * 78)
print("[B] _gate 状态")
print("=" * 78)
print("  glyph_gate_t    =", model.glyph_gate_t)
print("  glyph_gate_floor=", model.glyph_gate_floor)
print("  glyph_scale     =", float(model.glyph_scale))
print("  glyph_concat_input =", model.glyph_concat_input)
print("  -> _gate 在 t 全为 0.5 时 = %s (glyph_gate_t=0 -> None)" % (
    "None" if model.glyph_gate_t <= 0 else "非 None"))

print()
print("=" * 78)
print("[C] 逐层注入器 zero-init 状态")
print("=" * 78)
print("  glyph_inject_at =", getattr(model, "glyph_inject_at", None))
print("  glyph_injections is None:", model.glyph_injections is None)
if model.glyph_injections is not None:
    for k, inj in enumerate(model.glyph_injections):
        proj = getattr(inj, "proj", None)
        if proj is not None:
            print("   inj[%d].proj.weight |sum|=%.6e  |bias|sum=%.6e" % (
                k, proj.weight.abs().sum().item(), proj.bias.abs().sum().item()))
        else:
            cand = [n for n, _ in inj.named_parameters()]
            print("   inj[%d] params: %s" % (k, cand))

print()
print("=" * 78)
print("[D] 端到端: 换 g 是否改输出 (eval / no_grad)")
print("=" * 78)
B = 2
x = torch.randn(B, 4, 32, 32)
t = torch.full((B,), 0.5)
y_callig = torch.zeros(B, dtype=torch.long)
y_char = torch.zeros(B, dtype=torch.long)
with torch.no_grad():
    o1 = model(x, t, y_callig, y_char, g=g1)
    o2 = model(x, t, y_callig, y_char, g=g2)
o1 = o1[0] if isinstance(o1, (tuple, list)) else o1
o2 = o2[0] if isinstance(o2, (tuple, list)) else o2
print("  out shape =", tuple(o1.shape))
print("  |out1-out2| mean = %.6e   (<- 若为 0 才是真问题)" % (o1 - o2).abs().mean().item())

print()
print("=" * 78)
print("[E] 手工重放输入层加法, 验证梯度直通")
print("=" * 78)
# 反向: g 是否拿到梯度 (输入层加法 glyph_scale 非零 -> 应该有)
gg = torch.randn(B, 4, 32, 32, requires_grad=True)
out = model(x, t, y_callig, y_char, g=gg)
out = out[0] if isinstance(out, (tuple, list)) else out
out.sum().backward()
print("  d(out)/d(g) 非零比例 = %.4f  |grad| mean = %.6e" % (
    (gg.grad.abs() > 1e-12).float().mean().item(), gg.grad.abs().mean().item()))
print("  -> 非零说明 g 至少通过输入层加法拿到了梯度种子")

print()
print("=" * 78)
print("定位: 若 [D] 为 0 但 [E] 非零 -> 是 eval 下某个 mask/branch 把 g_tok 清零;")
print("      若 [D] 与 [E] 都非零 -> section[3] 是探针伪影 (no_grad/顺序问题)。")
print("=" * 78)
