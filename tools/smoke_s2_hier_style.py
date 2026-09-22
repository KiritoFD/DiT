# -*- coding: utf-8 -*-
"""smoke_s2_hier_style.py — S2（三层语义分解 + 局部风格-骨架引导）冒烟验证。

CPU 可跑，不需要数据/ckpt。逐条验证 docs/922/50_implementation.md 的硬约束。

## ⚠ 为什么必须 de-zero

adaLN-Zero 的 DiT 在**初始化时输出恒等于 0**：
  * ``block.adaLN_modulation[-1]`` 权重/bias 全 0 -> 每个 block 的 gate=0 -> 恒等
  * ``final_layer.adaLN_modulation[-1]`` + ``final_layer.linear`` 全 0 -> 输出 0

后果：**任何"改变条件应该改变输出"的测试在 step 0 都必然得到 Δ=0**，
看起来像"新通路没接上"，其实是初始化特性（实测踩到，排查了很久）。

所以本脚本先用 :func:`dezero` 把这两处推开（**对 base / S2 用同一份权重**），
再跑行为测试。等价性测试（T1）同样必须 de-zero 之后做才有意义。

用法:
    python tools/smoke_s2_hier_style.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import DiT_2Cond_models  # noqa: E402

torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
DEV = "cpu"
B, C, H = 2, 4, 32
NC, NP, NS = 45, 87, 8          # 书家 / pair / 书体
MODEL = "DiT-2Cond-XS6/2"       # depth 6，CPU 冒烟够用
FAIL = []


def check(name, ok, extra=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{('  ' + extra) if extra else ''}",
          flush=True)
    if not ok:
        FAIL.append(name)


def common(**kw):
    return dict(
        input_size=32, in_channels=C, num_calligraphers=NC, num_characters=1000,
        condition_fusion="factorized_cat", callig_embed_dim=128,
        glyph_vec_cond=True, glyph_vec_dim=128,
        use_glyph_cond=True, use_char_cond=False,
        glyph_inject_mode="adaln", use_checkpoint=False, learn_sigma=False,
        **kw)


def dezero(m, std=0.05):
    """把 adaLN-Zero 的零初始化推开，让条件真的能影响输出。"""
    with torch.no_grad():
        for blk in m.blocks:
            blk.adaLN_modulation[-1].weight.normal_(std=std)
            blk.adaLN_modulation[-1].bias.normal_(std=std)
        fl = m.final_layer
        fl.adaLN_modulation[-1].weight.normal_(std=std)
        fl.adaLN_modulation[-1].bias.normal_(std=std)
        fl.linear.weight.normal_(std=std)
        fl.linear.bias.normal_(std=std)
    return m


def inputs(seed=1, pair_shift=0, script_shift=0):
    torch.manual_seed(seed)
    return dict(
        x=torch.randn(B, C, H, H), t=torch.rand(B),
        y_callig=torch.arange(B, dtype=torch.long) + pair_shift,
        y_char=torch.zeros(B, dtype=torch.long),
        g=torch.randn(B, C, H, H),
        y_callig_raw=torch.arange(B, dtype=torch.long),
        y_pair=(torch.arange(B, dtype=torch.long) + pair_shift) % NP,
        y_script=(torch.arange(B, dtype=torch.long) + script_shift) % 3,
    )


def dmax(a, b):
    return (a - b).abs().max().item()


# ── 构造 base（旧路径）与 S2（全通路）────────────────────────────────────────
print("== 构造模型 ==", flush=True)
torch.manual_seed(0)
base = dezero(DiT_2Cond_models[MODEL](**common(
    glyph_inject_layers=4, glyph_drop_prob=0.0)).eval())

s2 = DiT_2Cond_models[MODEL](**common(
    glyph_inject_layers=4, glyph_drop_prob=0.0,
    hier_style=1, num_pairs=NP, num_scripts=NS, script_embed_dim=64,
    script_film=True, spatial_film_rank=64,
    local_ca_layers=2, local_ca_at="2,6", local_ca_q="g",
    local_ca_heads=4, local_ca_rank=64))
_missing, _unexpected = s2.load_state_dict(base.state_dict(), strict=False)
# 让 s2 的 adaLN/final_layer 与 base **完全一致**（否则 T1 等价性无从谈起）
dezero(s2)
s2.load_state_dict(base.state_dict(), strict=False)
s2 = s2.eval()

_new_prefix = ("style_hier.", "script_film.", "spatial_film.", "local_ca.")
check("T1a 共享权重全部命中（missing 只应是 S2 新模块）",
      all(k.startswith(_new_prefix) for k in _missing),
      f"missing={len(_missing)} unexpected={len(_unexpected)}")
check("T1b 无 unexpected key（不破坏旧 ckpt 兼容）", len(_unexpected) == 0,
      f"unexpected={list(_unexpected)[:3]}")

with torch.no_grad():
    _in = inputs()
    o_base = base(**_in)
    o_s2 = s2(**_in)
check("T1c 非退化前向（de-zero 后条件真的影响输出）",
      o_base.abs().max().item() > 1e-6, f"max|out|={o_base.abs().max().item():.3e}")
d = dmax(o_base, o_s2)
check("T1d zero-init => 与旧路径逐位相等（可安全 resume）", d < 1e-6,
      f"max|Δ|={d:.3e}")

# ── T2 形状 / CFG 路径 ───────────────────────────────────────────────────────
for name, fn in [
    ("T2a forward_with_cfg", lambda: s2.forward_with_cfg(**inputs(), cfg_scale=0.7)),
    ("T2b forward_with_2axis_cfg",
     lambda: s2.forward_with_2axis_cfg(**inputs(), cfg_callig=2.0, cfg_glyph=1.0)),
]:
    try:
        with torch.no_grad():
            _o = fn()
        check(f"{name} 跑通", tuple(_o.shape) == (B, C, H, H), str(tuple(_o.shape)))
    except Exception as e:  # noqa: BLE001
        check(f"{name} 跑通", False, repr(e))

s2x = DiT_2Cond_models[MODEL](**common(
    glyph_inject_layers=4, glyph_drop_prob=0.0,
    hier_style=1, num_pairs=NP, num_scripts=NS, script_film=True,
    local_ca_layers=2, local_ca_at="2,6", local_ca_q="x", local_ca_window=5))
dezero(s2x)
try:
    with torch.no_grad():
        ox = s2x.eval()(**inputs())
    check("T2c local_ca_q='x' + window=5 跑通", tuple(ox.shape) == (B, C, H, H))
except Exception as e:  # noqa: BLE001
    check("T2c local_ca_q='x' + window=5 跑通", False, repr(e))

# ── T3 梯度（注意 zero-init 的"一步延迟"）───────────────────────────────────
s2.zero_grad(set_to_none=True)
s2(**inputs()).sum().backward()
_g = {n: float(p.grad.abs().sum()) for n, p in s2.named_parameters()
      if p.grad is not None}
check("T3a E_pair 从 step 0 就有梯度（α=0.1≠0）",
      _g.get("style_hier.E_pair.weight", 0) > 0,
      f"|g|={_g.get('style_hier.E_pair.weight', 0):.3e}")
check("T3b script_film 从 step 0 就有梯度（∂out/∂β=1）",
      _g.get("script_film.film.weight", 0) > 0,
      f"|g|={_g.get('script_film.film.weight', 0):.3e}")
check("T3c spatial_film 从 step 0 就有梯度",
      _g.get("spatial_film.net.2.weight", 0) > 0,
      f"|g|={_g.get('spatial_film.net.2.weight', 0):.3e}")
check("T3d local_ca.out_proj 从 step 0 就有梯度",
      _g.get("local_ca.0.out_proj.weight", 0) > 0,
      f"|g|={_g.get('local_ca.0.out_proj.weight', 0):.3e}")
# α 与 style_qk 的梯度依赖"前一步先把 E_pair / out_proj 推开" -> 手动模拟第 1 步
with torch.no_grad():
    s2.style_hier.E_pair.weight.normal_(std=0.05)
    for _lc in s2.local_ca:
        _lc.out_proj.weight.normal_(std=0.05)
s2.zero_grad(set_to_none=True)
s2(**inputs()).sum().backward()
_g2 = {n: float(p.grad.abs().sum()) for n, p in s2.named_parameters()
       if p.grad is not None}
check("T3e α 在 E_pair 非零后有梯度（一步延迟，非死锁）",
      _g2.get("style_hier.alpha", 0) > 0, f"|g|={_g2.get('style_hier.alpha', 0):.3e}")
check("T3f style_qk 在 out_proj 非零后有梯度（一步延迟）",
      _g2.get("local_ca.0.style_qk.2.weight", 0) > 0,
      f"|g|={_g2.get('local_ca.0.style_qk.2.weight', 0):.3e}")

# ── T4 三条通路各自"能改变输出"──────────────────────────────────────────────
with torch.no_grad():
    s2.script_film.film.weight.normal_(std=0.05)
    s2.script_film.film.bias.normal_(std=0.05)
    s2.spatial_film.net[-1].weight.normal_(std=0.05)
    s2.spatial_film.net[-1].bias.normal_(std=0.05)
    s2.style_hier.E_pair.weight.normal_(std=0.3)
    _a = s2(**inputs(pair_shift=0))
    _b = s2(**inputs(pair_shift=1))
check("T4a 换 pair_id 改变输出（pair 残差生效）", dmax(_a, _b) > 1e-6,
      f"max|Δ|={dmax(_a, _b):.3e}")
with torch.no_grad():
    _c0 = s2(**inputs(script_shift=0))
    _c1 = s2(**inputs(script_shift=1))
check("T4b 换 script_id 改变输出（书体 FiLM 生效）", dmax(_c0, _c1) > 1e-6,
      f"max|Δ|={dmax(_c0, _c1):.3e}")
with torch.no_grad():
    _i = inputs()
    _d_full = s2(**_i)
    _i2 = dict(_i, g=torch.zeros_like(_i["g"]))
    _d_zero = s2(**_i2)
check("T4c g=0 改变输出（骨架通路生效）", dmax(_d_full, _d_zero) > 1e-6,
      f"max|Δ|={dmax(_d_full, _d_zero):.3e}")

# ── T5 drop 掩码纯净性（最容易踩的坑）───────────────────────────────────────
s2d = DiT_2Cond_models[MODEL](**common(
    glyph_inject_layers=0, glyph_drop_prob=0.0,
    hier_style=1, num_pairs=NP, num_scripts=NS, script_film=True,
    spatial_film_rank=64, local_ca_layers=2, local_ca_at="2,6", local_ca_q="g"))
dezero(s2d)
s2d.eval()
with torch.no_grad():
    s2d.script_film.film.weight.normal_(std=0.2)
    s2d.script_film.film.bias.normal_(std=0.2)
    s2d.spatial_film.net[-1].weight.normal_(std=0.2)
    s2d.spatial_film.net[-1].bias.normal_(std=0.2)
    for _lc in s2d.local_ca:
        _lc.out_proj.weight.normal_(std=0.2)
        _lc.out_proj.bias.normal_(std=0.2)
    _gt0 = s2d.glyph_embedder(torch.zeros(B, C, H, H)).flatten(2).transpose(1, 2)
    _zero_keep = torch.zeros(B)
    _es = s2d.style_hier(
        s2d.y_callig_embedder(torch.arange(B), False),
        torch.arange(B), torch.zeros(B, dtype=torch.long))[1]
    _cond = torch.randn(B, s2d.cond_dim)
    _g1 = s2d.script_film(_gt0, _es, _zero_keep)
    _g2 = s2d.spatial_film(_g1, _cond, s2d.local_pos, _zero_keep)
    for _lc in s2d.local_ca:
        _g2 = _lc(_g2, _g2, _cond, s2d.local_pos, _zero_keep)
check("T5a drop-g 时 g_tok 恒为 0（β 乘了 keep，不污染 uncond 分支）",
      _g2.abs().max().item() < 1e-6, f"max|g_tok|={_g2.abs().max().item():.3e}")
with torch.no_grad():
    _g1f = s2d.script_film(_gt0, _es, None)
    _g2f = s2d.spatial_film(_g1f, _cond, s2d.local_pos, None)
check("T5b keep=None 时 g_tok 非零（确认 T5a 不是假阳性）",
      _g2f.abs().max().item() > 1e-6, f"max|g_tok|={_g2f.abs().max().item():.3e}")

# ── T6 局部性（不能退化成全局调制）──────────────────────────────────────────
with torch.no_grad():
    # ⚠ batch 必须与 _cond 一致（B=2）—— 这里踩过一次：
    #   用 1 个 token 配 2 个 cond，会在 gq.view(B,H,hd) 报"shape 无效"，
    #   看起来像维度配错，实际是 batch 不一致。
    _tok = torch.randn(B, 256, 384)
    _o_lc = s2d.local_ca[0](_tok, _tok, _cond, s2d.local_pos, None)
    _per_tok_lc = (_o_lc - _tok)[0].abs().sum(-1)
    _o_sf = s2d.spatial_film(_tok, _cond, s2d.local_pos, None)
    _per_tok_sf = (_o_sf - _tok)[0].abs().sum(-1)
check("T6a local_ca 逐 token 不同（未退化成全局）",
      float(_per_tok_lc.std()) > 1e-5, f"std={float(_per_tok_lc.std()):.3e}")
check("T6b spatial_film 逐 token 不同（未退化成全局）",
      float(_per_tok_sf.std()) > 1e-5, f"std={float(_per_tok_sf.std()):.3e}")

# ── T7 参数量 ────────────────────────────────────────────────────────────────
n_base = sum(p.numel() for p in base.parameters())
n_new = sum(p.numel() for k, p in s2.named_parameters()
            if k.startswith(_new_prefix))
print(f"\n参数量: base={n_base/1e6:.4f}M  s2={n_base/1e6 + n_new/1e6:.4f}M  "
      f"新增={n_new/1e3:.1f}K ({n_new/n_base*100:.2f}%)")

print("\n" + "=" * 60)
if FAIL:
    print(f"❌ {len(FAIL)} 项失败: {FAIL}")
    sys.exit(1)
print("✅ 全部通过")
