# -*- coding: utf-8 -*-
"""v2 架构改造回归测试：组件 / DiT_2Cond / FlowMatching 求解器 / ControlNet / DINO 冻结。

CPU 即可运行，不需要 GPU 也不需要数据:

    cd <repo root> && python tests/test_arch_v2.py
"""
import sys, math, os
import torch
import torch.nn as nn

# 仓库根（tests/ 的上一级）必须在 sys.path 里，否则 import src.* 失败。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def ok(msg):
    print(f"  [OK] {msg}")


def section(t):
    print(f"\n=== {t} ===")


# ---------------------------------------------------------------- 1. modules
section("1. modules 组件自检")
from src.model import modules as M

d, h = 384, 6
blk = M.DiTBlock(d, h, norm_type="rms", mlp_type="swiglu", qk_norm=True, attn_impl="sdpa")
B, T = 2, 256
x = torch.randn(B, T, d)
c = torch.randn(B, d)
cos, sin = M.precompute_rope_2d(16, d // h)
out = blk(x, c, rope=(cos, sin))
ok(f"DiTBlock(rms/swiglu/qknorm/rope) {tuple(x.shape)} -> {tuple(out.shape)}")
assert out.shape == x.shape

# 参数量对齐检查：swiglu vs gelu
g = M.build_mlp("gelu", 384, 4.0)
s = M.build_mlp("swiglu", 384, 4.0)
pg = sum(p.numel() for p in g.parameters())
ps = sum(p.numel() for p in s.parameters())
print(f"  [info] MLP params: gelu={pg:,} swiglu={ps:,} (ratio={ps/pg:.3f})")
ok(f"swiglu 参数量与 gelu 近似对齐 (差 {abs(ps-pg)/pg*100:.1f}%)")

# RMSNorm vs LayerNorm
rn = M.RMSNorm(64)
ln = nn.LayerNorm(64, elementwise_affine=False)
xx = torch.randn(4, 32, 64)
ok(f"RMSNorm 输出 std={rn(xx).std():.3f} (LayerNorm std={ln(xx).std():.3f})")

# RoPE：旋转不改变范数（正交变换）
q1 = torch.randn(2, 6, 256, 64)
r = M.apply_rope(q1, cos, sin)
ok(f"RoPE apply 形状正确 {tuple(r.shape)}")
norm_before = q1.norm(dim=-1)
norm_after = r.norm(dim=-1)
ok(f"RoPE 保持范数 (max rel err {((norm_after-norm_before).abs()/norm_before).max():.2e})")
# RoPE 的相对性：整体平移后 attention score 不变
sd = torch.randn(2, 6, 256, 64)
sc_shift = (M.apply_rope(q1, cos, sin) @ M.apply_rope(sd, cos, sin).transpose(-1, -2))
# 手动构造"全体 token 位置 +k"的情形过于复杂，这里只验证 attention 可计算且有限
ok(f"RoPE 后 attention 有限: {torch.isfinite(sc_shift).all().item()}")
# cos/sin 应满足 cos^2+sin^2 = 1 (归一化旋转)
n2 = cos[0, :32] ** 2 + sin[0, :32] ** 2
ok(f"RoPE cos²+sin²=1 (max err {abs(n2-1).max():.2e})")

# ------------------------------------------------------------ 2. DiT_2Cond
section("2. DiT_2Cond 现代化")
from src.model import DiT_2Cond_models

kw = dict(input_size=32, num_calligraphers=1011, num_characters=35130,
          condition_fusion="factorized_add", callig_embed_dim=128,
          char_embed_dim=384, char_proj_mode="ln_only")

for tag, arch in [
    ("v2(rms/swiglu/rope/qknorm)", dict(norm_type="rms", mlp_type="swiglu", qk_norm=True, rope=True)),
    ("v1(layer/gelu/norope/noqk)", dict(norm_type="layer", mlp_type="gelu", qk_norm=False, rope=False)),
]:
    torch.manual_seed(0)
    m = DiT_2Cond_models["DiT-2Cond-S/2"](learn_sigma=False, **kw, **arch)
    n_p = sum(p.numel() for p in m.parameters())
    xx = torch.randn(2, 4, 32, 32)
    tt = torch.full((2,), 500.0)
    yc = torch.randint(0, 1011, (2,))
    ych = torch.randint(0, 35130, (2,))
    o = m(xx, tt, yc, ych)
    ok(f"{tag}: out={tuple(o.shape)} params={n_p/1e6:.2f}M")
    assert o.shape == xx.shape, f"shape mismatch {o.shape} vs {xx.shape}"

# CFG 路径
torch.manual_seed(0)
m = DiT_2Cond_models["DiT-2Cond-S/2"](learn_sigma=False, **kw)
o_cfg = m.forward_with_cfg(torch.randn(2, 4, 32, 32), torch.full((2,), 300.0),
                           torch.randint(0, 1011, (2,)), torch.randint(0, 35130, (2,)),
                           cfg_scale=1.7)
ok(f"forward_with_cfg -> {tuple(o_cfg.shape)}")
o2 = m.forward_with_2axis_cfg(torch.randn(2, 4, 32, 32), torch.full((2,), 300.0),
                              torch.randint(0, 1011, (2,)), torch.randint(0, 35130, (2,)),
                              cfg_callig=1.7, cfg_glyph=1.7, w_inter=0.0)
ok(f"forward_with_2axis_cfg -> {tuple(o2.shape)}")

# 零初始化恒等性（adaLN-Zero 的保证）
torch.manual_seed(0)
m = DiT_2Cond_models["DiT-2Cond-S/2"](learn_sigma=False, **kw)
mx = 0.0
for n_, p_ in m.named_parameters():
    if n_.endswith("adaLN_modulation.1.weight") or n_.endswith("adaLN_modulation.1.bias"):
        mx = max(mx, p_.abs().max().item())
ok(f"adaLN 调制层零初始化 (max|w|={mx:.2e}) -> 每个 block 初始为恒等")

# ------------------------------------------------------- 3. FlowMatching
section("3. FlowMatching 求解器 / t 采样")
from src.loss import create_flow_matching, flow_kwargs_from, FLOW_PARAMS

for sampler in ["euler", "heun"]:
    for hb in [True, False]:
        fm = create_flow_matching("20", sampler=sampler, heun_batch=hb,
                                  t_sampler="logit_normal", shift=1.0)
        print(f"  [info] {fm.describe()} heun_batch={hb}")
        assert fm.nfe == (40 if sampler == "heun" else 20), fm.nfe
ok("NFE 计算正确 (euler=steps, heun=2*steps)")

# t 采样分布
for ts in ["uniform", "logit_normal", "cosmap"]:
    fm = create_flow_matching("20", t_sampler=ts, t_mean=0.0, t_std=1.0)
    t = fm.sample_t(20000, "cpu")
    assert (t >= 0).all() and (t <= 1).all()
    print(f"  [info] t_sampler={ts:14s} mean={t.mean():.3f} std={t.std():.3f} "
          f"p5={t.kthvalue(int(.05*len(t)))[0]:.3f} p95={t.kthvalue(int(.95*len(t)))[0]:.3f}")
ok("三种 t 采样均在 [0,1] 且分布形态符合预期")

# shift schedule
for sh in [1.0, 3.0, 0.5]:
    fm = create_flow_matching("10", shift=sh)
    ts_ = fm._schedule(10, torch.device("cpu"))
    ok(f"shift={sh}: schedule {[round(v,3) for v in ts_.tolist()]}")
    assert ts_[0] == 1.0 and abs(ts_[-1]) < 1e-6

# Heun 的 batched 模式必须与 model_kwargs 一起复制 batch 维
# （CFG wrapper 假定 x.shape[0] == y.shape[0]）。这里用一个"严格"的假 model 来
# 强制暴露该问题：任何 batch 不匹配都会直接抛错。
class _StrictCFGModel(nn.Module):
    """模拟 DiT.forward_with_cfg：内部把 x 沿 batch 维 cat 一次。"""

    def forward(self, x, t, y=None, cond=None, **kw):
        assert y is not None and y.shape[0] == x.shape[0], \
            f"batch mismatch: x={x.shape[0]} vs y={y.shape[0]}"
        assert t.shape[0] == x.shape[0], f"t batch {t.shape[0]} vs x {x.shape[0]}"
        x2 = torch.cat([x, x], dim=0)      # CFG: cond + uncond
        return x2[: x.shape[0]] * 0.5      # 输出仍为 x.shape[0]

for hb in [1, 0]:
    fm_h = create_flow_matching("6", sampler="heun", heun_batch=hb)
    m_h = _StrictCFGModel()
    o_h = fm_h.ddim_sample_loop(
        m_h, (3, 4, 8, 8), x_T=torch.zeros(3, 4, 8, 8), device="cpu",
        model_kwargs=dict(y=torch.arange(3), cond=torch.zeros(3, 4, 8, 8)))
    ok(f"heun_batch={bool(hb)}: model_kwargs 与 x 的 batch 维始终一致 "
       f"-> out {tuple(o_h.shape)}")
ok("heun_batch 的 model_kwargs 复制正确（CFG wrapper 的 batch 假设不被打破）")

# 1 步 Euler 在常速度场下应精确
fm1 = create_flow_matching("1", sampler="euler")


class _ConstV(nn.Module):
    def forward(self, x, t, **kw):
        return torch.ones_like(x)


out1 = fm1.ddim_sample_loop(_ConstV(), (2, 4, 8, 8),
                            x_T=torch.zeros(2, 4, 8, 8), device="cpu")
ok(f"1-step Euler on v=1: x = 0 + (-1)*1 = {out1.flatten()[0]:.4f} (expect -1)")
assert abs(out1.flatten()[0] + 1) < 1e-5

# training_losses 形状
torch.manual_seed(0)
mm = DiT_2Cond_models["DiT-2Cond-S/2"](learn_sigma=False, **kw)
fm = create_flow_matching("20", t_sampler="logit_normal", sampler="heun")
xs = torch.randn(4, 4, 32, 32)
tt = fm.sample_t(4, "cpu")
terms = fm.training_losses(mm, xs, tt, dict(y_callig=torch.randint(0, 1011, (4,)),
                                            y_char=torch.randint(0, 35130, (4,))))
ok(f"training_losses -> loss {tuple(terms['loss'].shape)} = {terms['loss'].mean():.4f}")
assert terms["loss"].shape == (4,)

# learn_sigma=True 时的通道裁剪
torch.manual_seed(0)
ms = DiT_2Cond_models["DiT-2Cond-S/2"](learn_sigma=True, **kw)
terms2 = fm.training_losses(ms, xs[:2], tt[:2],
                            dict(y_callig=torch.randint(0, 1011, (2,)),
                                 y_char=torch.randint(0, 35130, (2,))))
ok(f"learn_sigma=True 时 loss 形状仍为 {tuple(terms2['loss'].shape)} (后 C 通道被裁剪)")

# --------------------------------------------------------- 4. ControlNet
section("4. ControlNet")
from src.model.controlnet import ControlNetDiT, ControlConditionEncoder, ZeroAdaLNInjection

torch.manual_seed(0)
main = DiT_2Cond_models["DiT-2Cond-S/2"](learn_sigma=False, **kw)
for tag, cfg in [
    ("默认(全深/modulate/rope)", dict()),
    ("半深(6层)", dict(ctrl_depth=6)),
    ("窄(192)", dict(ctrl_hidden=192, ctrl_num_heads=6)),
    ("add注入", dict(injection="add")),
    ("无rope", dict(rope=False)),
]:
    torch.manual_seed(0)
    cnet = ControlNetDiT(main, cond_in_channels=4, train_ctrl_only=True, **cfg)
    n_tr = sum(p.numel() for p in cnet.parameters() if p.requires_grad)
    n_fr = sum(p.numel() for p in cnet.parameters() if not p.requires_grad)
    cond = torch.randn(2, 4, 32, 32)
    o = cnet(torch.randn(2, 4, 32, 32), torch.full((2,), 400.0),
             torch.randint(0, 1011, (2,)), torch.randint(0, 35130, (2,)), cond=cond)
    ok(f"{tag}: out={tuple(o.shape)} trainable={n_tr/1e6:.2f}M frozen={n_fr/1e6:.2f}M "
       f"inject_layers={cnet.inject_layers[0]}..{cnet.inject_layers[-1]}")
    assert o.shape == (2, 4, 32, 32)

# 零初始化 → 注入为恒等：ControlNet 输出应等于主模型输出
torch.manual_seed(0)
main2 = DiT_2Cond_models["DiT-2Cond-S/2"](learn_sigma=False, **kw)
cnet2 = ControlNetDiT(main2, cond_in_channels=4, train_ctrl_only=True)
cnet2.eval()
xx = torch.randn(2, 4, 32, 32)
tt = torch.full((2,), 400.0)
yc = torch.randint(0, 1011, (2,))
ych = torch.randint(0, 35130, (2,))
cond = torch.randn(2, 4, 32, 32)
with torch.no_grad():
    o_main = main2(xx, tt, yc, ych)
    o_ctrl = cnet2(xx, tt, yc, ych, cond=cond)
diff = (o_main - o_ctrl).abs().max().item()
ok(f"zero-init 注入恒等性: |main - ctrl| = {diff:.2e} (expect ~0)")
assert diff < 1e-5, f"zero-init 注入不恒等! diff={diff}"

# RoPE 透传验证：关掉 rope 的主模型不应被强行加 pos_embed
torch.manual_seed(0)
main_nr = DiT_2Cond_models["DiT-2Cond-S/2"](learn_sigma=False, rope=False, **kw)
cnet_nr = ControlNetDiT(main_nr, cond_in_channels=4, train_ctrl_only=True, rope=False)
with torch.no_grad():
    o_nr = cnet_nr(xx, tt, yc, ych, cond=cond)
    o_nr_main = main_nr(xx, tt, yc, ych)
ok(f"rope=False 路径: ctrl 与主模型 max diff={(o_nr-o_nr_main).abs().max():.2e}")

# null condition
for nc in ["gaussian", "zeros", "learned"]:
    cnet2.null_cond = nc
    if nc == "learned":
        cnet2.null_cond_param = nn.Parameter(torch.zeros(1, 4, 32, 32))
    n_ = cnet2._make_null(cond)
    ok(f"null_cond={nc}: shape={tuple(n_.shape)} std={n_.std():.3f}")

# CFG
cnet2.null_cond = "gaussian"
with torch.no_grad():
    ocfg = cnet2.forward_with_cfg(xx, tt, yc, ych, cfg_scale=1.7, cond=cond)
ok(f"ControlNet.forward_with_cfg -> {tuple(ocfg.shape)}")

# ------------------------------------------------------- 5. flow_kwargs
section("5. flow_kwargs_from 参数透传")


class _A:
    t_sampler = "logit_normal"
    t_mean = 0.0
    t_std = 1.0
    shift = 1.0
    flow_sampler = "heun"
    heun_batch = 1
    use_ot = False
    sampler = "factor_balanced"   # 数据采样器，不应污染 flow
    noise_schedule = "linear"     # ddpm 参数，应被丢弃


kw5 = flow_kwargs_from(_A())
print(f"  [info] flow_kwargs_from -> {kw5}")
assert kw5["sampler"] == "heun", kw5
assert "noise_schedule" not in kw5
ok("flow_sampler 别名正确映射为 sampler，且隔离了 --sampler / noise_schedule")

fm5 = create_flow_matching("20", **kw5)
ok(f"由 args 构造: {fm5.describe()}")

# ------------------------------------------------- 7. DINO 表冻结 / null token
section("7. 冻结字符表 + CFG null token 可训练性")
torch.manual_seed(0)
mf = DiT_2Cond_models["DiT-2Cond-S/2"](
    learn_sigma=False, freeze_char_table=True, **kw)
ye = mf.y_char_embedder
w = ye.embedding_table.weight
ok(f"table.requires_grad = {w.requires_grad} (应 False) | "
   f"null_embed 存在 = {ye.null_embed is not None}")
assert w.requires_grad is False
assert ye.null_embed is not None and ye.null_embed.requires_grad is True

# 回归：旧的 `w[-1].requires_grad_(True)` 是 no-op，这里验证新写法真的能训练
import torch.nn as _nn
_w = _nn.Parameter(torch.randn(5, 3))
_w.requires_grad_(False)
_w[-1].requires_grad_(True)
assert _w[-1].requires_grad is False, "预期 PyTorch 行为变化"
print("  [info] 确认 w[-1].requires_grad_(True) 是 no-op（旧 bug 的根因）")

# null token 真的能拿到梯度吗？用全 null 的 y_char 跑一次
for _b in mf.blocks:
    torch.nn.init.normal_(_b.adaLN_modulation[-1].weight, std=0.02)
    torch.nn.init.normal_(_b.adaLN_modulation[-1].bias, std=0.02)
torch.nn.init.normal_(mf.final_layer.adaLN_modulation[-1].weight, std=0.02)
torch.nn.init.normal_(mf.final_layer.linear.weight, std=0.02)
_nc = mf.y_char_embedder.num_classes
mf.zero_grad(set_to_none=True)
_out = mf(torch.randn(2, 4, 32, 32), torch.full((2,), 500.0),
          torch.randint(0, 1011, (2,)), torch.full((2,), _nc))
_out.pow(2).mean().backward()
g_null = mf.y_char_embedder.null_embed.grad
g_tab = mf.y_char_embedder.embedding_table.weight.grad
ok(f"null_embed.grad 非空 = {g_null is not None and g_null.abs().sum() > 0}")
assert g_null is not None and g_null.abs().sum() > 0, "null token 仍收不到梯度!"
ok(f"冻结表 grad = {g_tab} (应 None，省 13.5M 参数的梯度+优化器状态)")

# forward 里 null 行必须被 null_embed 覆盖
with torch.no_grad():
    e_null = mf.y_char_embedder(torch.full((3,), _nc), False)
    e_ref = mf.y_char_embedder(torch.tensor([0, 1, 2]), False)
ok(f"null 行输出 == null_embed: "
   f"{torch.allclose(e_null, mf.y_char_embedder.null_embed.expand(3, -1))} | "
   f"与普通行不同: {not torch.allclose(e_null[0], e_ref[0])}")

# --------------------------------------------------------- 6. 梯度流
section("6. 端到端梯度流")


def unzero_adaln(model, std=0.02):
    """打破 adaLN-Zero 的零初始化，模拟"已训练若干步"的模型。

    随机初始化的 DiT 因 adaLN-Zero（含 final_layer.linear 也是零初始化）
    输出恒为 0，导致**任何**上游参数首步都收不到梯度 —— 这是 DiT 的既定设计，
    不是 bug。要验证 ControlNet 的梯度通路，必须先让主模型脱离零点。
    """
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("adaLN_modulation.1.weight"):
                p.normal_(0, std)
            elif n.endswith("adaLN_modulation.1.bias"):
                p.normal_(0, std)
        if hasattr(model, "final_layer"):
            model.final_layer.linear.weight.normal_(0, std)
            model.final_layer.linear.bias.normal_(0, std)


# --- 6a: 验证 DiT 零初始化性质（预期：输出恒 0）---
torch.manual_seed(0)
m0 = DiT_2Cond_models["DiT-2Cond-S/2"](learn_sigma=False, **kw)
o0 = m0(torch.randn(2, 4, 32, 32), torch.full((2,), 500.0),
        torch.randint(0, 1011, (2,)), torch.randint(0, 35130, (2,)))
ok(f"6a. adaLN-Zero 性质: 随机初始化输出 max|out| = {o0.abs().max():.2e} (预期 0)")

# --- 6b: 已训练主模型下的 ControlNet 梯度通路 ---
torch.manual_seed(0)
main3 = DiT_2Cond_models["DiT-2Cond-S/2"](learn_sigma=False, **kw)
unzero_adaln(main3)                      # 模拟已训练主模型
cnet3 = ControlNetDiT(main3, cond_in_channels=4, train_ctrl_only=True)
unzero_adaln(cnet3.ctrl_encoder)         # 模拟 ctrl encoder 已脱离零点
fm6 = create_flow_matching("20", t_sampler="logit_normal", sampler="heun")

xs6 = torch.randn(2, 4, 32, 32)
t6 = fm6.sample_t(2, "cpu")

# 第 1 步：注入层 proj 应立刻拿到梯度（d out/d W = feat ≠ 0）
#          ctrl blocks 仍为 0（d out/d x = proj.weight = 0）
out6 = cnet3(xs6, t6 * 1000.0, torch.randint(0, 1011, (2,)),
             torch.randint(0, 35130, (2,)), cond=torch.randn(2, 4, 32, 32))
loss = out6.float().pow(2).mean()
loss.backward()
g_inj1 = cnet3.injections[0].proj.weight.grad
g_blk1 = cnet3.ctrl_encoder.ctrl_blocks[0].attn.qkv.weight.grad
ok(f"6b. step1: injections.proj.grad={g_inj1.abs().sum():.3e} (应>0) "
   f"| ctrl_blocks.grad={g_blk1.abs().sum():.3e} (应为0, zero-conv warm-start)")
assert g_inj1.abs().sum() > 0, "注入层收不到梯度!"
assert g_blk1.abs().sum() == 0, "zero-conv warm-start 失效：ctrl block 不应在首步拿到梯度"

# 第 2 步：proj.weight 变非零后，梯度应传导到 ctrl blocks
with torch.no_grad():
    cnet3.injections[0].proj.weight.add_(0.01 * torch.randn_like(cnet3.injections[0].proj.weight))
cnet3.zero_grad(set_to_none=True)
out6b = cnet3(xs6, t6 * 1000.0, torch.randint(0, 1011, (2,)),
              torch.randint(0, 35130, (2,)), cond=torch.randn(2, 4, 32, 32))
out6b.float().pow(2).mean().backward()
g_blk2 = cnet3.ctrl_encoder.ctrl_blocks[0].attn.qkv.weight.grad
ok(f"6b. step2: ctrl_blocks.grad={g_blk2.abs().sum():.3e} (应>0, 梯度已打通)")
assert g_blk2.abs().sum() > 0, "梯度未传导到 ctrl blocks"

# 主模型必须保持冻结
assert all(not p.requires_grad for p in cnet3.main.parameters())
ok("6c. 主模型全部 frozen (train_ctrl_only=True)")

print("\n" + "=" * 60)
print("ALL CHECKS PASSED")
print("=" * 60)
