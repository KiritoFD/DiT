"""全开关端到端冒烟测试 (v12+)

覆盖: 模型尺寸 XS/S/M, condition_fusion {add,cat}, glyph_vec_cond,
      glyph_inject_mode {adaln,xattn}, 12ch(aux 通道) + CFG 作用域正确性。

每个组合都检查:
  1. forward 成功且形状正确
  2. 有梯度 (跑几步优化)
  3. **0 个无梯度参数** (DDP 安全, 否则 DDP 直接报错)
  4. forward_with_cfg 形状正确
  5. 12ch 时: CFG 只改前 image_channels 个通道, aux 通道必须逐位相同
"""
import sys, torch
sys.path.insert(0, "/root/Workspace/xy/DiT")
from src.model.dit import DiT_2Cond_models

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
print("device:", DEV, "torch:", torch.__version__)

BASE = dict(
    input_size=32, num_calligraphers=52, callig_embed_dim=128,
    use_char_cond=False, use_glyph_cond=True,
    glyph_scale_init=0.6, glyph_embedder_depth=2,
    norm_type="rms", mlp_type="swiglu", qk_norm=True, rope=True, attn_impl="sdpa",
)

FAILS = []


def check(tag, cond, msg):
    if not cond:
        FAILS.append(f"{tag}: {msg}")
        print(f"    ✗ {msg}")


def run(tag, model="DiT-2Cond-S/2", in_ch=4, img_ch=4, aux_groups=0, **over):
    kw = dict(BASE)
    kw.update(over)
    kw["in_channels"] = in_ch
    kw["image_channels"] = img_ch
    kw["learn_sigma"] = False   # flow 训练的真实取值 (train.py 对 flow 自动置 False)
    print(f"\n{'='*68}\n{tag}")
    print(f"  model={model} in_channels={in_ch} image_channels={img_ch} aux_groups={aux_groups}")
    m = DiT_2Cond_models[model](**kw).to(DEV)
    n = sum(p.numel() for p in m.parameters())
    ntr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  params={n/1e6:.2f}M (trainable {ntr/1e6:.2f}M)  out_ch={m.out_channels}")

    B = 2
    x = torch.randn(B, in_ch, 32, 32, device=DEV)
    t = torch.rand(B, device=DEV)
    yc = torch.randint(0, 52, (B,), device=DEV)
    ych = torch.randint(0, 100, (B,), device=DEV)
    g = torch.randn(B, 4, 32, 32, device=DEV)

    m.train()
    try:
        o = m(x, t, yc, ych, g=g)
    except Exception as e:
        check(tag, False, f"forward 崩溃: {type(e).__name__}: {e}")
        return None
    check(tag, tuple(o.shape) == (B, m.out_channels, 32, 32), f"forward 形状 {tuple(o.shape)} 期望 {(B, m.out_channels, 32, 32)}")

    # 梯度 + DDP 安全检查
    tgt = torch.randn_like(o)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    for _ in range(4):
        opt.zero_grad()
        torch.nn.functional.mse_loss(m(x, t, yc, ych, g=g), tgt).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    missing = [n_ for n_, p in m.named_parameters() if p.requires_grad and p.grad is None]
    check(tag, not missing, f"{len(missing)} 个无梯度参数 (DDP 会报错): {missing[:4]}")

    # CFG 正确性
    m.eval()
    with torch.no_grad():
        cond_only = m(x, t, yc, ych, g=g)
        cfg_out = m.forward_with_cfg(x, t, yc, ych, cfg_scale=0.7, g=g)
    check(tag, tuple(cfg_out.shape) == (B, m.out_channels, 32, 32), f"cfg 形状 {tuple(cfg_out.shape)}")
    d_img = (cfg_out[:, :img_ch] - cond_only[:, :img_ch]).abs().max().item()
    d_aux = (cfg_out[:, img_ch:] - cond_only[:, img_ch:]).abs().max().item() if in_ch > img_ch else 0.0
    check(tag, d_img > 0, f"CFG 没改变图像通道 (d_img={d_img:.2e})")
    check(tag, d_aux < 1e-5, f"CFG 污染了 aux 通道 (d_aux={d_aux:.2e}, 应 <1e-5)")
    print(f"  CFG: d_img={d_img:.4e} (>0)   d_aux={d_aux:.2e} (<1e-5)")
    return n


print("#" * 68)
print("# 1. 模型尺寸")
print("#" * 68)
sizes = {}
for name in ["DiT-2Cond-XS/2", "DiT-2Cond-S/2", "DiT-2Cond-M/2"]:
    sizes[name] = run(name, model=name, glyph_inject_layers=4, condition_fusion="factorized_add")

print("\n" + "#" * 68)
print("# 2. 条件融合 × glyph_vec_cond")
print("#" * 68)
for fusion in ["factorized_add", "factorized_cat"]:
    for gv in [False, True]:
        run(f"{fusion} glyph_vec={gv}", model="DiT-2Cond-XS/2", glyph_inject_layers=4,
            condition_fusion=fusion, glyph_vec_cond=gv)

print("\n" + "#" * 68)
print("# 3. glyph 注入方式")
print("#" * 68)
for mode in ["adaln", "xattn"]:
    for layers in [4, 12]:
        run(f"inject={mode} layers={layers}", model="DiT-2Cond-XS/2",
            condition_fusion="factorized_cat", glyph_vec_cond=True,
            glyph_inject_mode=mode, glyph_inject_layers=layers)

print("\n" + "#" * 68)
print("# 4. 12ch 辅助目标 (in=12, image_channels=4, 2 组 aux)")
print("#" * 68)
run("12ch + cat + xattn", model="DiT-2Cond-XS/2", in_ch=12, img_ch=4, aux_groups=2,
    condition_fusion="factorized_cat", glyph_vec_cond=True,
    glyph_inject_mode="xattn", glyph_inject_layers=4)
run("12ch + add + adaln", model="DiT-2Cond-XS/2", in_ch=12, img_ch=4, aux_groups=2,
    condition_fusion="factorized_add", glyph_vec_cond=False,
    glyph_inject_mode="adaln", glyph_inject_layers=4)

print("\n" + "#" * 68)
print(f"参数规模: " + "  ".join(f"{k.split('-')[-1]}={v/1e6:.2f}M" for k, v in sizes.items()))
if FAILS:
    print(f"\n!!! {len(FAILS)} 项失败:")
    for f in FAILS:
        print("   -", f)
    sys.exit(1)
print("\nALL SWITCH SMOKE TESTS PASSED")
