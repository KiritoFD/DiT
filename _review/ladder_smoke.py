"""容量探针阶梯 (doc63 §1/§2) 的**可启动性**验证。

对每个配置: 按 `src/train/train.py:242-303` 的同一映射构造模型 ->
报 depth/h/params/FLOPs 代理 -> 跑 fwd/bwd (g 与 g=None 两条) ->
断言**所有可训练参数都拿到梯度** (DDP 硬要求, 见 doc09 §6 "unused params -> DDP will error")。

用法: /opt/conda/envs/cu121/bin/python _review/ladder_smoke.py
"""
import io
import json
import os
import sys

import torch

REPO = os.environ.get("DIT_REPO") or "/root/Workspace/xy/DiT"
CFG_DIR = os.path.join(REPO, "src", "train", "configs")
sys.path.insert(0, REPO)
from src.model.dit import DiT_2Cond_models  # noqa: E402

# v12 日志实测: "Building 2-Cond model: ... (callig=52, glyph/char=7765, ...)"
NUM_CHARS = 7765

NAMES = [
    "v12_pretrain_S_cat_fame_kxl_tj_px60",
    "v13_pretrain_XS_cat_fame_kxl_tj_px60",
    "v14_pretrain_S320_cat_fame_kxl_tj_px60",
    "v15_pretrain_XS6_cat_fame_kxl_tj_px60",
    "v16_pretrain_S_cat_sep_fame_kxl_tj_px60",
]


def build_kwargs(cfg):
    """按 train.py:242-303 逐键复刻 (只保留模型构造相关的键)。"""
    aux = [s for s in str(cfg.get("aux_latent_shards_dirs", "") or "").split(",") if s]
    latent_channels = int(cfg.get("latent_channels", 4))
    return dict(
        input_size=32,
        num_calligraphers=int(cfg.get("num_calligraphers", 52)),
        num_characters=NUM_CHARS,
        use_checkpoint=bool(cfg.get("use_checkpoint", False)),
        learn_sigma=False,
        condition_fusion=cfg.get("condition_fusion", "factorized_cat"),
        callig_embed_dim=int(cfg.get("callig_embed_dim", 128)),
        char_embed_dim=int(cfg.get("char_embed_dim", 128)),
        glyph_vec_cond=bool(cfg.get("glyph_vec_cond", False)),
        glyph_vec_dim=int(cfg.get("glyph_vec_dim", 128)),
        glyph_vec_pool=cfg.get("glyph_vec_pool", "mean"),
        cond_drop_all_prob=float(cfg.get("cond_drop_all_prob", 0.1)),
        cond_drop_one_prob=float(cfg.get("cond_drop_one_prob", 0.0)),
        cond_drop_which_glyph_prob=float(cfg.get("cond_drop_which_glyph_prob", 0.5)),
        use_glyph_cond=bool(cfg.get("skel_as_glyph_cond", False)),
        use_char_cond=not bool(cfg.get("no_char_cond", False)),
        glyph_scale_init=float(cfg.get("glyph_scale_init", 0.4)),
        glyph_drop_prob=float(cfg.get("glyph_drop_prob", 0.0)),
        glyph_inject_layers=int(cfg.get("glyph_inject_layers", 0)),
        glyph_inject_mode=cfg.get("glyph_inject_mode", "adaln"),
        glyph_embedder_depth=int(cfg.get("glyph_embedder_depth", 0)),
        glyph_embedder_sep=bool(cfg.get("glyph_embedder_sep", False)),
        glyph_in_channels=4,
        in_channels=latent_channels + 4 * len(aux),
        image_channels=int(cfg.get("image_channels") or latent_channels),
        norm_type=cfg.get("norm_type", "rms"),
        mlp_type=cfg.get("mlp_type", "swiglu"),
        qk_norm=bool(cfg.get("qk_norm", 0)),
        rope=bool(cfg.get("rope", 0)),
        rope_theta=float(cfg.get("rope_theta", 100.0)),
        attn_impl=cfg.get("attn_impl", "sdpa"),
        freeze_char_table=bool(cfg.get("freeze_char_table", False)),
    )


def check(name):
    cfg = json.load(io.open(os.path.join(CFG_DIR, name + ".json"), encoding="utf-8"))
    variant = cfg["model"]
    if variant not in DiT_2Cond_models:
        print(f"  !! {variant} NOT REGISTERED -> 启动会 KeyError")
        return None

    kw = build_kwargs(cfg)
    m = DiT_2Cond_models[variant](**kw)
    d = len(m.blocks)
    h = m.blocks[0].norm1.dim
    n_param = sum(p.numel() for p in m.parameters())
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    flops = d * h * h   # doc63 §1 用的 GEMM/attention FLOPs 代理

    print(f"\n{'=' * 70}")
    print(f"{name}")
    print(f"  variant={variant:>18}  depth={d:>3}  h={h:>4}  "
          f"params={n_param/1e6:.2f}M  trainable={n_train/1e6:.2f}M")
    print(f"  d*h^2 = {flops:,}   glyph_inject_layers="
          f"{getattr(m, 'glyph_inject_layers', None)}   glyph_embedder_sep="
          f"{getattr(m, 'glyph_embedder_sep', None)}")
    print(f"  in_channels={kw['in_channels']}  image_channels={kw['image_channels']}  "
          f"fusion={kw['condition_fusion']}  glyph_vec_cond={kw['glyph_vec_cond']}")

    # ---- fwd/bwd: g 与 g=None 两条都必须过, 且所有可训练参数都要有梯度 ----
    B = 2
    torch.manual_seed(0)
    x = torch.randn(B, kw["in_channels"], 32, 32)
    t = torch.rand(B)
    yc = torch.randint(0, kw["num_calligraphers"], (B,))
    ych = torch.randint(0, NUM_CHARS, (B,))
    g = torch.randn(B, 4, 32, 32)

    m.train()
    o1 = m(x, t, yc, ych, g=g)
    o2 = m(x, t, yc, ych, g=None)
    assert o1.shape == o2.shape, (o1.shape, o2.shape)
    print(f"  forward(g)={tuple(o1.shape)}  forward(g=None)={tuple(o2.shape)}  OK")

    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    for _ in range(3):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(m(x, t, yc, ych, g=g),
                                            torch.randn_like(o1))
        loss.backward()
        opt.step()
    print(f"  after 3 steps loss={loss.item():.4f}")

    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    if missing:
        print(f"  !! {len(missing)} 个可训练参数没拿到梯度 (DDP 会报错):")
        for n in missing[:10]:
            print(f"       - {n}")
        return None
    print("  所有可训练参数均获得梯度 (DDP-safe) OK")

    m.eval()
    with torch.no_grad():
        oc = m.forward_with_cfg(x, t, yc, ych, cfg_scale=0.7, g=g)
    assert oc.shape == o1.shape, (oc.shape, o1.shape)
    print(f"  forward_with_cfg(cfg=0.7)={tuple(oc.shape)}  OK")
    return dict(variant=variant, depth=d, h=h, params=n_param, flops=flops)


rows = []
for nm in NAMES:
    r = check(nm)
    if r is None:
        print(f"\nFAILED: {nm}")
        sys.exit(1)
    rows.append(r)

print(f"\n{'=' * 70}")
print(f"{'config':>34} {'depth':>5} {'h':>4} {'params':>9} {'d*h^2':>12} {'vs S/2':>7}")
ref = [r for r in rows if r["variant"] == "DiT-2Cond-S/2"][0]["flops"]
for nm, r in zip(NAMES, rows):
    print(f"{nm:>34} {r['depth']:>5} {r['h']:>4} {r['params']/1e6:>8.2f}M "
          f"{r['flops']:>12,} {r['flops']/ref:>7.3f}x")
print("\nALL LADDER CONFIGS BUILD + TRAIN-STEP OK")