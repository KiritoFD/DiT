# -*- coding: utf-8 -*-
"""
verify_glyph_gradient.py — 验证标准字形条件(glyph cond)的梯度通路与参数开销。

回答三个问题
------------
1. 加了逐层注入后，参数量增加多少？
2. glyph_embedder 到底能不能收到梯度？（zero-init 会不会把它卡死）
3. init 时注入是否严格恒等（会不会破坏已有训练）？

背景
----
标准字形条件 g 是 4×32×32 的空间图，经 glyph_embedder(Conv2d 4→hidden) 编成
token 后注入。原本只在 **输入层** 做一次 token-add：

    x = x + glyph_scale * g_tok        # glyph_scale 初始 0.4，非零

改进后增加 **逐层注入**（与 ControlNet 的 ZeroAdaLNInjection 对齐）：

    for i, block in enumerate(blocks):
        x = block(x, c)
        x = glyph_injections[i](x, g_tok)   # out = x*(1+s) + t，s/t 由 zero-init Linear 产出

梯度分析（本脚本实测验证）
--------------------------
对逐层注入 out = x*(1+s) + t，其中 s,t = W@g_tok + b，W=0, b=0：

    d(out)/d(b)      = 1          → bias 立刻有梯度     ✅
    d(out)/d(W)      = g_tok      → W 立刻有梯度        ✅
    d(out)/d(g_tok)  = W = 0      → **glyph_embedder 收不到梯度**  ⚠️

这是 ControlNet zero-conv warm-start 的标准行为（先学注入权重，再学控制特征），
对 ControlNet 是正确的 —— 因为它的 ctrl_encoder 是独立大网络，本来就要慢慢学。

但 glyph_embedder 只是个 Conv2d，若**只**靠逐层注入，它会在训练早期完全冻结。
因此本实现**保留输入层 token-add**：glyph_scale=0.4 非零，
d(out)/d(g_tok) = 0.4 → glyph_embedder 从 step 0 就有直通梯度。

本脚本用四组配置实证上述结论。

用法
----
    python tools/verify_glyph_gradient.py [--device cuda] [--batch 2]
"""
import os, sys, argparse, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch


def build(glyph_inject_layers, glyph_scale_init, device):
    from src.model import DiT_2Cond_models
    model = DiT_2Cond_models["DiT-2Cond-S/2"](
        input_size=32, in_channels=4,
        num_calligraphers=1013, num_characters=35130,
        use_checkpoint=False, learn_sigma=False,
        condition_fusion="factorized_add",
        callig_embed_dim=128, char_embed_dim=384,
        cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
        cond_drop_which_glyph_prob=0.75,
        skel_head_enabled=False,
        use_glyph_cond=True,
        glyph_scale_init=glyph_scale_init,
        glyph_inject_layers=glyph_inject_layers,
        char_proj_mode="mlp", freeze_char_table=True,
        norm_type="rms", mlp_type="swiglu", qk_norm=True,
        rope=True, rope_theta=100.0, attn_impl="sdpa",
    ).to(device)
    return model


def probe(name, model, device, batch=2, T=1024, D=384):
    """前向 + 反向，报告关键参数的梯度范数。"""
    model.train()
    x = torch.randn(batch, 4, 32, 32, device=device)
    g = torch.randn(batch, 4, 32, 32, device=device)
    t = torch.rand(batch, device=device)
    yc = torch.randint(0, 1013, (batch,), device=device)
    yh = torch.randint(0, 35130, (batch,), device=device)

    out = model(x, t, y_callig=yc, y_char=yh, g=g)
    loss = out.float().pow(2).mean()
    model.zero_grad(set_to_none=True)
    loss.backward()

    def gnorm(p):
        return 0.0 if (p is None or p.grad is None) else p.grad.norm().item()

    res = {
        "name": name,
        "params": sum(p.numel() for p in model.parameters()),
        "output_shape": tuple(out.shape),
        "glyph_embedder_grad": gnorm(model.glyph_embedder.weight),
        "glyph_scale_grad": gnorm(model.glyph_scale),
    }
    inj = model.glyph_injections
    if inj is not None and len(inj) > 0:
        res["n_injections"] = len(inj)
        res["inj_at_blocks"] = list(model.glyph_inject_at)
        res["inj0_W_grad"] = gnorm(inj[0].proj.weight)
        res["inj0_b_grad"] = gnorm(inj[0].proj.bias)
        res["injN_W_grad"] = gnorm(inj[-1].proj.weight)
    else:
        res["n_injections"] = 0
    return res


def check_identity(model, device, batch=2):
    """验证 zero-init 下逐层注入严格恒等。

    做法：构造两个相同输入，一个正常前向；另一个把所有 inj 的 proj 权重
    显式置零后前向。两者应完全相等（因为 init 本来就全零，故只需确认
    当前权重确实全零）。
    """
    inj = model.glyph_injections
    if inj is None:
        return None
    max_abs = 0.0
    for m in inj:
        max_abs = max(max_abs, m.proj.weight.abs().max().item(),
                      m.proj.bias.abs().max().item())
    return max_abs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--out", default="assets/glyph_grad_probe.json")
    args = ap.parse_args()
    dev = torch.device(args.device)

    cases = [
        ("A: 仅输入层 token-add (旧行为, glyph_scale=0.4)", 0, 0.4),
        ("B: 输入层 + 逐层注入 6 层", 6, 0.4),
        ("C: 输入层 + 逐层注入 12 层", 12, 0.4),
        ("D: 仅逐层注入 6 层 (glyph_scale=0, 对照组)", 6, 0.0),
    ]

    results = []
    print(f"{'配置':<44}{'参数量':>12}{'embed_grad':>13}{'scale_grad':>12}")
    print("-" * 82)
    for name, layers, gs in cases:
        torch.manual_seed(0)
        m = build(layers, gs, dev)
        r = probe(name, m, dev, args.batch)
        r["inject_max_abs"] = check_identity(m, dev, args.batch)
        results.append(r)
        print(f"{name:<44}{r['params']:>12,}{r['glyph_embedder_grad']:>13.6f}"
              f"{r['glyph_scale_grad']:>12.6f}")
        del m
        torch.cuda.empty_cache() if dev.type == "cuda" else None

    base = results[0]["params"]
    print(f"\n参数增量（相对 A）:")
    for r in results[1:]:
        print(f"  {r['name'][:44]:<44} +{r['params']-base:,} "
              f"({(r['params']/base-1)*100:+.1f}%)")

    print(f"\n梯度通路诊断:")
    for r in results:
        flag = "✅" if r["glyph_embedder_grad"] > 0 else "❌ 无梯度"
        print(f"  {r['name'][:44]:<44} glyph_embedder {flag}")
        if r.get("n_injections"):
            print(f"      inj[0]: W_grad={r['inj0_W_grad']:.6f} "
                  f"b_grad={r['inj0_b_grad']:.6f}  "
                  f"inj[{r['n_injections']-1}]: W_grad={r['injN_W_grad']:.6f}")
        if r.get("inject_max_abs") is not None:
            ok = "✅ 恒等" if r["inject_max_abs"] == 0 else f"⚠️ {r['inject_max_abs']}"
            print(f"      zero-init 检查: max|W,b|={r['inject_max_abs']} {ok}")

    print(f"\n结论提示:")
    print("  · A/B/C 的 glyph_embedder 均应有非零梯度（来自输入层 token-add）")
    print("  · D 若为 0，证明『只靠逐层注入会冻死 glyph_embedder』")
    print("     → 这正是保留输入层 token-add 的原因")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
