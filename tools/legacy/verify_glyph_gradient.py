# -*- coding: utf-8 -*-
"""
verify_glyph_gradient.py — 验证标准字形条件(glyph cond)的梯度通路与参数开销。

回答三个问题
------------
1. 加了逐层注入后，参数量增加多少？
2. 梯度从动力学上能不能良好流动？（glyph_embedder 能否有效学习）
3. init 时注入是否严格恒等（会不会破坏已有训练）？

⚠️ 关于「梯度为 0」的正确解读
------------------------------
DiT 的 ``initialize_weights()`` 会把 ``final_layer`` **整体零初始化**
（DiT 的 adaLN-Zero 标准做法，让初始输出恒为 0）：

    nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
    nn.init.constant_(self.final_layer.linear.weight, 0)

于是 ``out = final_layer(x, c) ≡ 0``，且 ``d(out)/d(上游一切) = 0``。
**在真正的 step 0，任何上游参数的梯度都是 0 —— 这是设计使然，不是 bug。**
训练开始后 final_layer 迅速学到非零值，梯度随即正常流动。

所以「梯度是否为 0」必须用**训练中期**的状态来测，而不是 step 0。
本脚本因此提供两种模式：
  --mode init   : 真实 step 0（预期全 0，仅作对照）
  --mode trained: 把 final_layer 重新初始化为非零，模拟训练中期
                  （这是判断梯度通路是否健康的正确姿势）

逐层注入的梯度机制
------------------
注入为 ``out = x*(1+s) + t``，其中 ``s,t = W@g_tok + b``，W=0, b=0（zero-init）：

    d(out)/d(b)      = 1          → bias 立刻有梯度          ✅
    d(out)/d(W)      = g_tok      → W 立刻有梯度             ✅
    d(out)/d(g_tok)  = W = 0      → **step 0 时 glyph_embedder 收不到梯度**

这是 ControlNet zero-conv warm-start 的标准行为（先学注入权重，再学控制特征），
对 ControlNet 正确 —— 它的 ctrl_encoder 是独立大网络，本就要慢慢学。

但 glyph_embedder 只是个 Conv2d，若**只**靠逐层注入，它在早期会被完全冻住。
因此本实现**保留输入层 token-add**：``glyph_scale=0.4`` 非零，
``d(out)/d(g_tok) = 0.4`` → glyph_embedder 从 step 0 就有直通梯度。

用法
----
    python tools/verify_glyph_gradient.py --mode trained [--device cuda]
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
        rope=True, rope_theta=100.0,
        attn_impl="eager",      # CPU 上 eager 最稳，且本测只关心梯度拓扑
    ).to(device)
    return model


def simulate_trained(model):
    """把 final_layer 重新初始化为非零，模拟训练中期。

    step 0 时 final_layer 全零 → out≡0 → 上游梯度全为 0（DiT adaLN-Zero 设计）。
    要判断「梯度在动力学上能否良好流动」，必须看 final_layer 已学到非零之后。
    这里用 std=0.02 的正态初始化近似（与 DiT 其它 head 的初始化量级一致）。
    """
    fl = model.final_layer
    nn = torch.nn
    nn.init.normal_(fl.linear.weight, std=0.02)
    nn.init.normal_(fl.linear.bias, std=0.02)
    nn.init.normal_(fl.adaLN_modulation[-1].weight, std=0.02)
    nn.init.normal_(fl.adaLN_modulation[-1].bias, std=0.02)


def probe(name, model, device, batch=2, mode="trained"):
    model.train()
    x = torch.randn(batch, 4, 32, 32, device=device)
    g = torch.randn(batch, 4, 32, 32, device=device)
    t = torch.rand(batch, device=device)
    yc = torch.randint(0, 1013, (batch,), device=device)
    yh = torch.randint(0, 35130, (batch,), device=device)

    if mode == "trained":
        simulate_trained(model)

    out = model(x, t, y_callig=yc, y_char=yh, g=g)
    if isinstance(out, tuple):
        out = out[0]
    loss = out.float().pow(2).mean()
    model.zero_grad(set_to_none=True)
    loss.backward()

    def gnorm(p):
        return 0.0 if (p is None or p.grad is None) else p.grad.norm().item()

    res = {
        "name": name,
        "params": sum(p.numel() for p in model.parameters()),
        "loss": loss.item(),
        "glyph_embedder_grad": gnorm(model.glyph_embedder.weight),
        "glyph_scale_grad": gnorm(model.glyph_scale),
        "char_proj_grad": gnorm(getattr(model.char_proj, "weight", None)
                                if hasattr(model.char_proj, "weight")
                                else (model.char_proj[0].weight
                                      if isinstance(model.char_proj, torch.nn.Sequential)
                                      else None)),
        "x_embedder_grad": gnorm(model.x_embedder.proj.weight),
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


def check_zero_init(model):
    """init 时注入模块的 W/b 必须严格为 0（否则破坏恒等初始化）。"""
    inj = model.glyph_injections
    if inj is None:
        return None
    mx = 0.0
    for m in inj:
        mx = max(mx, m.proj.weight.abs().max().item(),
                 m.proj.bias.abs().max().item())
    return mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--mode", default="trained", choices=["init", "trained", "both"])
    ap.add_argument("--out", default="5script/glyph_grad_probe.json")
    args = ap.parse_args()
    dev = torch.device(args.device)

    cases = [
        ("A: 仅输入层 token-add (旧行为, scale=0.4)", 0, 0.4),
        ("B: 输入层 + 逐层注入 6 层", 6, 0.4),
        ("C: 输入层 + 逐层注入 12 层", 12, 0.4),
        ("D: 仅逐层注入 6 层 (scale=0, 对照组)", 6, 0.0),
    ]

    modes = ["trained", "init"] if args.mode == "both" else [args.mode]
    all_results = []

    for mode in modes:
        print("\n" + "=" * 96)
        print(f"模式 = {mode}  "
              f"({'真实 step 0：final_layer 零初始化，预期上游梯度全 0' if mode=='init' else '模拟训练中期：final_layer 已非零'})")
        print("=" * 96)
        print(f"{'配置':<44}{'参数量':>12}{'embed_grad':>14}{'scale_grad':>13}")
        print("-" * 96)
        results = []
        for name, layers, gs in cases:
            torch.manual_seed(0)
            m = build(layers, gs, dev)
            if mode == "init":
                m.initialize_weights()      # 回到真实 step 0
            r = probe(name, m, dev, args.batch, mode=mode)
            r["mode"] = mode
            r["inject_max_abs_at_init"] = check_zero_init(m)
            results.append(r)
            print(f"{name:<44}{r['params']:>12,}{r['glyph_embedder_grad']:>14.6f}"
                  f"{r['glyph_scale_grad']:>13.6f}")
            del m
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        all_results.extend(results)

        if mode == "trained":
            base = results[0]["params"]
            print(f"\n参数增量（相对 A）:")
            for r in results[1:]:
                print(f"  {r['name'][:44]:<44} +{r['params']-base:,} "
                      f"({(r['params']/base-1)*100:+.2f}%)")
            print(f"\n梯度通路（训练中期）:")
            for r in results:
                flag = "OK" if r["glyph_embedder_grad"] > 0 else "无梯度"
                print(f"  {r['name'][:44]:<44} glyph_embedder = "
                      f"{r['glyph_embedder_grad']:.6e}  [{flag}]")
                if r.get("n_injections"):
                    print(f"       inj[0]: W={r['inj0_W_grad']:.3e} "
                          f"b={r['inj0_b_grad']:.3e} | "
                          f"inj[{r['n_injections']-1}]: W={r['injN_W_grad']:.3e}")
                print(f"       x_embedder={r['x_embedder_grad']:.3e}  "
                      f"(参照：主干正常梯度量级)")
            print(f"\nzero-init 检查（应严格为 0，否则破坏恒等初始化）:")
            for r in results:
                v = r.get("inject_max_abs_at_init")
                if v is None:
                    print(f"  {r['name'][:44]:<44} 无注入模块")
                else:
                    print(f"  {r['name'][:44]:<44} max|W,b| = {v}  "
                          f"{'OK 恒等' if v == 0.0 else 'WARN 非零'}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
