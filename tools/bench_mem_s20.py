# -*- coding: utf-8 -*-
"""为 s20 选模型规格与 batch：在真实训练循环下测峰值显存与吞吐。

目标显存 ~21G。测 S/2 (46M) 与 WS/2 (70M) 在 v2 架构（RMSNorm+SwiGLU+RoPE+QKNorm）
下的峰值，并给出各自能塞进 21G 的最大 batch。
"""
import argparse, gc, os, sys, time
import torch

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")

from src.model import DiT_2Cond_models
from src.loss import create_flow_matching


def bench(model_name, batch, steps=8, warmup=3, lr=2e-4):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    kw = dict(input_size=32, num_calligraphers=1011, num_characters=35130,
              learn_sigma=False, condition_fusion="factorized_add",
              callig_embed_dim=128, char_embed_dim=384,
              char_proj_mode="mlp", freeze_char_table=True,
              norm_type="rms", mlp_type="swiglu", qk_norm=True,
              rope=True, rope_theta=100.0, attn_impl="sdpa")
    torch.manual_seed(0)
    model = DiT_2Cond_models[model_name](**kw).cuda()
    # 打破 adaLN-Zero，模拟真实训练（否则输出恒 0，计时不真实）
    with torch.no_grad():
        for b in model.blocks:
            b.adaLN_modulation[-1].weight.normal_(0, 0.02)
            b.adaLN_modulation[-1].bias.normal_(0, 0.02)
        model.final_layer.adaLN_modulation[-1].weight.normal_(0, 0.02)
        model.final_layer.adaLN_modulation[-1].bias.normal_(0, 0.02)
        model.final_layer.linear.weight.normal_(0, 0.02)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.02)
    # EMA（train.py 会 deepcopy 一份）
    ema = torch.nn.utils.parameters_to_vector(
        [p.detach().clone() for p in model.parameters()]) \
        if False else torch.optim.swa_utils.AveragedModel(
            model, avg_fn=lambda a, b, n: a + (b - a) * 0.01)

    fm = create_flow_matching("25", t_sampler="logit_normal",
                              sampler="heun", heun_batch=1, shift=1.0)
    x = torch.randn(batch, 4, 32, 32, device="cuda")
    yc = torch.randint(0, 1011, (batch,), device="cuda")
    ych = torch.randint(0, 35130, (batch,), device="cuda")

    model.train()
    t0 = None
    for s in range(steps + warmup):
        if s == warmup:
            torch.cuda.synchronize(); t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
        t = fm.sample_t(batch, torch.device("cuda"))
        terms = fm.training_losses(model, x, t, dict(y_callig=yc, y_char=ych))
        loss = terms["loss"].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        with torch.no_grad():
            ema.update_parameters(model)
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    sps = steps / dt
    del model, opt, ema, x, yc, ych, terms, loss
    gc.collect(); torch.cuda.empty_cache()
    return peak, sps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-gb", type=float, default=21.0)
    ap.add_argument("--model", default="", help="只测指定模型")
    args = ap.parse_args()

    print(f"{'model':16s} {'batch':>6s} {'peak_GB':>8s} {'steps/s':>8s} "
          f"{'img/s':>8s}  {'fit<=21G':>8s}")
    print("-" * 62)
    results = {}
    plans = {
        "DiT-2Cond-S/2": [160, 200, 240, 280],
        "DiT-2Cond-WS/2": [96, 128, 160, 192],
    }
    if args.model:
        plans = {k: v for k, v in plans.items() if k == args.model}

    for name, batches in plans.items():
        for b in batches:
            try:
                peak, sps = bench(name, b)
            except torch.cuda.OutOfMemoryError:
                print(f"{name:16s} {b:6d}   OOM")
                torch.cuda.empty_cache(); gc.collect()
                break
            fit = "OK" if peak <= args.target_gb else "OVER"
            print(f"{name:16s} {b:6d} {peak:8.2f} {sps:8.2f} {sps*b:8.0f}  {fit:>8s}")
            results[(name, b)] = (peak, sps)
            if peak > args.target_gb:
                break

    print("\n" + "=" * 62)
    print("建议（目标显存 %.1f GB）:" % args.target_gb)
    for name in plans:
        cands = [(b, v) for (n, b), v in results.items()
                 if n == name and v[0] <= args.target_gb]
        if not cands:
            print(f"  {name}: 无可用 batch（需 <={args.target_gb}G）")
            continue
        b, (peak, sps) = max(cands, key=lambda t: t[1][1] * t[0])
        print(f"  {name:16s} batch={b:4d}  peak={peak:.2f}GB  "
              f"throughput={sps*b:.0f} img/s  ({sps:.2f} steps/s)")


if __name__ == "__main__":
    main()
