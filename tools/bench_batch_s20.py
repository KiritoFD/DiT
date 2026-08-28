# -*- coding: utf-8 -*-
"""为 s20 选 batch：忠实复刻 train.py 的训练步（bf16 autocast + AdamW + clip + EMA），
实测峰值显存与 img/s，据此在"调小 batch"和"开 gradient checkpointing"之间选。

背景：4090 服务器是 torch 1.13.1，没有 F.scaled_dot_product_attention，
也没有 xformers/flash-attn，因此 attention 必然是 eager（会物化 B*H*N^2 的
attention 矩阵）。torch 版本不可动，所以只能靠 infra 解决显存。

    python tools/bench_batch_s20.py --target-gb 21
"""
import argparse
import copy
import gc
import os
import sys
import time

ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch

from src.model import DiT_2Cond_models
from src.loss import create_flow_matching

KW = dict(
    input_size=32, num_calligraphers=1011, num_characters=35130,
    learn_sigma=False, condition_fusion="factorized_add",
    callig_embed_dim=128, char_embed_dim=384,
    char_proj_mode="mlp", freeze_char_table=True,
    norm_type="rms", mlp_type="swiglu", qk_norm=True,
    rope=True, rope_theta=100.0, attn_impl="auto",
)


def unzero_adaln(model, std=0.02):
    """打破 adaLN-Zero，模拟已训练若干步（否则输出恒 0，测的不是真实开销）。"""
    with torch.no_grad():
        for b in model.blocks:
            b.adaLN_modulation[-1].weight.normal_(0, std)
            b.adaLN_modulation[-1].bias.normal_(0, std)
        model.final_layer.adaLN_modulation[-1].weight.normal_(0, std)
        model.final_layer.adaLN_modulation[-1].bias.normal_(0, std)
        model.final_layer.linear.weight.normal_(0, std)


def bench(model_name, batch, ckpt, steps=6, warmup=2):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(0)
    model = DiT_2Cond_models[model_name](use_checkpoint=ckpt, **KW).cuda()
    unzero_adaln(model)
    # train.py 的做法：EMA 是完整 deepcopy，常驻显存
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.02)
    fm = create_flow_matching("25", t_sampler="logit_normal",
                              sampler="heun", heun_batch=1, shift=1.0)

    dev = torch.device("cuda")
    x = torch.randn(batch, 4, 32, 32, device=dev)
    yc = torch.randint(0, 1011, (batch,), device=dev)
    ych = torch.randint(0, 35130, (batch,), device=dev)

    model.train()
    t0 = None
    try:
        for s in range(warmup + steps):
            if s == warmup:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
            t = fm.sample_t(batch, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                terms = fm.training_losses(model, x, t,
                                           dict(y_callig=yc, y_char=ych))
                loss = terms["loss"].mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            with torch.no_grad():
                for ep, p in zip(ema.parameters(), model.parameters()):
                    ep.mul_(0.999).add_(p.detach(), alpha=1 - 0.999)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        del model, ema, opt
        gc.collect()
        torch.cuda.empty_cache()
        return None

    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    del model, ema, opt, x, yc, ych, terms, loss
    gc.collect()
    torch.cuda.empty_cache()
    return peak, steps / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-gb", type=float, default=21.0)
    ap.add_argument("--model", default="DiT-2Cond-S/2")
    args = ap.parse_args()

    print(f"torch={torch.__version__}  gpu={torch.cuda.get_device_name(0)}  "
          f"capability=sm_{torch.cuda.get_device_capability(0)}")
    from src.model.modules import resolve_attn_impl
    impl = resolve_attn_impl("auto")
    print(f"resolved attn_impl = {impl}  "
          f"(torch<2.0 且无 xformers => 必然 eager)")
    print(f"total VRAM = {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    print()

    print(f"{'ckpt':>5s} {'batch':>6s} {'peak_GB':>8s} {'steps/s':>8s} "
          f"{'img/s':>7s}  {'fit':>5s}")
    print("-" * 48)

    results = {}
    plan = [
        (False, [96, 128, 160, 192, 224, 240]),
        (True,  [160, 192, 240, 320, 384, 448]),
    ]
    for ckpt, batches in plan:
        for b in batches:
            r = bench(args.model, b, ckpt)
            if r is None:
                print(f"{str(ckpt):>5s} {b:6d}      OOM")
                # 大 batch 已 OOM，更大的不用再试
                if b >= max(batches):
                    break
                continue
            peak, sps = r
            fit = "OK" if peak <= args.target_gb else "OVER"
            print(f"{str(ckpt):>5s} {b:6d} {peak:8.2f} {sps:8.2f} "
                  f"{sps*b:7.0f}  {fit:>5s}")
            results[(ckpt, b)] = (peak, sps)
            if peak > args.target_gb:
                break   # 已经超了，更大的不用试

    print()
    print("=" * 48)
    print(f"结论（目标显存 <= {args.target_gb} GB）:")
    ok = {k: v for k, v in results.items() if v[0] <= args.target_gb}
    if not ok:
        print("  无配置满足，需继续调小 batch 或减小模型")
        return
    best = max(ok.items(), key=lambda kv: kv[1][1] * kv[0][1])
    (ck, b), (peak, sps) = best
    print(f"  最优: use_checkpoint={ck}, batch={b}")
    print(f"        peak={peak:.2f} GB, {sps:.2f} steps/s, {sps*b:.0f} img/s")

    # 同 batch 下对比 ckpt 开/关的开销（若有交集）
    common = set(b for c, b in results if not c) & set(b for c, b in results if c)
    if common:
        print("\n  同 batch 下 checkpoint 的代价:")
        for b in sorted(common):
            p0, s0 = results[(False, b)]
            p1, s1 = results[(True, b)]
            print(f"    batch={b:4d}: 显存 {p0:.2f} -> {p1:.2f} GB "
                  f"({(p1-p0)/p0*100:+.0f}%)  |  速度 {s0:.2f} -> {s1:.2f} steps/s "
                  f"({(s1-s0)/s0*100:+.0f}%)")

    # 无 ckpt 的 best vs 有 ckpt 的 best
    for tag, sel in (("no-ckpt", False), ("ckpt", True)):
        cand = {k: v for k, v in ok.items() if k[0] is sel}
        if cand:
            (ck2, b2), (p2, s2) = max(cand.items(), key=lambda kv: kv[1][1] * kv[0][1])
            print(f"  {tag:8s} 最优: batch={b2:4d}  peak={p2:.2f}GB  "
                  f"{s2*b2:.0f} img/s")


if __name__ == "__main__":
    main()
