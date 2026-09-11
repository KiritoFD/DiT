#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lr_range_test.py — LR range test (Leslie Smith) 科学找合适学习率.

从 ckpt resume 构建模型, LR 从极小指数增长到极大, 每步跑真实 forward/backward/step
记录 flow diff loss, 最后输出 loss-LR 曲线的最小值邻域 (= loss 下降最快的 LR 区间,
即建议 LR)。模型/数据路径一律从 ckpt args 读 (不单独传配置)。

用法 (远程, 需整卡 GPU; 训练占满时跑会 OOM):
    python tools/lr_range_test.py --ckpt <path/to/xxxx.pt> [--steps 300] \
        [--lr-min 1e-7] [--lr-max 1e-2] [--batch 16] [--style-token-n N]

判读: 
    loss 最低点的 LR 的 1/10 作为保守复用点; loss 仍在陡降区间的 LR 可作为激进点。
"""
import argparse, glob, os, sys, csv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.environ.get("DIIT_ROOT", "/root/Workspace/xy/DiT")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import torch
import torch.nn.functional as F


def load_all(d):
    ids, lats = [], []
    for sp in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        with np.load(sp) as z:
            ids.append(z["img_ids"])
            lats.append(z["latents"])
    ids = np.concatenate(ids)
    lats = np.concatenate(lats, axis=0).astype(np.float32)
    return {int(i): lats[j] for j, i in enumerate(ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr-min", type=float, default=1e-7)
    ap.add_argument("--lr-max", type=float, default=1e-2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--style-token-n", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device("cuda")
    torch.manual_seed(0)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck.get("args", {}) or {}
    if not isinstance(a, dict):
        a = vars(a) if hasattr(a, "__dict__") else {}

    from src.model import DiT_2Cond_models
    arch = dict(norm_type=a.get("norm_type", "rms"), mlp_type=a.get("mlp_type", "swiglu"),
                qk_norm=bool(a.get("qk_norm", 1)), rope=bool(a.get("rope", 1)),
                rope_theta=float(a.get("rope_theta", 100.0)), attn_impl="sdpa")
    model = DiT_2Cond_models[a.get("model", "DiT-2Cond-Sp/2")](
        num_calligraphers=int(a.get("num_calligraphers", 41)),
        num_characters=int(a.get("num_characters", 35130)),
        condition_fusion=a.get("condition_fusion", "factorized_add"),
        callig_embed_dim=int(a.get("callig_embed_dim", 128)),
        char_embed_dim=int(a.get("char_embed_dim", 384)),
        char_proj_mode=a.get("char_proj_mode", "mlp"),
        callig_proj_mode=a.get("callig_proj_mode", "mlp"),
        use_glyph_cond=True, use_char_cond=not bool(a.get("no_char_cond", False)),
        glyph_scale_init=float(a.get("glyph_scale_init", 0.6)),
        glyph_drop_prob=0.0,
        glyph_inject_layers=int(a.get("glyph_inject_layers", 12)),
        glyph_inject_mode=a.get("glyph_inject_mode", "xattn"),
        glyph_embedder_depth=int(a.get("glyph_embedder_depth", 2)),
        style_token_n=args.style_token_n,
        style_role_init=float(a.get("style_role_init", 0.02)), learn_sigma=False, **arch)
    sd = ck.get("ema") or ck.get("model") or ck
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    model.to(dev).train()
    print(f"[model] {a.get('model')} miss={len(miss)} (style_token_n={args.style_token_n})",
          flush=True)

    # 数据: csv 前 n 行, 固定一个 batch 贯穿全程 (LR range test 惯例)
    csv_path = "5script/train_fame3_e_full.csv" if "e_full" in str(a.get("data_csv", "")) \
        else a.get("data_csv", "5script/train_fame3_clean_v8.csv")
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))[: args.batch]
    img_ids = [int(r["image_path"].rsplit(".", 1)[0].rsplit("/", 1)[-1]) for r in rows]
    y_callig = torch.tensor([int(r["calligrapher_id"]) for r in rows], dtype=torch.long).to(dev)

    lat_map = load_all("final_latents_fame_e" if "fame_e" in csv_path else "final_latents_fame3_v8")
    g_map = load_all("std_skel1_latents_fame_e" if "fame_e" in csv_path else "std_skel1_latents_fame3_v8")
    x0 = torch.stack([torch.from_numpy(lat_map[i]) for i in img_ids]).to(dev)
    g = torch.stack([torch.from_numpy(g_map.get(i, np.zeros_like(x0[0].cpu()))) for i in img_ids]).to(dev)
    print(f"[data] csv={csv_path} batch={args.batch} ||x0||={x0.norm():.2f}", flush=True)

    # callig id 重映射 (若启用词表收紧, raw id -> 0..40)
    if a.get("callig_id_map"):
        from src.utils.callig_map import load_callig_id_map
        _m, _ = load_callig_id_map(a["callig_id_map"])
        y_callig = torch.tensor([_m.get(int(i), int(i)) for i in y_callig.tolist()],
                                dtype=torch.long, device=dev)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_min, weight_decay=0.02)
    mult = (args.lr_max / args.lr_min) ** (1.0 / max(args.steps - 1, 1))

    results = []
    noise = torch.randn_like(x0)
    print(f"\n[LR range test] lr_min={args.lr_min:.1e} lr_max={args.lr_max:.1e} "
          f"steps={args.steps} multiplier={mult:.5f}", flush=True)
    for step in range(args.steps):
        lr = args.lr_min * (mult ** step)
        for pg in opt.param_groups:
            pg["lr"] = lr
        t = torch.rand(args.batch, device=dev)
        x_t = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * noise
        v = noise - x0
        opt.zero_grad(set_to_none=True)
        pred = model(x_t, t * 1000.0, y_callig=y_callig, y_char=None, g=g)
        # learn_sigma=False 时模型直接输出 v 预测
        loss = F.mse_loss(pred, v)
        loss.backward()
        opt.step()
        results.append((lr, float(loss.item())))
        if step % 50 == 0 or step == args.steps - 1:
            print(f"  step {step:>4}  lr={lr:.2e}  loss={loss.item():.4f}", flush=True)

    # 建议 LR: loss 最低点
    lrs = np.array([r[0] for r in results])
    losses = np.array([r[1] for r in results])
    best_i = int(np.argmin(losses))
    print(f"\n[结论]")
    print(f"  loss 最低点: lr={lrs[best_i]:.2e} loss={losses[best_i]:.4f}")
    print(f"  保守建议 (最低点/10): lr={lrs[best_i]/10:.2e}")
    print(f"  激进建议 (最低点):    lr={lrs[best_i]:.2e}")
    # loss 陡降区间 (loss 对 log-lr 的负梯度最大处)
    dl = np.gradient(losses, np.log10(lrs))
    steep_i = int(np.argmin(dl))
    print(f"  loss 陡降最快的 LR: {lrs[steep_i]:.2e} (该区为'还在快速下降'区)")
    np.savez("/tmp/lr_range_test.npz", lrs=lrs, losses=losses)
    print(f"  曲线已存 /tmp/lr_range_test.npz")


if __name__ == "__main__":
    main()