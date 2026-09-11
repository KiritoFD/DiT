#!/usr/bin/env python3
"""debug_v10b_gradients.py — v10b 梯度/瓶颈诊断 (GPU, 可靠性优先).

从 v10b ckpt 加载真模型, 用真分布数据 (MCCDLatentDataset + skel latent g)
跑 n 个 batch 的 forward-backward, 诊断:

1. 模块相对更新量 grad_norm/param_norm: 找死区 (rel<<1e-4) / 过激模块
2. g 注入作用: 有/无 g 的整网输出相对差 + glyph_scale 学到值
3. g 的 token 贡献在各层的衰减: 逐 block 输出在有/无 g 下的差异
4. REPA loss 梯度占比: w 乘以梯度后占总梯度比例

用法: python tools/debug_v10b_gradients.py --ckpt X.pt --n-batch 4
"""
import os, sys, json, argparse, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
os.chdir(_root)

import numpy as np
import torch


def _strip(sd):
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-batch", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="5script/results/v10b_debug_grad.json")
    args = ap.parse_args()

    dev = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck.get("args", {}) or {}
    if isinstance(a, argparse.Namespace):
        a = vars(a)

    from src.model import DiT_2Cond_models
    arch = dict(norm_type=a.get("norm_type", "rms"), mlp_type=a.get("mlp_type", "swiglu"),
                qk_norm=bool(a.get("qk_norm", 1)), rope=bool(a.get("rope", 1)),
                rope_theta=float(a.get("rope_theta", 100.0)),
                attn_impl="sdpa")
    use_g = bool(a.get("w_glyph_cond", False) or a.get("skel_as_glyph_cond", False))
    model = DiT_2Cond_models[a.get("model", "DiT-2Cond-S/2")](
        input_size=int(a.get("image_size", 256)) // 8,
        num_calligraphers=int(a.get("num_calligraphers", 1013)),
        num_characters=int(a.get("num_characters", 35130)),
        condition_fusion=a.get("condition_fusion", "factorized_add"),
        callig_embed_dim=int(a.get("callig_embed_dim", 128)),
        char_embed_dim=int(a.get("char_embed_dim", 384)),
        char_proj_mode=a.get("char_proj_mode", "mlp"),
        freeze_char_table=bool(a.get("freeze_char_table", True)),
        cond_drop_all_prob=0.05, cond_drop_one_prob=0.3,
        cond_drop_which_glyph_prob=0.85, use_checkpoint=False, learn_sigma=False,
        use_glyph_cond=use_g,
        use_char_cond=not bool(a.get("no_char_cond", False)),
        glyph_scale_init=float(a.get("glyph_scale_init", 0.4)),
        glyph_drop_prob=float(a.get("glyph_drop_prob", 0.0)),
        glyph_inject_layers=int(a.get("glyph_inject_layers", 0)), **arch).to(dev)
    sd = _strip(ck.get("ema") or ck.get("model") or ck)
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[load] miss={len(miss)} unexp={len(unexp)}", flush=True)
    print(f"[model] glyph_scale param = {getattr(model.glyph_embedder, 'scale', 'n/a')}", flush=True)

    # ── 数据 ──
    from src.utils.latent_dataset import MCCDLatentDataset
    from torch.utils.data import DataLoader, Subset
    csv_path = a.get("gpu_eval_csv") or a.get("eval_csv") or a.get("data_csv")
    ds = MCCDLatentDataset(
        csv_file=csv_path, latent_shards_dir=a.get("latent_shards_dir"),
        img_root=a.get("img_root"), image_size=int(a.get("image_size", 256)),
        preload=True, load_image=True,
        use_glyph_cond=False,
        skel_latent_shards_dir=(a.get("skel_latent_shards_dir") if use_g else None))
    n_use = min(args.n_batch * args.batch, len(ds))
    dl = DataLoader(Subset(ds, list(range(n_use))), batch_size=args.batch,
                    shuffle=True, num_workers=4, drop_last=True)
    print(f"[data] csv={csv_path} use={n_use}", flush=True)

    # ── REPA ──
    repa = None
    if float(a.get("w_repa", 0)) > 0:
        from src.loss.repa import build_repa_module
        raw = str(a.get("repa_layers", "") or "")
        layers = tuple(int(x) for x in raw.split(",") if x.strip()) or (8,)
        D = model.x_embedder.proj.out_channels  # hidden_size
        repa = build_repa_module(
            student_dim=D, layers=layers,
            w_repa=float(a["w_repa"]),
            warmup_steps=int(a.get("repa_warmup", 0) or 0),
            teacher_ckpt=a.get("repa_teacher_ckpt", "") or None)
        repa.to(dev)
        print(f"[repa] layers={layers} w={a['w_repa']}", flush=True)

    # ── flow ──
    from src.loss.flow_matching import FlowMatching
    diff = FlowMatching(num_steps=int(a.get("eval_steps", 50)), sampler=a.get("flow_sampler", "heun"),
                        t_sampler=a.get("t_sampler", "logit_normal"),
                        t_mean=float(a.get("t_mean", 0.0)), t_std=float(a.get("t_std", 1.0)),
                        shift=float(a.get("shift", 1.0)), use_ot=bool(a.get("use_ot", False)))

    model.train()
    t0 = time.time()
    grp_mods = []   # (name, [modules])
    grp_mods.append(("x_embedder", [model.x_embedder]))
    grp_mods.append(("t_embedder", [model.t_embedder]))
    grp_mods.append(("callig_chain", [model.y_callig_embedder, model.callig_proj]))
    if getattr(model, "char_proj", None) is not None:
        grp_mods.append(("char_chain", [model.y_char_embedder, model.char_proj]))
    grp_mods.append(("glyph_embedder", [model.glyph_embedder]))
    grp_mods.append(("final_layer", [model.final_layer]))
    for i, blk in enumerate(model.blocks):
        grp_mods.append((f"block{i:02d}", [blk]))
    if repa is not None:
        grp_mods.append(("repa", [repa]))

    acc = {name: [0.0, 0.0, 0] for name, _ in grp_mods}   # grad_sq, param_sq, n_batches
    repa_grad_sq, total_grad_sq = 0.0, 0.0
    g_fracs = []     # 有/无 g 输出差
    main_loss_acc, repa_loss_acc = 0.0, 0.0
    n_batches_done = 0

    for bi, batch in enumerate(dl):
        if bi >= args.n_batch:
            break
        x = batch["latent"].to(dev)
        y_callig = batch["y_callig"].to(dev)
        y_char = batch["y_char"].to(dev)
        has_g = use_g and batch.get("skel_latent") is not None
        g = batch["skel_latent"].to(dev).float() if has_g else None

        mk = dict(y_callig=y_callig, y_char=y_char)
        if g is not None:
            mk["g"] = g
        if repa is not None:
            if len(repa.layers) > 1:
                mk["return_intermediate_layers"] = repa.layers
            else:
                mk["return_intermediate_layer"] = repa.layers[0]

        t = torch.rand(x.shape[0], device=dev)
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ld = diff.training_losses(model, x, t, model_kwargs=mk)
            loss_main = ld["loss"]
            inter = ld.get("intermediate_feats", None)
            img = batch.get("image")
            loss_repa = torch.tensor(0.0, device=dev)
            if repa is not None and inter is not None and img is not None:
                loss_repa = repa(inter, img.to(dev), step=10**9)
        (loss_main.mean() + loss_repa).backward()
        main_loss_acc += float(loss_main.mean().item())
        repa_loss_acc += float(loss_repa.item())

        # 组梯度
        for name, mods in grp_mods:
            gs = ps = 0.0
            for m in mods:
                for p in m.parameters():
                    if p.requires_grad:
                        ps += p.float().pow(2).sum().item()
                        g = p.grad
                        gs += (g if g is not None else torch.zeros_like(p)).float().pow(2).sum().item()
            acc[name][0] += gs; acc[name][1] += ps; acc[name][2] += 1

        # 全局 + REPA 梯度
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                total_grad_sq += p.grad.float().pow(2).sum().item()
        if repa is not None:
            for p in repa.parameters():
                if p.requires_grad and p.grad is not None:
                    repa_grad_sq += p.grad.float().pow(2).sum().item()

        n_batches_done += 1
    print(f"[done] {n_batches_done} batches in {time.time()-t0:.0f}s", flush=True)

    # ── 输出 ──
    res = {"n_batches": n_batches_done, "w_repa": float(a.get("w_repa", 0)),
           "repa_layers": a.get("repa_layers", "")}

    # ── g 注入作用 (eval 模式, 干净测量: 有/无 g 输出差) ──
    model.eval()
    if use_g:
        with torch.no_grad():
            try:
                b2 = next(iter(dl))
                xb = b2["latent"][:8].to(dev)
                tb = torch.rand(8, device=dev)
                ycb = b2["y_callig"][:8].to(dev)
                yhb = b2["y_char"][:8].to(dev)
                gb = b2["skel_latent"][:8].to(dev).float()
                if gb.numel() > 0:
                    o_nog = model(xb, tb, ycb, yhb, g=None).float()
                    o_g = model(xb, tb, ycb, yhb, g=gb).float()
                    rel = (o_nog - o_g).norm() / o_g.norm().clamp_min(1e-8)
                    g_fracs.append(float(rel))
                    print(f"\n=== g 注入作用 (eval) ===")
                    print(f"  有/无 g 输出相对差: {rel:.5f} "
                          f"({'条件生效' if rel > 0.02 else '<<< 条件几乎无影响'})")
                    res["g_out_diff"] = round(float(rel), 5)
            except Exception as e:
                print(f"[warn] g 测量失败: {e}", flush=True)

    for name, mods in grp_mods:
        gs, ps, nb = acc[name]
        if nb == 0 or ps == 0:
            continue
        rel = np.sqrt(gs / ps)
        flag = " <<< 死区" if rel < 5e-5 else (" <<< 过激" if rel > 0.5 else "")
        print(f"  {name:<14} grad_norm={np.sqrt(gs/nb):>10.4f}  rel={rel:>8.5f}{flag}")
        res[name] = {"grad_norm": round(float(np.sqrt(gs / nb)), 4),
                     "rel_update": round(float(rel), 6)}
    if total_grad_sq > 0:
        repa_share = repa_grad_sq / total_grad_sq if repa is not None else 0.0
        print(f"\n=== REPA 梯度占比: {repa_share:.5f} ({(repa_share*100):.2f}%) ===")
        res["repa_grad_share"] = round(float(repa_share), 5)
    if n_batches_done > 0:
        res["main_loss_mean"] = round(main_loss_acc / n_batches_done, 5)
        res["repa_loss_mean"] = round(repa_loss_acc / n_batches_done, 5)
        print(f"\n=== loss 数值: main={res['main_loss_mean']} repa={res['repa_loss_mean']} "
              f"(repa/main 数值比 {(repa_loss_acc/main_loss_acc if main_loss_acc else 0):.4f}) ===")
    if g_fracs:
        m = float(np.mean(g_fracs))
        print(f"\n=== g 注入作用: 有/无 g 输出差 = {m:.5f} "
              f"({'条件生效' if m > 0.02 else '<<< 条件几乎无影响'}) ===")
        res["g_out_diff"] = round(m, 5)
    gs = getattr(model.glyph_embedder, "scale", None)
    if gs is not None:
        res["glyph_scale"] = float(gs.item())
        print(f"glyph_scale (可学习?): {gs.item():.4f} requires_grad={gs.requires_grad}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()