# -*- coding: utf-8 -*-
"""backfill_g_gpu.py — GPU 快速回填: g 条件预训练 ckpt 的 GT 骨架评测 (flat json).

用法: python tools/eval/backfill_g_gpu.py --run-dir assets/results/v10a_skel_cond_pretrain
对 run 下所有缺 eval_auto_{step}.json 的 ckpt: EMA 权重 → GPU 采样 (Heun50, CFG,
g=该样本 GT 实例骨架 latent) → decode → 指标 → flat json (train.py 早停直读).
"""
import os, sys, json, time, glob, argparse
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
os.chdir(_root)
sys.stdout.reconfigure(encoding="utf-8")


def _strip(sd):
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--batch", type=int, default=50)
    args = ap.parse_args()

    import torch
    from src.model import DiT_2Cond_models
    from src.eval.inference import make_eval_cache, load_eval_vae, sample_latents, decode_and_save, compute_metrics

    dev = torch.device("cuda")
    cks = sorted(glob.glob(os.path.join(args.run_dir, "*", "checkpoints", "0*.pt")),
                 key=lambda p: int(os.path.basename(p).split(".")[0]))
    todo = [c for c in cks
            if not os.path.exists(os.path.join(os.path.dirname(c),
                                               f"eval_auto_{int(os.path.basename(c).split('.')[0])}.json"))]
    print(f"ckpts={len(cks)} todo={len(todo)}", flush=True)
    if not todo:
        return

    ck0 = torch.load(todo[0], map_location="cpu", weights_only=False)
    import argparse as _ap
    _na = ck0.get("args")
    a = vars(_na) if isinstance(_na, _ap.Namespace) else (_na or {})
    arch = dict(norm_type=a.get("norm_type", "rms"), mlp_type=a.get("mlp_type", "swiglu"),
                qk_norm=bool(a.get("qk_norm", 1)), rope=bool(a.get("rope", 1)),
                rope_theta=float(a.get("rope_theta", 100.0)), attn_impl=a.get("attn_impl", "sdpa"))
    model = DiT_2Cond_models[a.get("model", "DiT-2Cond-S/2")](
        num_calligraphers=int(a.get("num_calligraphers", 1013)),
        num_characters=int(a.get("num_characters", 35130)),
        condition_fusion=a.get("condition_fusion", "factorized_add"),
        callig_embed_dim=int(a.get("callig_embed_dim", 128)),
        char_embed_dim=int(a.get("char_embed_dim", 384)),
        char_proj_mode=a.get("char_proj_mode", "mlp"),
        freeze_char_table=bool(a.get("freeze_char_table", False)),
        cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
        cond_drop_which_glyph_prob=0.5, use_checkpoint=False, learn_sigma=False,
        use_glyph_cond=bool(a.get("w_glyph_cond", False) or a.get("skel_as_glyph_cond", False)),
        glyph_scale_init=float(a.get("glyph_scale_init", 0.4)),
        glyph_drop_prob=0.0, **arch).to(dev).eval()
    cache = make_eval_cache(a.get("gpu_eval_csv") or a.get("data_csv"),
                            a.get("gpu_eval_img_root") or a.get("img_root"), None, 256,
                            args.n, 8, 4, 0.18215,
                            skel_latent_shards_dir=(a.get("gpu_eval_skel_latent_shards_dir")
                                                    or a.get("skel_latent_shards_dir")))
    g_all = cache["skels_latent"].float()
    hit = int((g_all.view(args.n, -1).sum(1) != 0).sum())
    print(f"GT 骨架覆盖 {hit}/{args.n}", flush=True)
    vae = load_eval_vae(dev, "data/pretrained/sd-vae-ft-ema")
    from src.loss import create_diffusion_or_flow
    flow = create_diffusion_or_flow(str(int(a.get("eval_steps", a.get("gpu_eval_steps", 50)))),
                                    diffusion_type=a.get("diffusion_type", "flow"),
                                    t_sampler=a.get("t_sampler", "logit_normal"),
                                    sampler=a.get("flow_sampler", "heun"),
                                    t_mean=float(a.get("t_mean", 0.0)),
                                    t_std=float(a.get("t_std", 1.0)),
                                    shift=float(a.get("shift", 1.0)))
    cfg = float(a.get("eval_cfg", a.get("gpu_eval_cfg", 0.7)))
    noise, conds = cache["noise"], cache["conds"]

    for ck in todo:
        step = int(os.path.basename(ck).split(".")[0])
        d = torch.load(ck, map_location="cpu", weights_only=False)
        sd = _strip(d.get("ema") or d.get("model") or d.get("delta") or d)
        miss, unexp = model.load_state_dict(sd, strict=False)
        assert len(unexp) == 0, f"unexpected={len(unexp)}"
        t0 = time.time()
        with torch.no_grad():
            lat = sample_latents(model, flow, noise, conds, cfg, args.batch, dev,
                                 skel=g_all, seed=0)
            decode_and_save(vae, lat, 0.18215,
                            os.path.join(os.path.dirname(ck), "eval_samples_ctrl",
                                         f"step{step:07d}", "g"),
                            "g", gts=cache["gts"], vae_batch=32)
            m = compute_metrics(os.path.join(os.path.dirname(ck), "eval_samples_ctrl",
                                             f"step{step:07d}", "g"),
                                os.path.join(os.path.dirname(ck), "eval_samples_ctrl",
                                             f"step{step:07d}", "g"),
                                "g", cache["n"], use_lpips=True)
        flat = {"n": m["n"], "ssim": m["ssim_mean"], "ssim_std": m["ssim_std"],
                "mse": m["mse_mean"], "mse_std": m["mse_std"],
                "lpips": m.get("lpips_mean"), "skel_iou": m["skel_iou_mean"],
                "step": step, "cfg": cfg,
                "ddim_steps": int(a.get("eval_steps", a.get("gpu_eval_steps", 50))),
                "elapsed_s": round(time.time() - t0, 1), "engine": "gpu_bf16_g"}
        out = os.path.join(os.path.dirname(ck), f"eval_auto_{step}.json")
        json.dump(flat, open(out, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"step {step}: {time.time()-t0:.0f}s ssim={flat['ssim']:.4f} "
              f"iou={flat['skel_iou']:.4f}", flush=True)


if __name__ == "__main__":
    main()
