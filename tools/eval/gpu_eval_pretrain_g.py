#!/usr/bin/env python3
"""gpu_eval_pretrain_g.py — GPU 版 pretrain_g eval (与 CPU daemon flat 协议一致)。

对 v10a/v10b/v10a-dino 等 train.py 预训练 ckpt 跑 g 单臂评测:
  * 模型从 ckpt args 重建 (含 use_std_dino_char_embedder / skel_as_glyph_cond 透传)
  * g 条件 = GT 实例骨架 latent (默认, 与训练条件域/历史曲线可比) 或 std_glyph
  * 采样 GPU sample_latents (use_glyph_cond 自动路由 'g')
  * 输出 {ckpt_dir}/eval_auto_{step}.json (flat, 与 daemon 同格式) + eval_samples_ctrl PNG

用法:
  python tools/eval/gpu_eval_pretrain_g.py --ckpt_dir DIR [--steps 10k|all] [--g-source gt_skel|std_glyph] [--n 100]
"""
import os, sys, csv, time, argparse, json, glob
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
os.chdir(_root)
sys.stdout.reconfigure(encoding="utf-8")

import torch
import numpy as np


def _strip(sd):
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True, help="含 *.pt 的 checkpoints 目录")
    ap.add_argument("--steps", default="all",
                    help="'all' 或逗号分隔 (如 '12500,25000,50000,75000,80000')")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--cfg", type=float, default=0.7)
    ap.add_argument("--ddim-steps", type=int, default=50)
    ap.add_argument("--dit-batch", type=int, default=8)
    ap.add_argument("--vae-batch", type=int, default=16)
    ap.add_argument("--g-source", choices=["gt_skel", "std_glyph"], default="gt_skel")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda")
    cks = sorted(glob.glob(os.path.join(args.ckpt_dir, "*.pt")),
                 key=lambda p: int(os.path.basename(p).split(".")[0]))
    if args.steps != "all":
        want = {int(s) for s in args.steps.split(",")}
        cks = [c for c in cks if int(os.path.basename(c).split(".")[0]) in want]
    print(f"ckpts: {len(cks)}  {[os.path.basename(c).split('.')[0] for c in cks]}", flush=True)

    from src.model import DiT_2Cond_models
    from src.eval.inference import (make_eval_cache, load_eval_vae, decode_and_save,
                                    compute_metrics, sample_latents, build_diffusion)

    # 首 ckpt 定 cache (csv/img_root/skel shards 相同)
    ck0 = torch.load(cks[0], map_location="cpu", weights_only=False)
    a = ck0.get("args", {}) or {}
    if isinstance(a, argparse.Namespace):
        a = vars(a)
    csv_path = a.get("gpu_eval_csv") or a.get("eval_csv") or a.get("data_csv")
    img_root = a.get("gpu_eval_img_root") or a.get("img_root")
    shards = (a.get("gpu_eval_skel_latent_shards_dir")
              or a.get("skel_latent_shards_dir") or None)
    n = args.n
    print(f"eval csv={csv_path} img_root={img_root} skel_shards={shards} n={n}", flush=True)
    cache = make_eval_cache(csv_path, img_root, None, 256, n, 8, 4, 0.18215,
                            skel_latent_shards_dir=shards)
    noise, conds = cache["noise"], cache["conds"]
    vae = load_eval_vae(dev, "data/pretrained/sd-vae-ft-ema")
    diff = build_diffusion(args.ddim_steps, "flow")

    arch = dict(norm_type=a.get("norm_type", "rms"), mlp_type=a.get("mlp_type", "swiglu"),
                qk_norm=bool(a.get("qk_norm", 1)), rope=bool(a.get("rope", 1)),
                rope_theta=float(a.get("rope_theta", 100.0)),
                attn_impl=a.get("attn_impl", "sdpa"))
    use_g = bool(a.get("w_glyph_cond", False) or a.get("skel_as_glyph_cond", False))

    # g 条件
    if args.g_source == "gt_skel":
        g_all = cache["skels_latent"].float()
        hit = int((g_all.view(n, -1).sum(1) != 0).sum())
        print(f"GT 实例骨架覆盖 {hit}/{n}", flush=True)
    else:
        from src.utils import get_glyph_lookup_v2
        lk = get_glyph_lookup_v2()
        rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))[:n]
        g_all = torch.zeros(n, 4, 32, 32)
        hit = 0
        for i, r in enumerate(rows):
            gv = lk.get(int(r["script_id"]), r.get("character", ""), random=False)
            if gv is not None:
                g_all[i] = gv.float(); hit += 1
        print(f"glyph 库命中 {hit}/{n} (部署态)", flush=True)

    for ck_path in cks:
        step = int(os.path.basename(ck_path).split(".")[0])
        out_json = os.path.join(args.ckpt_dir, f"eval_auto_{step}.json")
        if args.skip_existing and os.path.isfile(out_json):
            print(f"skip {step} (已有)", flush=True)
            continue
        t0 = time.time()
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        a2 = ck.get("args", {}) or {}
        if isinstance(a2, argparse.Namespace):
            a2 = vars(a2)
        model = DiT_2Cond_models[a2.get("model", "DiT-2Cond-S/2")](
            num_calligraphers=int(a2.get("num_calligraphers", 1013)),
            num_characters=int(a2.get("num_characters", 35130)),
            condition_fusion=a2.get("condition_fusion", "factorized_add"),
            callig_embed_dim=int(a2.get("callig_embed_dim", 128)),
            char_embed_dim=int(a2.get("char_embed_dim", 384)),
            char_proj_mode=a2.get("char_proj_mode", "mlp"),
            freeze_char_table=bool(a2.get("freeze_char_table", True)),
            cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
            cond_drop_which_glyph_prob=0.5, use_checkpoint=False, learn_sigma=False,
            use_glyph_cond=use_g,
            use_char_cond=not bool(a2.get("no_char_cond", False)),
            use_std_dino_char_embedder=bool(a2.get("use_std_dino_char_embedder", False)),
            std_dino_table_path=a2.get("std_dino_table_path"),
            glyph_scale_init=float(a2.get("glyph_scale_init", 0.4)),
            glyph_drop_prob=float(a2.get("glyph_drop_prob", 0.0)),
            glyph_inject_layers=int(a2.get("glyph_inject_layers", 0)), **arch).to(dev)
        sd = _strip(ck.get("ema") or ck.get("model") or ck)
        miss, unexp = model.load_state_dict(sd, strict=False)
        assert len(unexp) == 0, f"step {step}: unexpected={len(unexp)}"
        if len(miss) > 0:
            print(f"  warn missing={len(miss)} (首几个: {list(miss)[:5]})", flush=True)
        model.eval()

        # g 单臂采样 (GPU)
        step_tag = f"step{step:07d}"
        arm_dir = os.path.join(args.ckpt_dir, "..", "eval_samples_ctrl", step_tag, "g")
        arm_dir = os.path.normpath(arm_dir)
        os.makedirs(arm_dir, exist_ok=True)
        t1 = time.time()
        lat = sample_latents(model, diff, noise, conds, args.cfg, args.dit_batch,
                             dev, skel=g_all, seed=0)
        t_s = time.time() - t1
        decode_and_save(vae, lat, 0.18215, arm_dir, "g",
                        gts=cache["gts"], vae_batch=args.vae_batch)
        m, lists = compute_metrics(arm_dir, arm_dir, "g", n, use_lpips=True,
                                   idx_range=(0, n), with_lists=True)
        flat = {"step": step, "n": n, "cfg": args.cfg, "ddim_steps": args.ddim_steps,
                "engine": "gpu_eval_pretrain_g", "g_source": args.g_source,
                "ssim": m.get("ssim_mean"), "ssim_std": m.get("ssim_std"),
                "lpips": m.get("lpips_mean"), "mse": m.get("mse_mean"),
                "mse_std": m.get("mse_std"), "skel_iou": m.get("skel_iou_mean"),
                "elapsed_s": round(time.time() - t0, 1)}
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(flat, f, ensure_ascii=False, indent=1)
        print(f"[{step_tag}] ssim={flat['ssim']:.4f} lpips={flat['lpips']:.4f} "
              f"sample={t_s:.0f}s total={time.time()-t0:.0f}s -> {os.path.basename(out_json)}",
              flush=True)
        del lat, model
        torch.cuda.empty_cache()
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()