# -*- coding: utf-8 -*-
"""E4a: strict 集 (n=237) CFG 复核 — 确认 cfg 峰值 (doc 48 §2 的 n=10 结论).

复用 src/eval/gpu_ablate_eval 的构建/采样/指标函数, 只把评测集换成 strict.
用法: python tools/eval/e4a_cfg_sweep.py --ckpt <path> --cfgs 0.7,1.3,1.5,2.0,2.5,3.0
"""
import argparse
import json
import sys
import time

import torch as th

sys.path.insert(0, "/root/Workspace/xy/DiT")
import src.eval.legacy.gpu_ablate_eval as G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", default="assets/eval_fame3_strict_clean_v9.csv")
    ap.add_argument("--n", type=int, default=237)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--vae-batch", type=int, default=24)
    ap.add_argument("--cfgs", default="0.7,1.3,1.5,2.0,2.5,3.0")
    ap.add_argument("--out", default="/tmp/e4a_strict_cfg.json")
    args = ap.parse_args()
    cfgs = [float(c) for c in args.cfgs.split(",")]

    dev = "cuda"
    th.manual_seed(0)
    G.log(f"ckpt={args.ckpt}")
    ck = th.load(args.ckpt, map_location="cpu", weights_only=False)
    na = ck.get("args", {})
    a = vars(na) if isinstance(na, argparse.Namespace) else (na or {})
    model = G.build_model(a, dev)
    miss, unexp = model.load_state_dict(G.strip(ck.get("ema") or ck.get("model") or ck),
                                        strict=False)
    G.log(f"load: missing={len(miss)} unexpected={len(unexp)} "
          f"inject_mode={a.get('glyph_inject_mode')} layers={a.get('glyph_inject_layers')}")
    del ck

    from src.eval.inference import make_eval_cache, load_eval_vae
    from src.utils.callig_map import load_callig_id_map
    cmap = None
    if a.get("callig_id_map"):
        p = a["callig_id_map"]
        if not G.os.path.isabs(p) and not G.os.path.exists(p):
            p = G.os.path.join(G.BASE, p)
        cmap, _ = load_callig_id_map(p)
    cache = make_eval_cache(args.csv, a.get("img_root"), None, 256, args.n, 8, 4, 0.18215,
                            skel_latent_shards_dir=a.get("skel_latent_shards_dir"),
                            callig_id_map=cmap)
    noise, conds = cache["noise"], cache["conds"]
    g = cache["skels_latent"].float()
    gts = cache["gts"]
    n_eff = gts.shape[0]
    G.log(f"strict n={n_eff}, skel hit={int((g.view(n_eff, -1).sum(1) != 0).sum())}/{n_eff}")

    vae = load_eval_vae(dev, "data/pretrained/sd-vae-ft-ema")
    results = {}
    for c in cfgs:
        t0 = time.time()
        lat = G.heun_gpu(model, noise, conds, c, args.batch, skel=g,
                         steps=args.steps, shift=float(a.get("shift", 1.0)), dev=dev)
        # 分批 decode (n=237 一次性 decode 会 OOM; gpu_ablate 原版只面向 n=10)
        s_sum, i_sum, s2_sum, cnt = 0.0, 0.0, 0.0, 0
        for i0 in range(0, n_eff, args.vae_batch):
            i1 = min(i0 + args.vae_batch, n_eff)
            s, iou, sstd = G.decode_metrics(vae, lat[i0:i1], gts[i0:i1], 0.18215, dev)
            b = i1 - i0
            s_sum += s * b
            i_sum += iou * b
            s2_sum += (sstd ** 2 + s ** 2) * b
            cnt += b
        s = s_sum / cnt
        iou = i_sum / cnt
        sstd = (max(s2_sum / cnt - s * s, 0.0)) ** 0.5
        dt = time.time() - t0
        results[c] = {"ssim": round(s, 4), "skel_iou": round(iou, 4),
                      "ssim_std": round(sstd, 4), "sec": round(dt, 1)}
        G.log(f"  cfg={c:<4} ssim={s:.4f} (+-{sstd:.4f})  skel_iou={iou:.4f}  ({dt:.0f}s)")
        del lat
        th.cuda.empty_cache()

    json.dump({"ckpt": args.ckpt, "n": n_eff, "steps": args.steps, "results": results},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    base = results[cfgs[0]]["skel_iou"]
    print("\ncfg      ssim     skel_iou   vs cfg_first (iou)")
    for c in cfgs:
        r = results[c]
        print(f"{c:<8} {r['ssim']:.4f}   {r['skel_iou']:.4f}     {r['skel_iou'] - base:+.4f}")
    G.log(f"DONE -> {args.out}")


if __name__ == "__main__":
    main()
