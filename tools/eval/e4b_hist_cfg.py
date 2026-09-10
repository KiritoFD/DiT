# -*- coding: utf-8 -*-
"""E4b: 历史 ckpt 用修正后的 CFG 口径重测 (doc 48 §4 P0).

默认: c41x 曲线 6 点 + sty16 的 seen(n=10, 与历史 eval_auto_*.json 同集), cfg 0.7/2.0 对照.
用法: python tools/eval/e4b_hist_cfg.py --ckpts a.pt,b.pt --cfgs 0.7,2.0
"""
import argparse
import json
import sys
import time

import torch as th

sys.path.insert(0, "/root/Workspace/xy/DiT")
import src.eval.gpu_ablate_eval as G


def eval_one(ckpt, cfgs, csv, n, steps, batch, vae_batch):
    th.manual_seed(0)
    ck = th.load(ckpt, map_location="cpu", weights_only=False)
    na = ck.get("args", {})
    a = vars(na) if isinstance(na, argparse.Namespace) else (na or {})
    model = G.build_model(a, "cuda")
    miss, unexp = model.load_state_dict(G.strip(ck.get("ema") or ck.get("model") or ck),
                                        strict=False)
    del ck
    assert len(unexp) == 0, f"unexpected={unexp[:3]}"
    from src.eval.inference import make_eval_cache, load_eval_vae
    from src.utils.callig_map import load_callig_id_map
    cmap = None
    if a.get("callig_id_map"):
        p = a["callig_id_map"]
        if not G.os.path.isabs(p) and not G.os.path.exists(p):
            p = G.os.path.join(G.BASE, p)
        cmap, _ = load_callig_id_map(p)
    cache = make_eval_cache(csv, a.get("img_root"), None, 256, n, 8, 4, 0.18215,
                            skel_latent_shards_dir=a.get("skel_latent_shards_dir"),
                            callig_id_map=cmap)
    noise, conds = cache["noise"], cache["conds"]
    g = cache["skels_latent"].float()
    gts = cache["gts"]
    n_eff = gts.shape[0]
    vae = load_eval_vae("cuda", "pretrained_models/sd-vae-ft-ema")
    tag = G.os.path.basename(G.os.path.dirname(G.os.path.dirname(ckpt))) + "/" + \
        G.os.path.basename(ckpt)
    out = {"ckpt": ckpt, "n": n_eff, "miss": len(miss)}
    for c in cfgs:
        lat = G.heun_gpu(model, noise, conds, c, batch, skel=g, steps=steps,
                         shift=float(a.get("shift", 1.0)), dev="cuda")
        s_sum, i_sum, cnt = 0.0, 0.0, 0
        for i0 in range(0, n_eff, vae_batch):
            i1 = min(i0 + vae_batch, n_eff)
            s, iou, _ = G.decode_metrics(vae, lat[i0:i1], gts[i0:i1], 0.18215, "cuda")
            b = i1 - i0
            s_sum += s * b
            i_sum += iou * b
            cnt += b
        out[f"cfg{c}"] = {"ssim": round(s_sum / cnt, 4), "skel_iou": round(i_sum / cnt, 4)}
        G.log(f"  {tag}  cfg={c}: ssim={s_sum / cnt:.4f} skel_iou={i_sum / cnt:.4f}")
        del lat
        th.cuda.empty_cache()
    del model, vae
    th.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--csv", default="5script/eval_seen_v10.csv")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--vae-batch", type=int, default=10)
    ap.add_argument("--cfgs", default="0.7,2.0")
    ap.add_argument("--out", default="/tmp/e4b_hist.json")
    args = ap.parse_args()
    cfgs = [float(c) for c in args.cfgs.split(",")]
    ckpts = [p.strip() for p in args.ckpts.split(",") if p.strip()]
    results = []
    for ck in ckpts:
        G.log(f"=== {ck}")
        results.append(eval_one(ck, cfgs, args.csv, args.n, args.steps,
                                args.batch, args.vae_batch))
    json.dump(results, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n{'ckpt':60s} " + " ".join(f"cfg{c}" for c in cfgs))
    for r in results:
        base = G.os.path.basename(r["ckpt"])
        line = f"{base:60s}"
        for c in cfgs:
            v = r[f"cfg{c}"]
            line += f"  {v['ssim']:.4f}/{v['skel_iou']:.4f}"
        print(line)
    G.log(f"DONE -> {args.out}")


if __name__ == "__main__":
    main()
