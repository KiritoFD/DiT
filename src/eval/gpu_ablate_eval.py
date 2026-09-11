# -*- coding: utf-8 -*-
"""gpu_ablate_eval.py — GPU 推理时消融: 判定哪些模块可删。

思路: 对已训练 ckpt, 在**推理时**屏蔽某个模块的输出, 看指标掉多少。
  - 指标基本不变 -> 该模块在推理路径上未被使用 -> 可删
  - 指标大幅下降 -> 该模块是必需的 -> 不能删
(注: 这是"是否被使用"的强证据, 不完全等价于"训练时删掉它"的对照,
 但足以发现纯冗余/死重模块, 且成本 = 一次评测而非一次训练。)

配置:
  baseline      : 原样
  no_glyph_add  : glyph_scale=0      (屏蔽输入层 token-add)
  no_xattn      : 12 层 out_proj 置零 (屏蔽逐层 cross-attn 注入)
  no_callig     : y_callig=null      (屏蔽书家条件)
  no_g          : g=None             (屏蔽骨架条件)
  cfg 扫描      : 0.5 / 0.7 / 1.0 / 1.3 / 1.5

用法:
  /opt/conda/envs/cu121/bin/python src/eval/gpu_ablate_eval.py \
      --ckpt <path.pt> --n 20 --steps 50 --out /tmp/ablate.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch as th

BASE = "/root/Workspace/xy/DiT"
sys.path.insert(0, BASE)
os.chdir(BASE)
sys.stdout.reconfigure(encoding="utf-8")


def log(msg):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ── 指标 (CPU numpy, 与 gpu_batch_eval_v2 同实现) ────────────────────────────
from skimage.morphology import skeletonize
from scipy.ndimage import uniform_filter


def ssim_np(a, b, win=7):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    out = []
    for ch in range(3):
        x = a[:, :, ch].astype(np.float64)
        y = b[:, :, ch].astype(np.float64)
        mx, my = uniform_filter(x, win), uniform_filter(y, win)
        mx2, my2, mxy = mx ** 2, my ** 2, mx * my
        sx2 = uniform_filter(x * x, win) - mx2
        sy2 = uniform_filter(y * y, win) - my2
        sxy = uniform_filter(x * y, win) - mxy
        out.append((((2 * mxy + c1) * (2 * sxy + c2))
                    / ((mx2 + my2 + c1) * (sx2 + sy2 + c2))).mean())
    return float(np.mean(out))


def skel_iou(a, b, t=0.5):
    b1, b2 = a.mean(2) < t, b.mean(2) < t
    if not b1.any() and not b2.any():
        return 1.0
    if not b1.any() or not b2.any():
        return 0.0
    s1, s2 = skeletonize(b1), skeletonize(b2)
    u = (s1 | s2).sum()
    return float((s1 & s2).sum() / u) if u > 0 else 1.0


# ── 构建模型 (与 cpu_eval_worker 同一套参数, 含 style_token_n) ───────────────
def build_model(a, device):
    from src.model import DiT_2Cond_models
    arch = dict(norm_type=a.get("norm_type", "rms"),
                mlp_type=a.get("mlp_type", "swiglu"),
                qk_norm=bool(a.get("qk_norm", 1)),
                rope=bool(a.get("rope", 1)),
                rope_theta=float(a.get("rope_theta", 100.0)),
                attn_impl=a.get("attn_impl", "sdpa"))
    m = DiT_2Cond_models[a["model"]](
        num_calligraphers=int(a.get("num_calligraphers", 1013)),
        num_characters=int(a.get("num_characters") or 35130),
        condition_fusion=a.get("condition_fusion", "factorized_add"),
        callig_embed_dim=int(a.get("callig_embed_dim", 128)),
        char_embed_dim=int(a.get("char_embed_dim") or 384),
        char_proj_mode=(a.get("char_proj_mode") or "mlp"),
        freeze_char_table=bool((a.get("freeze_char_table") or False)),
        cond_drop_all_prob=0.0, cond_drop_one_prob=0.0,
        cond_drop_which_glyph_prob=0.5,
        use_checkpoint=False, learn_sigma=False,
        use_glyph_cond=bool(a.get("skel_as_glyph_cond") or a.get("w_glyph_cond")),
        use_char_cond=not bool(a.get("no_char_cond", False)),
        glyph_scale_init=float(a.get("glyph_scale_init", 0.4)),
        glyph_drop_prob=0.0,
        glyph_embedder_depth=int(a.get("glyph_embedder_depth", 0)),
        glyph_inject_layers=int(a.get("glyph_inject_layers", 0)),
        in_channels=(int(a.get("latent_channels", 4))
                     + 4 * len([s for s in str(a.get("aux_latent_shards_dirs", "") or "").split(",") if s])),
        **arch)
    if a.get("freeze_callig_table"):
        m.y_callig_embedder.freeze_table()
    return m.to(device).eval()


def strip(sd):
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}


# ── GPU Heun 采样 (复刻 cpu_sampler, 张量全部上 device) ───────────────────────
@th.no_grad()
def heun_gpu(model, noise, conds, cfg, batch, skel=None, steps=50, shift=1.0,
             dev="cuda"):
    n = noise.shape[0]
    # aux 目标通道: 噪声扩到模型 in_channels
    _tgt = int(getattr(model, "in_channels", noise.shape[1]))
    if noise.shape[1] < _tgt:
        _extra = th.randn(n, _tgt - noise.shape[1], *noise.shape[2:],
                          dtype=noise.dtype, device=noise.device)
        noise = th.cat([noise, _extra], dim=1)
    out = th.zeros(n, *noise.shape[1:], dtype=th.float32)
    s = th.linspace(1.0, 0.0, steps + 1, dtype=th.float64)
    ts = (shift * s / (1.0 + (shift - 1.0) * s)).tolist() if shift != 1.0 else s.tolist()
    use_cfg = bool(cfg) and cfg > 0
    for i0 in range(0, n, batch):
        i1 = min(i0 + batch, n)
        b = i1 - i0
        x = noise[i0:i1].to(dev).float()
        mk = dict(y_callig=th.tensor([c[0] for c in conds[i0:i1]], device=dev, dtype=th.long),
                  y_char=th.tensor([c[1] for c in conds[i0:i1]], device=dev, dtype=th.long))
        if skel is not None:
            mk["g"] = skel[i0:i1].to(dev).float()
        for k in range(steps):
            t_i, t_nx = ts[k], ts[k + 1]
            dt = t_nx - t_i
            t1 = th.full((b,), t_i, device=dev)
            v1 = model.forward_with_cfg(x, t1 * 1000.0, cfg_scale=cfg, **mk) if use_cfg \
                else model(x, t1 * 1000.0, **mk)
            if isinstance(v1, tuple):
                v1 = v1[0]
            x_e = x + dt * v1
            t2 = th.full((b,), t_nx, device=dev)
            v2 = model.forward_with_cfg(x_e, t2 * 1000.0, cfg_scale=cfg, **mk) if use_cfg \
                else model(x_e, t2 * 1000.0, **mk)
            if isinstance(v2, tuple):
                v2 = v2[0]
            x = x + dt * 0.5 * (v1 + v2)
        out[i0:i1] = x.float().cpu()
    return out


@th.no_grad()
def decode_metrics(vae, lat, gts, sf, dev="cuda"):
    if lat.shape[1] > 4:                 # aux 目标通道: 只解码图像 4 通道
        lat = lat[:, :4]
    dec = vae.decode((lat.to(dev).float() / sf)).sample.float().cpu()
    dn = ((dec.clamp(-1, 1) + 1) / 2).numpy()
    gn = ((gts.clamp(-1, 1) + 1) / 2).numpy()
    ss, ii = [], []
    for i in range(dn.shape[0]):
        p, g = dn[i].transpose(1, 2, 0), gn[i].transpose(1, 2, 0)
        ss.append(ssim_np(p, g))
        ii.append(skel_iou(p, g))
    return float(np.mean(ss)), float(np.mean(ii)), float(np.std(ss))


# ── 消融干预 ────────────────────────────────────────────────────────────────
class Ablation:
    """推理时屏蔽某模块; 退出时恢复。"""

    def __init__(self, model, mode):
        self.model, self.mode, self.saved = model, mode, []

    def __enter__(self):
        m = self.model
        if self.mode == "no_glyph_add":
            self.saved.append(("glyph_scale", float(m.glyph_scale.data)))
            m.glyph_scale.data.fill_(0.0)
        elif self.mode == "no_xattn":
            for inj in (m.glyph_injections or []):
                self.saved.append((inj, inj.out_proj.weight.data.clone(),
                                   inj.out_proj.bias.data.clone()))
                inj.out_proj.weight.data.zero_()
                inj.out_proj.bias.data.zero_()
        return self

    def __exit__(self, *exc):
        m = self.model
        for s in self.saved:
            if s[0] == "glyph_scale":
                m.glyph_scale.data.fill_(s[1])
            else:
                s[0].out_proj.weight.data.copy_(s[1])
                s[0].out_proj.bias.data.copy_(s[2])
        self.saved = []
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=0.7)
    ap.add_argument("--out", default="/tmp/gpu_ablate.json")
    args = ap.parse_args()

    dev = "cuda"
    th.manual_seed(0)
    log(f"ckpt={args.ckpt}")
    ck = th.load(args.ckpt, map_location="cpu", weights_only=False)
    na = ck.get("args", {})
    a = vars(na) if isinstance(na, argparse.Namespace) else (na or {})
    model = build_model(a, dev)
    sd = strip(ck.get("ema") or ck.get("model") or ck)
    miss, unexp = model.load_state_dict(sd, strict=False)
    log(f"load: missing={len(miss)} unexpected={len(unexp)}")
    del ck
    # 关键参数回显
    log(f"inject_layers={a.get('glyph_inject_layers')}")
    log(f"learned glyph_scale = {float(model.glyph_scale.data):.4f} "
        f"(init {a.get('glyph_scale_init')})")

    from src.eval.inference import make_eval_cache, load_eval_vae
    from src.utils.callig_map import load_callig_id_map
    cmap = None
    if a.get("callig_id_map"):
        p = a["callig_id_map"]
        if not os.path.isabs(p) and not os.path.exists(p):
            p = os.path.join(BASE, p)
        cmap, _ = load_callig_id_map(p)
    cache = make_eval_cache(a.get("eval_csv"), a.get("img_root"), None, 256, args.n,
                            8, 4, 0.18215,
                            skel_latent_shards_dir=a.get("skel_latent_shards_dir"),
                            callig_id_map=cmap)
    noise = cache["noise"]
    conds = cache["conds"]
    g = cache["skels_latent"].float()
    gts = cache["gts"]
    n_eff = gts.shape[0]
    log(f"eval samples n={n_eff}, skel hit={int((g.view(n_eff, -1).sum(1) != 0).sum())}/{n_eff}, "
        f"cfg={args.cfg}, steps={args.steps}")

    vae = load_eval_vae(dev, "pretrained_models/sd-vae-ft-ema")
    sf = 0.18215
    null_callig = model.y_callig_embedder.num_classes
    null_char = model.y_char_embedder.num_classes if model.y_char_embedder is not None else 0

    results = {}

    def run(tag, cfg=None, skel="std", conds_use=None, abl=None):
        t0 = time.time()
        c = cfg if cfg is not None else args.cfg
        sk = g if skel == "std" else None      # "zero" -> 不传 g (屏蔽骨架条件)
        cd = conds if conds_use is None else conds_use
        ctx = Ablation(model, abl) if abl else None
        if ctx:
            ctx.__enter__()
        try:
            lat = heun_gpu(model, noise, cd, c, args.batch, skel=sk,
                           steps=args.steps, shift=float(a.get("shift", 1.0)), dev=dev)
            s, iou, sstd = decode_metrics(vae, lat, gts, sf, dev)
        finally:
            if ctx:
                ctx.__exit__()
        dt = time.time() - t0
        results[tag] = {"ssim": round(s, 4), "skel_iou": round(iou, 4),
                        "ssim_std": round(sstd, 4), "cfg": c,
                        "n_g": "zero" if skel == "zero" else "std",
                        "sec": round(dt, 1)}
        log(f"  {tag:22s} ssim={s:.4f} (+-{sstd:.4f})  skel_iou={iou:.4f}  ({dt:.0f}s)")
        th.cuda.empty_cache()

    log("=== 消融 (cfg=%.2f) ===" % args.cfg)
    run("baseline")
    run("no_glyph_add", abl="no_glyph_add")
    run("no_xattn", abl="no_xattn")
    run("no_callig", conds_use=[(null_callig, c[1]) for c in conds])
    run("no_g", skel="zero")
    run("no_callig_no_g", skel="zero",
        conds_use=[(null_callig, c[1]) for c in conds])

    log("=== CFG 扫描 (上探上限) ===")
    for c in (1.5, 2.0, 2.5, 3.0, 4.0):
        run(f"cfg={c}", cfg=c)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"ckpt": args.ckpt, "n": n_eff, "steps": args.steps,
                   "args": {k: a.get(k) for k in
                            ("model", "glyph_inject_layers",
                             "glyph_scale_init", "w_repa", "cond_drop_all_prob",
                             "cond_drop_one_prob", "eval_cfg")},
                   "learned_glyph_scale": float(model.glyph_scale.data),
                   "results": results}, f, ensure_ascii=False, indent=2)
    log(f"DONE -> {args.out}")

    # 汇总表
    print("\n" + "=" * 66)
    print(f"{'config':24s} {'ssim':>8s} {'skel_iou':>9s} {'vs base':>9s}")
    print("-" * 66)
    base = results["baseline"]["ssim"]
    for k, v in results.items():
        print(f"{k:24s} {v['ssim']:8.4f} {v['skel_iou']:9.4f} "
              f"{v['ssim'] - base:+9.4f}")
    print("=" * 66)


if __name__ == "__main__":
    main()
