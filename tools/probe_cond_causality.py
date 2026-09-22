# -*- coding: utf-8 -*-
"""
probe_cond_causality.py — D6：**条件因果性**测试。

要回答的问题
------------
D3 已证：换书家的像素差异 ≈ 换噪声（下界为负）。
但那可能有两种解释：
  (a) 模型**根本没学会**书家条件 -> 换任何 id 都没区别（通路失效）
  (b) 模型学会了，但**表达幅度极小** -> 有区别，只是淹没在噪声里

本探针用**同一张图、同一噪声、同一骨架**，只改书家 id，测：
  1. ★ **null / uncond 对照**：把书家换成 null（id = num_calligraphers）
     - 若"换真实书家"与"换成 null"的差异**一样大** -> 模型没把书家当条件 (a)
     - 若"换成 null"差异**明显更大** -> 模型知道"有/无条件"，但分不清具体书家
  2. ★ **嵌入空间距离 vs 输出空间距离**的相关性
     - 若 ‖Δy_emb‖ 大但 ‖Δimg‖ 小 -> 通路衰减（真·瓶颈在注入）
  3. ★ **t 扫描**：在不同 timestep 上做同一对比
     - 风格通常在 t 中段起作用；若全 t 都无差异 -> 条件彻底失效
  4. **随机 id 对照**：换成一个随机书家 vs 换成相邻 id
     - 排除"只有特定 id 对才有反应"

判据
----
  ratio_null = D(换书家) / D(换null)
     >> 1  -> 模型学会区分书家（好）
     ~ 1   -> 只知道"有/无条件"，具体书家无信息
     < 1   -> 换书家比换 null 差异还小（严重，说明条件被忽略）

用法
----
  python tools/probe_cond_causality.py \
      --ckpt <best.pt> --eval-csv assets/eval_v13_strict.csv \
      --n-chars 6 --steps 50 --cfg 1.0 --out assets/d6_<tag>.json
"""
import os, sys, json, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch as th


def log(m):
    print(m, flush=True)


def _strip(sd_):
    out = {}
    for k, v in sd_.items():
        kk = k
        ch = True
        while ch:
            ch = False
            for pfx in ("module.", "_orig_mod.", "model."):
                if kk.startswith(pfx):
                    kk = kk[len(pfx):]
                    ch = True
        out[kk] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval-csv", default="assets/eval_v13_strict.csv")
    ap.add_argument("--img-root", default="")
    ap.add_argument("--skel-shards", default="")
    ap.add_argument("--callig-id-map", default="")
    ap.add_argument("--callig-script-map", default="")
    ap.add_argument("--n-chars", type=int, default=6)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--t-probe", default="0.3,0.5,0.7,0.9",
                    help="单步前向探测用的 t 列表（归一化 0-1）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    a.image_size, a.vae_downscale, a.latent_channels = 256, 8, 4

    import csv
    from collections import defaultdict
    from types import SimpleNamespace
    from src.eval.model_io import build_model_from_args
    from src.eval.inference import load_eval_vae, make_eval_cache

    dev = th.device(a.device if th.cuda.is_available() else "cpu")

    # ---- 载 ckpt / 配置（复用 model_io，别手搓参数字典）
    ck = th.load(a.ckpt, map_location="cpu", weights_only=False)
    ckargs = ck.get("args")
    if ckargs is None:
        raise SystemExit("ckpt 里没有 args")
    gv = (lambda k, d=None: getattr(ckargs, k, d))
    cfg = None
    for cand in (os.path.join(os.path.dirname(os.path.dirname(a.ckpt)), "resolved_config.json"),):
        if os.path.isfile(cand):
            cfg = json.load(open(cand, encoding="utf-8"))
    if cfg:
        merged = dict(cfg.get("args", cfg))
        for k, v in vars(ckargs).items():
            merged.setdefault(k, v)
        ns = SimpleNamespace(**merged)
    else:
        ns = SimpleNamespace(**vars(ckargs))
    model = build_model_from_args(ns, dev)
    sd = ck.get("ema") or ck.get("delta")
    if sd is None:
        raise SystemExit("ckpt 里没有 ema/delta")
    sd = _strip(sd)
    miss = model.load_state_dict(sd, strict=False)
    log("[D6] load: missing=%d unexpected=%d"
        % (len(miss.missing_keys), len(miss.unexpected_keys)))
    model = model.eval()

    num_callig = int(getattr(model, "y_callig_embedder").num_classes)
    log("[D6] num_calligraphers(含 null) = %d" % num_callig)
    NULL_ID = num_callig - 1

    # ---- VAE / cache
    vae = load_eval_vae("cpu", "data/pretrained/sd-vae-ft-ema")
    sf = float(gv("vae_scaling_factor", 0.18215) or 0.18215)

    shards = a.skel_shards or gv("skel_latent_shards_dir", "")
    if shards and not os.path.isdir(shards):
        alt = os.path.join("data/skel", os.path.basename(shards))
        if os.path.isdir(alt):
            shards = alt
    cmap = None
    p1 = a.callig_id_map or gv("callig_id_map", "")
    if p1:
        if not os.path.isabs(p1) and not os.path.exists(p1):
            p1 = os.path.join(ROOT, p1)
        if os.path.exists(p1):
            cmap = json.load(open(p1, encoding="utf-8"))
    csm = None
    p2 = a.callig_script_map or gv("callig_script_map", "")
    if p2:
        if not os.path.isabs(p2) and not os.path.exists(p2):
            p2 = os.path.join(ROOT, p2)
        if os.path.exists(p2):
            from src.utils.callig_script_map import load_callig_script_map
            csm = load_callig_script_map(p2)

    # ---- 选字：取 strict 集里 glyph_id 有 skel 覆盖的（同 D3）
    import glob as _glob
    skel_ids = None
    if shards and os.path.isdir(shards):
        skel_ids = set()
        for _f in _glob.glob(os.path.join(shards, "shard_*.npz")):
            try:
                _z = np.load(_f, allow_pickle=True)
                skel_ids.update([str(x) for x in _z["img_ids"]])
            except Exception:
                pass
    rows_all = list(csv.DictReader(open(a.eval_csv, encoding="utf-8")))
    id_field = "glyph_id" if "glyph_id" in rows_all[0] else "old_50k_id"
    by_char = defaultdict(list)
    for i, r in enumerate(rows_all):
        if skel_ids is not None and str(r.get(id_field, "")) not in skel_ids:
            continue
        by_char[r["character"]].append(i)
    picked = [(ch, idxs[0]) for ch, idxs in sorted(by_char.items())][:a.n_chars]
    log("[D6] 选中 %d 个字: %s" % (len(picked), ", ".join(c for c, _ in picked)))
    if not picked:
        raise SystemExit("没选到字")

    tmp = os.path.join("/tmp", "_d6_rows.csv")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_all[0].keys()))
        w.writeheader()
        for _, i in picked:
            w.writerow(rows_all[i])
    cache = make_eval_cache(tmp, a.img_root or None, None, a.image_size, len(picked),
                            a.vae_downscale, a.latent_channels, sf,
                            skel_latent_shards_dir=(shards or None),
                            callig_id_map=cmap, callig_script_map=csm)
    hier = cache.get("hier_conds")

    def sample(noise_lat, cid, char_id, g_lat, hier_idx=None):
        n = noise_lat.shape[0]
        mk = dict(y_callig=th.tensor([cid] * n, device=dev, dtype=th.long),
                  y_char=th.tensor([char_id] * n, device=dev, dtype=th.long))
        if g_lat is not None and getattr(model, "use_glyph_cond", False):
            mk["g"] = g_lat.to(dev).float()
        if hier_idx is not None and getattr(model, "style_hier", None) is not None:
            mk["y_callig_raw"] = th.tensor([hier_idx[0]] * n, device=dev, dtype=th.long)
            mk["y_pair"] = th.tensor([hier_idx[1]] * n, device=dev, dtype=th.long)
            mk["y_script"] = th.tensor([hier_idx[2]] * n, device=dev, dtype=th.long)
        shift = float(gv("shift", 1.0) or 1.0)
        s = th.linspace(1.0, 0.0, a.steps + 1, dtype=th.float64)
        ts = (shift * s / (1.0 + (shift - 1.0) * s)).tolist() if shift != 1.0 else s.tolist()
        with th.no_grad():
            x = noise_lat.to(dev).float()
            for k in range(a.steps):
                t_i, t_nx = ts[k], ts[k + 1]
                dt = t_nx - t_i
                v1 = model.forward_with_cfg(x, th.full((n,), t_i, device=dev) * 1000.0,
                                            cfg_scale=a.cfg, **mk)
                if isinstance(v1, tuple):
                    v1 = v1[0]
                x_e = x + dt * v1
                v2 = model.forward_with_cfg(x_e, th.full((n,), t_nx, device=dev) * 1000.0,
                                            cfg_scale=a.cfg, **mk)
                if isinstance(v2, tuple):
                    v2 = v2[0]
                x = x + dt * 0.5 * (v1 + v2)
            img = vae.decode(x.float().cpu() / sf).sample
        return ((img.clamp(-1, 1) + 1) / 2)[0].float().mean(axis=0).numpy()

    # ══ 试验 1：换书家 vs 换 null ═══════════════════════════════════════════
    res_pairs = []
    for ci, (ch, i) in enumerate(picked):
        cond = cache["conds"][i]
        cid_true = int(cond[0])
        char_id = int(cond[1])
        noise = cache["noise"][i:i + 1].clone()
        g0 = cache["skels_latent"][i:i + 1] if cache.get("skels_latent") is not None else None
        hi = None
        if hier:
            hi = (hier[0][i], hier[1][i], hier[2][i])

        im_true = sample(noise, cid_true, char_id, g0, hi)
        im_null = sample(noise, NULL_ID, char_id, g0, None)
        # 另一个真实书家（同集内最远的 id）
        alt_cids = [int(c[0]) for c in cache["conds"]
                    if int(c[0]) != cid_true and int(c[0]) != NULL_ID]
        cid_alt = alt_cids[(ci + 1) % len(alt_cids)] if alt_cids else cid_true
        idx_alt = [j for j in range(len(cache["conds"])) if int(cache["conds"][j][0]) == cid_alt]
        hi_alt = None
        if hier and idx_alt:
            j = idx_alt[0]
            hi_alt = (hier[0][j], hier[1][j], hier[2][j])
        im_alt = sample(noise, cid_alt, char_id, g0, hi_alt)
        # 纯噪声地板
        im_n1 = sample(th.randn_like(noise), cid_true, char_id, g0, hi)
        im_n2 = sample(th.randn_like(noise), cid_true, char_id, g0, hi)

        D_style = float(np.abs(im_true - im_alt).mean())
        D_null = float(np.abs(im_true - im_null).mean())
        D_noise = float(np.abs(im_n1 - im_n2).mean())
        res_pairs.append({"char": ch, "cid_true": cid_true, "cid_alt": cid_alt,
                          "D_style": D_style, "D_null": D_null, "D_noise": D_noise,
                          "ratio_null": D_style / max(D_null, 1e-9),
                          "ratio_noise": D_style / max(D_noise, 1e-9)})
        log("  [%d/%d] %s  换书家=%.5f  换null=%.5f  换噪声=%.5f  | null比=%.2f  噪声比=%.2f"
            % (ci + 1, len(picked), ch, D_style, D_null, D_noise,
               res_pairs[-1]["ratio_null"], res_pairs[-1]["ratio_noise"]))

    def m(rs, k):
        v = [r[k] for r in rs]
        return float(np.mean(v)) if v else float("nan")

    # ══ 试验 2：t 扫描（单步前向，看条件在不同 t 的作用幅度）════════════════
    #   用 x=0（纯噪声起点）与固定条件，比较「换书家」和「换null」的
    #   速度场差异随 t 的变化 —— 找模型在哪个 t 最"在意"书家。
    t_probe = [float(x) for x in a.t_probe.split(",") if x.strip()]
    t_rows = []
    i0 = picked[0][1]
    cond = cache["conds"][i0]
    cid_true, char_id = int(cond[0]), int(cond[1])
    g0 = cache["skels_latent"][i0:i0 + 1] if cache.get("skels_latent") is not None else None
    x0 = cache["noise"][i0:i0 + 1].clone().to(dev).float()
    for tv in t_probe:
        mk_a = dict(y_callig=th.tensor([cid_true], device=dev, dtype=th.long),
                    y_char=th.tensor([char_id], device=dev, dtype=th.long))
        mk_b = dict(y_callig=th.tensor([NULL_ID], device=dev, dtype=th.long),
                    y_char=th.tensor([char_id], device=dev, dtype=th.long))
        if g0 is not None and getattr(model, "use_glyph_cond", False):
            mk_a["g"] = g0.to(dev).float()
            mk_b["g"] = g0.to(dev).float()
        if hier:
            mk_a.update(dict(y_callig_raw=th.tensor([hier[0][i0]], device=dev, dtype=th.long),
                             y_pair=th.tensor([hier[1][i0]], device=dev, dtype=th.long),
                             y_script=th.tensor([hier[2][i0]], device=dev, dtype=th.long)))
        with th.no_grad():
            tt = th.tensor([tv * 1000.0], device=dev)
            vA = model.forward_with_cfg(x0, tt, cfg_scale=a.cfg, **mk_a)
            vB = model.forward_with_cfg(x0, tt, cfg_scale=a.cfg, **mk_b)
        vA = vA[0] if isinstance(vA, tuple) else vA
        vB = vB[0] if isinstance(vB, tuple) else vB
        dv = float((vA - vB).abs().mean())
        nv = float(vA.abs().mean())
        t_rows.append({"t": tv, "d_cond": dv, "v_norm": nv,
                       "rel": dv / max(nv, 1e-9)})
        log("  t=%.2f  条件引起的速度差=%.5f  速度幅度=%.5f  相对=%.3f%%"
            % (tv, dv, nv, 100 * t_rows[-1]["rel"]))

    summ = {
        "ckpt": os.path.basename(a.ckpt), "cfg": a.cfg, "n_chars": len(picked),
        "D_style": m(res_pairs, "D_style"),
        "D_null": m(res_pairs, "D_null"),
        "D_noise": m(res_pairs, "D_noise"),
        "ratio_null": m(res_pairs, "ratio_null"),
        "ratio_noise": m(res_pairs, "ratio_noise"),
        "num_calligraphers": num_callig,
        "null_id": NULL_ID,
        "t_scan": t_rows,
    }
    print()
    print("=" * 76)
    print("D6  条件因果性   ckpt=%s   cfg=%.1f   字数=%d" % (summ["ckpt"], a.cfg, len(picked)))
    print("=" * 76)
    print("%-18s %10s %10s %10s %10s %10s"
          % ("", "换书家D", "换nullD", "换噪声D", "null比", "噪声比"))
    print("%-18s %10.5f %10.5f %10.5f %10.2f %10.2f"
          % ("均值", summ["D_style"], summ["D_null"], summ["D_noise"],
             summ["ratio_null"], summ["ratio_noise"]))
    print()
    print("  判读：")
    if summ["ratio_null"] > 1.5:
        print("   ✓ 换书家 >> 换null (%.2f×) -> 模型**确实在区分具体书家**（好）"
              % summ["ratio_null"])
    elif summ["ratio_null"] > 1.05:
        print("   ◐ 换书家略大于换null (%.2f×) -> 模型只知道「有/无条件」，具体书家信息很弱"
              % summ["ratio_null"])
    else:
        print("   ✗ 换书家 ≈ 换null (%.2f×) -> 模型**没把书家当条件**，换哪个 id 都一样"
              % summ["ratio_null"])
    if summ["ratio_noise"] < 1.5:
        print("   ✗ 换书家/换噪声 = %.2f < 1.5 -> 书家效应淹没在生成随机性里"
              % summ["ratio_noise"])
    print("   注：换null 用 forward_with_cfg，cfg=%.1f 时 uncond 通路权重为 0 -> "
          "换null 的差异反映的是**条件向量的直接作用**" % a.cfg)

    if a.out:
        json.dump({"summary": summ, "pairs": res_pairs}, open(a.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
