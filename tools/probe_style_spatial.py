# -*- coding: utf-8 -*-
"""D3 — 风格差异的**空间分布**。

问题：固定噪声 + 固定字（同 g）**只换书家**，两张生成图的差异是
    * 均匀铺满全图     → 风格是**全局属性** → adaLN 语义正确，不需要局部通路
    * 集中在起笔/收笔/转折/波磔 → 风格有**空间结构** → 需要局部通路（S2 的 B2/B3）

做法（对齐 `docs/922/60_diagnostics.md` §3）：
    1. 从 eval rows 里找同一个字有多个书家的行
    2. **固定噪声**（所有对照都用同一份 noise）+ **固定 g**（同一字同一骨架）
    3. 只改 y_callig -> 采样多张 -> decode
    4. 差异图 D = |imgA - imgB|（逐像素 L1）
    5. 用 GT 墨迹 mask 切分：
          比值 r = mean(D | 笔画内) / mean(D | 背景)
          r ≈ 1  -> 差异均匀（风格全局）
          r >> 1 -> 差异集中在笔画（风格有空间结构）
    6. **噪声地板**：同一书家、换噪声采两次 —— 这是"与风格无关的采样抖动"。
       只有超过地板的差异才算风格信号。

⚠ 白底占比 ~90%，全图均值会被背景稀释（与墨迹域指标的教训同源），
   所以必须**分区域算**，必须报 `style/noise` 的比值。

用法:
    python tools/probe_style_spatial.py \
        --ckpt assets/results/v13_base_50k/<ts>/checkpoints/0155000.pt \
        --eval-csv assets/eval_v13_strict.csv \
        --img-root data/imgs/final_imgs_fame_v8 \
        --n-chars 12 --n-calligs 4 --cfg 1.0 \
        --out assets/d3_v13_base.json
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(m):
    print(m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval-csv", default="assets/eval_v13_strict.csv")
    ap.add_argument("--img-root", default="")
    ap.add_argument("--callig-id-map", default="")
    ap.add_argument("--callig-script-map", default="")
    ap.add_argument("--skel-shards", default="")
    ap.add_argument("--n-chars", type=int, default=12)
    ap.add_argument("--n-calligs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--n-ring", type=int, default=4,
                    help="按到骨架的距离分环数（看差异聚集在中心还是边缘）")
    ap.add_argument("--n-floor-rep", dest="n_floor_rep", type=int, default=4,
                    help="噪声地板重复次数（地板是随机量，1 次估计方差过大；至少 4）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--script-mode", default="any", choices=["any", "cross", "same"],
                    help="any=同字不同书家(跨书体也算); cross=必须跨书体; same=必须同书体(纯书家对照)")
    ap.add_argument("--require-skel", dest="require_skel", action="store_true", default=True,
                    help="选字时排除 skel latent 缺失的行（默认开，避免撞 make_eval_cache 的 98%% 闸）")
    ap.add_argument("--no-require-skel", dest="require_skel", action="store_false")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    a.image_size, a.vae_downscale, a.latent_channels = 256, 8, 4

    import numpy as np
    import torch as th
    from PIL import Image
    from scipy import ndimage

    dev = th.device(a.device if th.cuda.is_available() else "cpu")

    # ---- 0) ckpt 里存了完整 args。★ 绝对不要手搓构造参数字典：
    #   train.py 的字段有几十个（glyph_vec_proj / no_char_cond / ...），
    #   漏一个就 Missing key 崩，且**形状不匹配时会静默保持随机初始化**。
    #   必须复用 model_io.build_model_from_args（与 train.py 逐字段对齐）。
    ck = th.load(a.ckpt, map_location="cpu", weights_only=False)
    ckargs = ck.get("args") or {}
    if not isinstance(ckargs, dict):
        ckargs = vars(ckargs) if hasattr(ckargs, "__dict__") else {}

    def _strip(sd_):
        """去掉 module. / _orig_mod. 前缀（两者可能叠加）。"""
        out = {}
        for k, v in sd_.items():
            kk = k
            for pfx in ("module.", "_orig_mod.", "model."):
                while kk.startswith(pfx):
                    kk = kk[len(pfx):]
            out[kk] = v
        return out

    sd = _strip(ck.get("ema") or ck.get("model") or ck)

    from src.eval.model_io import build_model_from_args
    import argparse as _ap
    ns = _ap.Namespace(**{k: v for k, v in ckargs.items()})
    model = build_model_from_args(ns, dev)

    # 预训练表（与 train.py 一致：表要先填再 load_state_dict）
    _cep = ckargs.get("callig_emb_pretrained") or ""
    if _cep:
        if not os.path.isabs(_cep) and not os.path.exists(_cep):
            _cep = os.path.join(ROOT, _cep)
        if os.path.exists(_cep):
            _d = th.load(_cep, map_location="cpu", weights_only=False)
            _emb = _d["embedding"] if isinstance(_d, dict) else _d
            _shp = tuple(_emb.shape)
            with th.no_grad():
                model.y_callig_embedder.embedding_table.weight[:_emb.shape[0]].copy_(
                    _emb.float())
            log("  callig 预训练表已加载 %s" % (_shp,))
            del _d, _emb
    if ckargs.get("freeze_callig_table"):
        model.y_callig_embedder.freeze_table()

    # S2 的 zero-init 模块：ckpt 里若有它们的权重，load 会覆盖，无需特殊处理
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        log("[D3] ✗ strict 加载失败 —— ckpt 与 model_io 构造不一致:\n%s" % str(e)[:800])
        raise
    model.eval()
    log("[D3] 模型加载 OK (strict=True)")

    def gv(k, d=None):
        v = ckargs.get(k)
        return d if v is None else v


    from src.eval.inference import make_eval_cache, load_eval_vae

    # ---- id 映射
    cmap = None
    p = gv("callig_id_map", "")
    if p:
        cands = [p, os.path.join(ROOT, p), os.path.join(ROOT, "assets", os.path.basename(p)),
                 os.path.join("assets", os.path.basename(p))]
        for c in cands:
            if c and os.path.exists(c):
                from src.utils.callig_map import load_callig_id_map
                cmap, _ = load_callig_id_map(c)
                log("  callig_id_map: %s" % c)
                break
    csm = None
    p2 = gv("callig_script_map", "")
    if p2:
        if not os.path.isabs(p2) and not os.path.exists(p2):
            p2 = os.path.join(ROOT, p2)
        if os.path.exists(p2):
            from src.utils.callig_script_map import load_callig_script_map
            csm = load_callig_script_map(p2)
            log("  callig_script_map: %s (%d pairs)" % (p2, csm["num_pairs"]))

    shards = a.skel_shards or gv("skel_latent_shards_dir", "")
    if shards and not os.path.isdir(shards):
        alt = os.path.join("data/skel", os.path.basename(shards))
        if os.path.isdir(alt):
            shards = alt
    img_root = a.img_root or None

    # ── skel 覆盖预检 ────────────────────────────────────────────────────────
    #   make_eval_cache 对覆盖率 <98% 会直接 raise（设计如此：g=ZERO 是静默失效）。
    #   与其撞错，不如**先算出哪些行有骨架**，只从有骨架的行里选字。
    skel_ids = None
    if a.require_skel and shards and os.path.isdir(shards):
        import glob as _glob
        skel_ids = set()
        for _f in _glob.glob(os.path.join(shards, "shard_*.npz")):
            try:
                _z = np.load(_f, allow_pickle=True)
                skel_ids.update([str(x) for x in _z["img_ids"]])
            except Exception:
                pass
        log("  [skel] %s -> %d 个 id" % (shards, len(skel_ids)))

    rows_all = list(csv.DictReader(open(a.eval_csv, encoding="utf-8")))

    #   id 字段探测：strict 集用 glyph_id（≠ old_50k_id！），训练 csv 可能用别的
    id_field = None
    if skel_ids is not None:
        for _cand in ("glyph_id", "old_50k_id", "character_id"):
            if _cand not in rows_all[0]:
                continue
            _cov = sum(1 for r in rows_all if str(r.get(_cand, "")) in skel_ids)
            log("  [skel] 字段 %-14s 覆盖 %d/%d (%.1f%%)"
                % (_cand, _cov, len(rows_all), 100.0 * _cov / max(len(rows_all), 1)))
            if _cov > 0.9 * len(rows_all):
                id_field = _cand
                break
        if id_field is None:
            log("  [skel] ⚠ 没有字段覆盖率 >90%% —— 关闭 skel 过滤（会撞 make_eval_cache 的 98%% 闸）")
            skel_ids = None
        else:
            log("  [skel] 采用字段: %s" % id_field)

    # ---- 1) 挑字：同字 >= n_calligs 个不同书家
    #   ⚠ 评测集若按定义是"未见三元组"（如 strict），每个字只有 1 位书家，
    #     这里必然选不出 —— 改用该 run 的**训练 csv**（同字多书家很常见）。
    from collections import defaultdict
    by_char = defaultdict(list)
    for i, r in enumerate(rows_all):
        by_char[r["character"]].append(i)

    def _pick(min_callig):
        """选字：同字 >= min_callig 个**不同书家**。

        script-mode:
          any   同字不同书家即可（书体可同可不同）
          cross 强制样本书体不同（测「书家+书体」联合差异）
          same  强制样本书体相同（**纯书家对照**，把书体混杂消掉）
        """
        out = []
        stats = {"same_callig": 0, "script_excluded": 0, "skel_miss": 0}
        # 样本多的字优先（更可能有同书体的多书家组合）
        for ch, idxs in sorted(by_char.items(), key=lambda kv: -len(kv[1])):
            cids, sel = set(), []
            for i in idxs:
                # skel 覆盖过滤：该行没有骨架 -> 选它会让 make_eval_cache raise
                if skel_ids is not None and str(rows_all[i].get(id_field, "")) not in skel_ids:
                    stats["skel_miss"] += 1
                    continue
                cid = rows_all[i]["calligrapher_id"]
                if cid in cids:
                    stats["same_callig"] += 1
                    continue
                cids.add(cid)
                sel.append(i)
            if len(sel) < min_callig:
                continue
            if a.script_mode != "any":
                sids = [str(rows_all[i].get("script_id", "")) for i in sel]
                ok = None
                for k in range(1, len(sel)):
                    cross = (sids[0] != sids[k])
                    want_cross = (a.script_mode == "cross")
                    if cross != want_cross:
                        continue
                    ok = [sel[0], sel[k]]
                    for j in range(1, len(sel)):
                        if len(ok) >= min_callig:
                            break
                        if j == k:
                            continue
                        if (sids[j] != sids[0]) == want_cross:
                            ok.append(sel[j])
                    break
                if not ok or len(ok) < min_callig:
                    stats["script_excluded"] += 1
                    continue
                sel = ok
            out.append((ch, sel[:min_callig]))
            if len(out) >= a.n_chars:
                break
        log("  [pick] 跳过重复书家 %d 行; skel 缺失 %d 行; 因书体约束排除 %d 字"
            % (stats["same_callig"], stats["skel_miss"], stats["script_excluded"]))
        return out

    picked = _pick(a.n_calligs)
    if not picked and a.n_calligs > 2:
        log("⚠ %s 里没有字满足 >= %d 书家（strict 集按定义每字只 1 书家）"
            % (os.path.basename(a.eval_csv), a.n_calligs))
        log("   → 回退到 %d 书家" % (a.n_calligs - 1))
        while not picked and a.n_calligs > 2:
            a.n_calligs -= 1
            picked = _pick(a.n_calligs)
    if not picked:
        log("✗ %s 里连 2 个书家都没有 —— 换用训练 csv（--eval-csv assets/train_*.csv）"
            % os.path.basename(a.eval_csv))
        sys.exit(1)
    log("[D3] 选中 %d 个字，每个 %d 位书家" % (len(picked), a.n_calligs))
    log("     " + ", ".join(ch for ch, _ in picked))

    # ---- 2) cache：只装选中的行（拼一个临时 csv）
    tmp = os.path.join("/tmp", "_d3_rows.csv")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_all[0].keys()))
        w.writeheader()
        for _, sel in picked:
            for i in sel:
                w.writerow(rows_all[i])
    cache = make_eval_cache(tmp, img_root, None, a.image_size, len(picked) * a.n_calligs,
                            a.vae_downscale, a.latent_channels, float(gv("vae_scaling_factor", 0.18215)),
                            skel_latent_shards_dir=(shards or None),
                            callig_id_map=cmap, callig_script_map=csm)
    hier = cache.get("hier_conds")
    log("[D3] cache: conds=%d gts=%s skels=%s"
        % (len(cache["conds"]), tuple(cache["gts"].shape),
           None if cache.get("skels_latent") is None else tuple(cache["skels_latent"].shape)))

    vae = load_eval_vae("cpu", "data/pretrained/sd-vae-ft-ema")
    sf = float(gv("vae_scaling_factor", 0.18215))
    shift = float(gv("shift", 1.0) or 1.0)

    def sample(noise_lat, cond_pair, g_lat, hier_idx=None):
        """heun 采样 + decode -> (H,W) 灰度 [0,1]。

        ★ 2026-09-22 修 OOM：整个采样必须包在 no_grad 里。
          原实现只在 VAE decode 那一步包了 no_grad，而 2*steps 次
          `forward_with_cfg` 全在 grad 模式下跑 —— 每一步都往计算图里挂节点，
          显存随步数**线性增长**，50 步（100 次前向）直接顶到 24G 触发 OOM。
          冒烟用 20 步（40 次前向）能过，正是因为这个原因，不是别的问题。
        """
        n = noise_lat.shape[0]
        mk = dict(y_callig=th.tensor([cond_pair[0]] * n, device=dev, dtype=th.long),
                  y_char=th.tensor([cond_pair[1]] * n, device=dev, dtype=th.long))
        if g_lat is not None and getattr(model, "use_glyph_cond", False):
            mk["g"] = g_lat.to(dev).float()
        if hier_idx is not None and hasattr(model, "style_hier") \
                and getattr(model, "style_hier", None) is not None:
            mk["y_callig_raw"] = th.tensor([hier_idx[0]] * n, device=dev, dtype=th.long)
            mk["y_pair"] = th.tensor([hier_idx[1]] * n, device=dev, dtype=th.long)
            mk["y_script"] = th.tensor([hier_idx[2]] * n, device=dev, dtype=th.long)
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
        img = ((img.clamp(-1, 1) + 1) / 2)[0].float().mean(axis=0)
        return img.numpy()

    # ---- 3) 逐个字采样
    pairs, floors = [], []
    base = 0
    for ci, (ch, sel) in enumerate(picked):
        idx0 = base
        base += len(sel)
        noise = cache["noise"][idx0:idx0 + 1].clone()      # ★ 固定噪声
        g0 = cache["skels_latent"][idx0:idx0 + 1] \
            if cache.get("skels_latent") is not None else None   # ★ 固定骨架
        gt = ((cache["gts"][idx0].mean(axis=0).numpy() + 1) / 2)
        ink = gt < 0.5
        if ink.sum() < 100:
            log("  [skip] %s: GT 墨迹太少" % ch)
            continue

        imgs, ids = [], []
        for j in range(len(sel)):
            row = idx0 + j
            cp = cache["conds"][row]
            hi = None
            if hier:
                hi = (hier[0][row], hier[1][row], hier[2][row])
            im = sample(noise, cp, g0, hi)
            imgs.append(im)
            ids.append(int(rows_all[sel[j]]["calligrapher_id"]))
        # 噪声地板：同书家同骨架，换噪声。
        # ★ 2026-09-22 修：原来只采 1 次 → 地板是**单样本估计**，方差极大
        #   （实测同一 ckpt 两次跑出 0.208 vs 0.337，比值随之在 1.59/0.98 间跳）。
        #   改成采 n_floor_rep 次、各自与 imgs[0] 比、再平均，并报标准差。
        _cp0 = cache["conds"][idx0]
        _hi0 = (hier[0][idx0], hier[1][idx0], hier[2][idx0]) if hier else None
        Dn_list = []
        for _r in range(max(1, a.n_floor_rep)):
            im_n = sample(th.randn_like(noise), _cp0, g0, _hi0)
            Dn_list.append(np.abs(imgs[0] - im_n))
        Dn = np.mean(Dn_list, axis=0)
        Dn_std = float(np.mean([((d - Dn) ** 2).mean() ** 0.5 for d in Dn_list])) \
            if len(Dn_list) > 1 else 0.0

        for x in range(len(imgs)):
            for y in range(x + 1, len(imgs)):
                D = np.abs(imgs[x] - imgs[y])
                pairs.append({"char": ch, "A": ids[x], "B": ids[y],
                              "D_in_ink": float(D[ink].mean()),
                              "D_in_bg": float(D[~ink].mean()),
                              "D_global": float(D.mean()),
                              "ratio": float(D[ink].mean() / max(D[~ink].mean(), 1e-9))})
        floors.append({"char": ch,
                       "n_rep": len(Dn_list),
                       "D_in_ink": float(Dn[ink].mean()),
                       "D_in_bg": float(Dn[~ink].mean()),
                       "D_global": float(Dn.mean()),
                       "D_global_std": Dn_std,
                       "ratio": float(Dn[ink].mean() / max(Dn[~ink].mean(), 1e-9))})
        log("  [%2d/%d] %s  换书家 ratio=%.3f  换噪声 ratio=%.3f (±%.4f)"
            % (ci + 1, len(picked), ch,
               np.mean([p["ratio"] for p in pairs[-len(imgs) * (len(imgs) - 1) // 2:]]),
               floors[-1]["ratio"], Dn_std))

    def m(rs, k):
        v = [r[k] for r in rs]
        return float(sum(v) / len(v)) if v else float("nan")

    summ = {"ckpt": os.path.basename(a.ckpt), "cfg": a.cfg,
            "n_pairs": len(pairs), "n_floor": len(floors),
            "style_D_in_ink": m(pairs, "D_in_ink"),
            "style_D_in_bg": m(pairs, "D_in_bg"),
            "style_D_global": m(pairs, "D_global"),
            "style_ratio": m(pairs, "ratio"),
            "noise_D_in_ink": m(floors, "D_in_ink"),
            "noise_D_in_bg": m(floors, "D_in_bg"),
            "noise_D_global": m(floors, "D_global"),
            "noise_ratio": m(floors, "ratio")}
    summ["style_over_noise_ink"] = summ["style_D_in_ink"] / max(summ["noise_D_in_ink"], 1e-9)
    summ["style_over_noise_global"] = summ["style_D_global"] / max(summ["noise_D_global"], 1e-9)
    summ["noise_D_global_std"] = m(floors, "D_global_std")
    summ["n_floor_rep"] = a.n_floor_rep
    # 严格判据：扣除一个噪声标准差后的下界仍然 > 1 才算风格超噪声
    _lo = summ["style_D_global"] - summ["noise_D_global_std"]
    summ["style_over_noise_global_lb"] = _lo / max(summ["noise_D_global"], 1e-9)

    print()
    print("=" * 74)
    print("D3   ckpt=%s   cfg=%.1f   对数=%d   地板=%d(每字 %d 次)"
          % (summ["ckpt"], a.cfg, summ["n_pairs"], summ["n_floor"], a.n_floor_rep))
    print("=" * 74)
    print("%-20s %12s %12s %10s %12s" % ("", "笔画内D", "背景D", "内/外", "全图D"))
    print("%-20s %12.5f %12.5f %10.3f %12.5f"
          % ("换书家(风格)", summ["style_D_in_ink"], summ["style_D_in_bg"],
             summ["style_ratio"], summ["style_D_global"]))
    print("%-20s %12.5f %12.5f %10.3f %12.5f"
          % ("换噪声(地板)", summ["noise_D_in_ink"], summ["noise_D_in_bg"],
             summ["noise_ratio"], summ["noise_D_global"]))
    print()
    print("  ★ 风格/噪声  笔画内 = %.3f   全图 = %.3f"
          % (summ["style_over_noise_ink"], summ["style_over_noise_global"]))
    print("  ★ 扣除噪声 1σ 后的下界(全图) = %.3f   (地板 σ=%.5f)"
          % (summ["style_over_noise_global_lb"], summ["noise_D_global_std"]))
    print()
    print("  判读：")
    if summ["style_ratio"] < 1.5:
        print("   内/外 = %.2f < 1.5 -> 差异**均匀铺满**" % summ["style_ratio"])
    else:
        print("   内/外 = %.2f >= 1.5 -> 差异集中在笔画内" % summ["style_ratio"])
    # ★ 关键：地板本身也有 内/外 比。如果地板的 内/外 同样 >=1.5，
    #   说明"集中笔画内"是**骨架约束的必然结果**，不是风格的证据。
    if summ["noise_ratio"] >= 1.5:
        print("   ⚠ 但**噪声地板的内/外 = %.2f 同样 >=1.5** —— 说明「差异集中在笔画内」"
              % summ["noise_ratio"])
        print("      是骨架把任何扰动都限制在墨迹内的必然结果，**不能当作风格有空间结构的证据**")
    if summ["style_over_noise_global_lb"] < 1.0:
        print("   ✗ 扣除噪声 σ 后下界 = %.2f < 1.0 -> 换书家的差异**落在噪声波动范围内**，"
              % summ["style_over_noise_global_lb"])
        print("      即「换书家」与「换随机噪声」在像素上不可区分 -> 风格基本没进像素")
    elif summ["style_over_noise_global_lb"] < 1.5:
        print("   ◐ 下界 = %.2f 略超噪声，但未达 1.5 -> 风格信号很弱"
              % summ["style_over_noise_global_lb"])
    else:
        print("   ✓ 下界 = %.2f >= 1.5 -> 风格信号显著超噪声地板"
              % summ["style_over_noise_global_lb"])

    if a.out:
        json.dump({"summary": summ, "pairs": pairs, "noise_floor": floors},
                  open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
