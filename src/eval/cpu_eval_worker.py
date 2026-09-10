# -*- coding: utf-8 -*-
"""cpu_eval_worker.py — 单 socket eval worker (由 cpu_eval_daemon 双开).

一个 worker 依次执行多段任务 (segments), 模型/VAE 只载一次:
    --segments "base:0:100,ctrl:0:34"     (node0 示例)
    --segments "ctrl:34:100"              (node1 示例)

每段: 采样 (heun_sample_cpu) → VAE decode → PNG 落盘 (全局下标) →
      分段指标 (with_lists) → 写 {ckpt_dir}/.part_{k}.json
daemon 汇总各 part (逐样本列表精确合并) → eval_auto_ctrl_{step}.json
"""
import os, sys, json, time, argparse

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _root)
os.chdir(_root)
sys.stdout.reconfigure(encoding="utf-8")


def _strip(sd):
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--segments", required=True, help="逗号分隔 arm:start:end")
    ap.add_argument("--eval-sets", default="",
                    help="多 eval 集 'name=csv[,name=csv...]'; 空 = 旧单 csv (arm 'g')")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--dit-batch", type=int, default=16)
    ap.add_argument("--vae-batch", type=int, default=24)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--part-tag", required=True, help="part 标识 (p0/p1)")
    ap.add_argument("--mode", choices=["ctrl_pair", "pretrain_g"], default="ctrl_pair",
                    help="ctrl_pair: ControlNetDiT 双臂; pretrain_g: 裸主模型 g 条件单臂"
                         " (train.py ckpt)")
    ap.add_argument("--g-source", choices=["gt_skel", "std_glyph"], default="gt_skel",
                    help="g 条件来源: gt_skel=该样本 GT 实例骨架 (默认, 与训练/历史"
                         " ctrl 臂协议一致); std_glyph=标准字形库 (部署态零样本)")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    dev = torch.device("cpu")
    t0 = time.time()

    from src.model.controlnet import load_main_model, ControlNetDiT
    from src.model import DiT_2Cond_models
    from src.eval.inference import (make_eval_cache, load_eval_vae, decode_and_save,
                                    compute_metrics)
    from src.eval.cpu_sampler import heun_sample_cpu

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck.get("args", {}) or {}
    if isinstance(a, argparse.Namespace):   # train.py ckpt 存的是 Namespace
        a = vars(a)
    ctrl_sd = _strip(ck.get("ema") or ck.get("ctrl") or {})
    main_sd = _strip(ck.get("model") or ck.get("ema_model") or {})

    arch = dict(norm_type=a.get("norm_type", "rms"), mlp_type=a.get("mlp_type", "swiglu"),
                qk_norm=bool(a.get("qk_norm", 1)), rope=bool(a.get("rope", 1)),
                rope_theta=float(a.get("rope_theta", 100.0)),
                attn_impl=a.get("attn_impl", "sdpa"))
    common = dict(model_name=a.get("model", "DiT-2Cond-S/2"), device=dev,
                  num_calligraphers=int(a.get("num_calligraphers", 1013)),
                  num_characters=int(a.get("num_characters") or 35130),
                  condition_fusion=a.get("condition_fusion", "factorized_add"),
                  callig_embed_dim=int(a.get("callig_embed_dim", 128)),
                  char_embed_dim=int(a.get("char_embed_dim") or 384),
                  char_proj_mode=(a.get("char_proj_mode") or "mlp"),
                  freeze_char_table=bool(a.get("freeze_char_table", True)),
                  learn_sigma=False, **arch)

    if args.mode == "pretrain_g":
        from src.utils import get_glyph_lookup_v2
        return _run_pretrain_g(args, ck, a, arch, common, t0)
    main_ckpt = a.get("main_ckpt", "")
    if main_sd:
        # 联训 ckpt 自含 main.*: 先建目标架构 (有 base 路径则加载后再灌, 双保险)
        if main_ckpt and os.path.isfile(main_ckpt):
            main_model = load_main_model(ckpt_path=main_ckpt, **common)
        else:
            main_model = DiT_2Cond_models[common["model_name"]](
                num_calligraphers=common["num_calligraphers"],
                num_characters=common["num_characters"],
                condition_fusion=common["condition_fusion"],
                callig_embed_dim=common["callig_embed_dim"],
                char_embed_dim=common["char_embed_dim"],
                char_proj_mode=common["char_proj_mode"],
                cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
                cond_drop_which_glyph_prob=0.5, use_checkpoint=False,
                learn_sigma=False, **arch)
        miss, unexp = main_model.load_state_dict(main_sd, strict=False)
        assert len(unexp) == 0, f"main weights unexpected={len(unexp)}"
        if main_ckpt and os.path.isfile(main_ckpt) and len(miss) > 0:
            raise RuntimeError(f"main weights missing={len(miss)} (arch mismatch?)")
    else:
        if not main_ckpt or not os.path.isfile(main_ckpt):
            raise RuntimeError(f"ctrl-only ckpt 但 main_ckpt 无效: {main_ckpt!r}")
        main_model = load_main_model(ckpt_path=main_ckpt, **common)
    main_model.eval()

    c = ControlNetDiT(main_model, cond_in_channels=int(a.get("skel_cond_channels", 4)),
                      train_ctrl_only=True, injection=a.get("injection", "modulate"),
                      null_cond=a.get("null_cond", "gaussian"), **arch)
    miss, unexp = c.load_state_dict(ctrl_sd, strict=False)
    assert len(unexp) == 0, f"ctrl weights unexpected={len(unexp)}"
    c.eval()
    t_load = time.time() - t0

    csv = a.get("gpu_eval_csv") or a.get("eval_csv")
    img_root = a.get("gpu_eval_img_root") or a.get("img_root") or None
    n = args.n or int(a.get("gpu_eval_n", 100))
    cache = make_eval_cache(csv, img_root, None, 256, n, 8, 4, 0.18215,
                            skel_latent_shards_dir=a.get("gpu_eval_skel_latent_shards_dir")
                            or a.get("skel_latent_shards_dir"))
    cfg = float(a.get("gpu_eval_cfg", 0.7))
    steps = int(a.get("gpu_eval_steps", 50))
    shift = float(a.get("shift", 1.0))
    noise, conds = cache["noise"], cache["conds"]
    skels_lat = cache["skels_latent"].float()
    vae = load_eval_vae(dev, "pretrained_models/sd-vae-ft-ema")
    step = int(a.get("_eval_step", os.path.basename(args.ckpt).split(".")[0]))
    step_tag = f"step{step:07d}"

    parts = []
    for seg in args.segments.split(","):
        arm, s0, s1 = seg.split(":")
        s0, s1 = int(s0), int(s1)
        arm_dir = os.path.join(args.out_dir, "eval_samples_ctrl", step_tag, arm)
        os.makedirs(arm_dir, exist_ok=True)
        skel = skels_lat[s0:s1] if arm == "ctrl" else None
        t1 = time.time()
        lat = heun_sample_cpu(c, noise[s0:s1], conds[s0:s1], cfg, args.dit_batch,
                              skel=skel, seed=0, steps=steps, shift=shift)
        t_s = time.time() - t1
        t2 = time.time()
        decode_and_save(vae, lat, 0.18215, arm_dir, arm,
                        gts=cache["gts"][s0:s1], vae_batch=args.vae_batch,
                        idx_offset=s0)
        t_d = time.time() - t2
        m, lists = compute_metrics(arm_dir, arm_dir, arm, n,
                                   use_lpips=True, idx_range=(s0, s1), with_lists=True)
        m["t_sample"], m["t_decode"] = round(t_s, 1), round(t_d, 1)
        parts.append({"arm": arm, "range": [s0, s1], "metrics": m, "lists": lists})
        print(f"[worker:{args.part_tag}] {arm}[{s0}:{s1}] sample={t_s:.0f}s "
              f"dec={t_d:.0f}s ssim={m.get('ssim_mean')}", flush=True)

    out_json = os.path.join(os.path.dirname(args.ckpt), f".part_{args.part_tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"step": step, "part": args.part_tag, "t_load": round(t_load, 1),
                   "parts": parts}, f, ensure_ascii=False)
    print(f"[worker:{args.part_tag}] DONE -> {out_json}", flush=True)


def _run_pretrain_g(args, ck, a, arch, common, t0):
    """v10a 预训练 ckpt 的 g 单臂评测: 裸主模型 + 标准字形库 g (部署态)."""
    import torch
    from src.model import DiT_2Cond_models
    from src.eval.inference import make_eval_cache, load_eval_vae, decode_and_save, compute_metrics
    from src.eval.cpu_sampler import heun_sample_cpu
    dev = torch.device("cpu")
    use_g = bool(a.get("w_glyph_cond", False) or a.get("skel_as_glyph_cond", False))
    model = DiT_2Cond_models[common["model_name"]](
        num_calligraphers=common["num_calligraphers"],
        num_characters=common["num_characters"],
        condition_fusion=common["condition_fusion"],
        callig_embed_dim=common["callig_embed_dim"],
        char_embed_dim=common["char_embed_dim"],
        char_proj_mode=common["char_proj_mode"],
        freeze_char_table=common["freeze_char_table"],
        cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
        cond_drop_which_glyph_prob=0.5, use_checkpoint=False, learn_sigma=False,
        use_glyph_cond=use_g,
        use_char_cond=not bool(a.get("no_char_cond", False)),
        use_std_dino_char_embedder=bool(a.get("use_std_dino_char_embedder", False)),
        std_dino_table_path=a.get("std_dino_table_path"),
        glyph_scale_init=float(a.get("glyph_scale_init", 0.4)),
        glyph_drop_prob=float(a.get("glyph_drop_prob", 0.0)),
        glyph_embedder_depth=int(a.get("glyph_embedder_depth", 0)),
        glyph_inject_layers=int(a.get("glyph_inject_layers", 0)),
        callig_style_attn=bool(a.get("callig_style_attn", False)),
        callig_n_style=int(a.get("callig_n_style", 8)),
        # 风格 token 每层注入 (GlyphStyleCrossAttn): 漏传会使 ckpt 的
        # style_proj.*/style_role 变成 unexpected -> assert 崩 (2026-09-10 修)
        style_token_n=int(a.get("style_token_n", 0)),
        style_role_init=float(a.get("style_role_init", 0.02)),
        glyph_inject_mode=a.get("glyph_inject_mode", "adaln"), **arch)
    # 冻结书家表会把 null token 拆成独立 Parameter (y_callig_embedder.null_embed),
    # ckpt 里带着这个键 —— eval 构建必须复现冻结结构, 否则 unexp=1 assert 崩
    if a.get("freeze_callig_table"):
        m_ye = model.y_callig_embedder
        m_ye.freeze_table()
    sd = _strip(ck.get("ema") or ck.get("model") or ck)
    miss, unexp = model.load_state_dict(sd, strict=False)
    if unexp:
        # F5 修复: 崩溃前先打印明细 (缺哪个模块一眼可见, 不必复现调试)
        _msg = (f"main weights unexpected={len(unexp)}: {sorted(unexp)[:10]} | "
                f"missing={len(miss)}: {sorted(miss)[:5]}")
        log(_msg)
        raise RuntimeError(_msg)
    model.eval()

    csv = a.get("gpu_eval_csv") or a.get("eval_csv") or a.get("data_csv")
    img_root = a.get("gpu_eval_img_root") or a.get("img_root") or None
    n = args.n or int(a.get("gpu_eval_n", a.get("eval_n", 100)))
    shards = (a.get("gpu_eval_skel_latent_shards_dir")
              or a.get("skel_latent_shards_dir") or None)
    # 书家词表收紧: y_callig 必须与训练数据层同一张映射表 (raw -> 0..N-1)
    cmap = None
    if a.get("callig_id_map"):
        from src.utils.callig_map import load_callig_id_map
        _cm_path = a["callig_id_map"]
        if not os.path.isabs(_cm_path) and not os.path.exists(_cm_path):
            _cm_path = os.path.join("/root/Workspace/xy/DiT", _cm_path)
        cmap, _ = load_callig_id_map(_cm_path)
    cfg = float(a.get("eval_cfg", a.get("gpu_eval_cfg", 0.7)))
    steps = int(a.get("eval_steps", a.get("gpu_eval_steps", 50)))
    shift = float(a.get("shift", 1.0))
    vae = load_eval_vae(dev, "pretrained_models/sd-vae-ft-ema")

    # ── eval 集: --eval-sets "name=csv[,name=csv...]" 多集; 空 = 旧单 csv (arm "g")
    eval_sets = []
    if getattr(args, "eval_sets", ""):
        for _spec in args.eval_sets.split(","):
            _nm, _p = _spec.split("=", 1)
            eval_sets.append((_nm, _p))
    else:
        eval_sets = [("g", csv)]
    # segments: {arm: [(s0,s1), ...]} —— daemon 按集切半分给两个 worker
    seg_map = {}
    for seg in args.segments.split(","):
        arm, s0, s1 = seg.split(":")
        seg_map.setdefault(arm, []).append((int(s0), int(s1)))
    step = int(a.get("_eval_step", os.path.basename(args.ckpt).split(".")[0]))
    step_tag = f"step{step:07d}"
    parts = []

    for name, csvp in eval_sets:
        _segs = seg_map.get(name) or seg_map.get("g") or []
        if not _segs:
            continue
        rows_n = max(sum(1 for _ in open(csvp, encoding="utf-8")) - 1, 2)
        cache = make_eval_cache(csvp, img_root, None, 256, rows_n, 8, 4, 0.18215,
                                skel_latent_shards_dir=shards, callig_id_map=cmap)
        if args.g_source == "gt_skel":
            g_all = cache["skels_latent"].float()
            # 用 g_all 实际行数 (cache 按 csv 行数截断)
            n_eff = g_all.shape[0]
            hit = int((g_all.view(n_eff, -1).sum(1) != 0).sum())
            print(f"[worker:{args.part_tag}:{name}] 骨架覆盖 {hit}/{n_eff}", flush=True)
        else:
            from src.utils import get_glyph_lookup_v2
            lk = get_glyph_lookup_v2()
            import csv as _csv
            rows = list(_csv.DictReader(open(csvp, encoding="utf-8")))[:rows_n]
            g_all = torch.zeros(rows_n, 4, 32, 32)
            hit = 0
            for i, r in enumerate(rows):
                gv = lk.get(int(r["script_id"]), r.get("character", ""), random=False)
                if gv is not None:
                    g_all[i] = gv.float()
                    hit += 1
            print(f"[worker:{args.part_tag}:{name}] glyph 库命中 {hit}/{rows_n}", flush=True)
        # seen 集落盘保持旧路径 .../g/ (poster 兼容); 其他集独立子目录
        sub = "g" if name in ("g", "seen") else name
        arm_dir = os.path.join(args.out_dir, "eval_samples_ctrl", step_tag, sub)
        os.makedirs(arm_dir, exist_ok=True)
        noise, conds = cache["noise"], cache["conds"]
        for (s0, s1) in _segs:
            t1 = time.time()
            print(f"[worker:{args.part_tag}:{name}] sampling [{s0}:{s1}] "
                  f"({s1 - s0} samples) ...", flush=True)

            def _prog(msg, _t=t1, _n=name):
                print(f"[worker:{args.part_tag}:{_n}] {msg} "
                      f"(elapsed {time.time() - _t:.0f}s)", flush=True)

            lat = heun_sample_cpu(model, noise[s0:s1], conds[s0:s1], cfg, args.dit_batch,
                                  skel=g_all[s0:s1], seed=0, steps=steps, shift=shift,
                                  cond_key="g", log_fn=_prog)
            t_s = time.time() - t1
            t2 = time.time()
            decode_and_save(vae, lat, 0.18215, arm_dir, "g",
                            gts=cache["gts"][s0:s1], vae_batch=args.vae_batch,
                            idx_offset=s0)
            t_d = time.time() - t2
            m, lists = compute_metrics(arm_dir, arm_dir, "g", s1,
                                       use_lpips=True, idx_range=(s0, s1), with_lists=True)
            m["t_sample"], m["t_decode"] = round(t_s, 1), round(t_d, 1)
            parts.append({"arm": name, "range": [s0, s1], "metrics": m, "lists": lists})
            print(f"[worker:{args.part_tag}:{name}] [{s0}:{s1}] sample={t_s:.0f}s "
                  f"dec={t_d:.0f}s ssim={m.get('ssim_mean')}", flush=True)
    out_json = os.path.join(os.path.dirname(args.ckpt), f".part_{args.part_tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"step": step, "part": args.part_tag, "t_load": round(time.time()-t0, 1),
                   "parts": parts}, f, ensure_ascii=False)
    print(f"[worker:{args.part_tag}] DONE -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
