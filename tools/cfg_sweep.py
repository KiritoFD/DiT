"""推理参数扫描（cfg × steps），用**固定 ckpt**，零训练成本，不干扰训练。

为什么单独写而不是复用现有工具：
  - `gpu_batch_eval_v2.py` 是旧 ControlNet 时代的（用 MCCDDataset、不传 g）-> 不适用
  - `auto_eval_gpu.build_model` 已过期（硬编码 learn_sigma=True，缺 glyph_inject_mode /
    norm_type / rope / glyph_vec_cond 等）-> 会在 strict=False 下**静默加载错架构**
  所以这里**照 src/train/train.py 的构造原样复制**，并用 `load_state_dict(strict=True)`
  当护栏 —— 任何字段不一致会直接抛错，而不是静默用随机权重。

用法:
  python tools/cfg_sweep.py --ckpt <path.pt> --tag v12_S2_100k \
      --cfgs 0.0,0.3,0.5,0.7,1.0,1.3,1.7,2.2 --steps 50
"""
import argparse, csv, json, os, sys, time

import torch

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

DEFAULT_SETS = "seen:assets/eval_seen_v10.csv:10,strict:assets/eval_fame3_strict_clean_v9.csv:50"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_model_from_args(a, device):
    """**严格复刻** train.py 的 DiT_2Cond_models[...] 构造（字段与默认值逐一对齐）。"""
    from src.model import DiT_2Cond_models

    latent_size = a.image_size // int(getattr(a, "vae_downscale", 4))
    _qk = bool(getattr(a, "qk_norm", True))
    _rope = bool(getattr(a, "rope", True))
    _n_aux = len([s for s in str(getattr(a, "aux_latent_shards_dirs", "") or "").split(",") if s])
    _ids_map = None

    model = DiT_2Cond_models[a.model](
        input_size=latent_size,
        num_calligraphers=a.num_calligraphers,
        num_characters=a.num_characters,
        use_checkpoint=getattr(a, "use_checkpoint", False),
        learn_sigma=bool(getattr(a, "learn_sigma", False)),
        condition_fusion=a.condition_fusion,
        callig_embed_dim=a.callig_embed_dim,
        char_embed_dim=a.char_embed_dim,
        glyph_vec_cond=getattr(a, "glyph_vec_cond", False),
        glyph_vec_dim=int(getattr(a, "glyph_vec_dim", 128)),
        glyph_vec_pool=getattr(a, "glyph_vec_pool", "mean"),
        cond_drop_all_prob=a.cond_drop_all_prob,
        cond_drop_one_prob=a.cond_drop_one_prob,
        cond_drop_which_glyph_prob=getattr(a, "cond_drop_which_glyph_prob", 0.5),
        use_glyph_cond=(getattr(a, "w_glyph_cond", 0) > 0 or getattr(a, "skel_as_glyph_cond", False)),
        use_char_cond=not getattr(a, "no_char_cond", False),
        glyph_scale_init=getattr(a, "glyph_scale_init", 0.4),
        glyph_drop_prob=getattr(a, "glyph_drop_prob", 0.0),
        glyph_inject_layers=getattr(a, "glyph_inject_layers", 0),
        glyph_inject_mode=getattr(a, "glyph_inject_mode", "adaln"),
        glyph_embedder_depth=getattr(a, "glyph_embedder_depth", 0),
        glyph_embedder_sep=getattr(a, "glyph_embedder_sep", False),
        style_token_n=getattr(a, "style_token_n", 0),
        style_role_init=getattr(a, "style_role_init", 0.02),
        glyph_in_channels=4,
        in_channels=(int(getattr(a, "latent_channels", 4)) + 4 * _n_aux),
        char_proj_mode=getattr(a, "char_proj_mode", "full"),
        callig_proj_mode=getattr(a, "callig_proj_mode", "linear"),
        callig_scale_init=float(getattr(a, "callig_scale_init", 1.0)),
        callig_style_attn=getattr(a, "callig_style_attn", False),
        callig_n_style=getattr(a, "callig_n_style", 8),
        callig_spatial=getattr(a, "callig_spatial", False),
        callig_spatial_rank=int(getattr(a, "callig_spatial_rank", 64)),
        freeze_char_table=getattr(a, "freeze_char_table", False),
        use_ids_char_embedder=getattr(a, "use_ids_char_embedder", False),
        ids_file=getattr(a, "ids_file", None),
        char_id_to_char=_ids_map,
        use_std_dino_char_embedder=getattr(a, "use_std_dino_char_embedder", False),
        std_dino_table_path=getattr(a, "std_dino_table_path", None),
        norm_type=getattr(a, "norm_type", "rms"),
        mlp_type=getattr(a, "mlp_type", "swiglu"),
        qk_norm=_qk,
        rope=_rope,
        rope_theta=getattr(a, "rope_theta", 100.0),
        attn_impl=getattr(a, "attn_impl", "sdpa"),
        image_channels=(a.image_channels
                        if getattr(a, "image_channels", None) is not None
                        else int(getattr(a, "latent_channels", 4))),
    )
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cfgs", default="0.0,0.3,0.5,0.7,1.0,1.3,1.7,2.2")
    ap.add_argument("--steps", default="50")
    ap.add_argument("--sets", default=DEFAULT_SETS,
                    help="'name:csv:n' 逗号分隔; 传 'skip' 则用 ckpt args 里的设置")
    ap.add_argument("--out-root", default="assets/results/_sweep")
    ap.add_argument("--ema", default="1", help="1=用 ema 权重, 0=用 model 权重")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from src.eval import in_mem_eval as im

    log(f"loading ckpt {args.ckpt}")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    use_ema = str(args.ema) == "1"
    sd = ck["ema"] if (use_ema and "ema" in ck) else ck["model"]
    log(f"weights from {'ema' if (use_ema and 'ema' in ck) else 'model'}")

    model = build_model_from_args(a, args.device)

    # ---- 复刻 train.py 构造后的两步后处理 (否则 state_dict 对不上) ----
    # 1) 预训练书家表: 覆盖 [0,N) 行, null 行保持随机
    _cep = getattr(a, "callig_emb_pretrained", "")
    if _cep:
        if not os.path.isabs(_cep) and not os.path.exists(_cep):
            _cep = os.path.join(ROOT, _cep)
        if os.path.exists(_cep):
            _d = torch.load(_cep, map_location="cpu", weights_only=False)
            _emb = _d["embedding"] if isinstance(_d, dict) else _d
            _w = model.y_callig_embedder.embedding_table.weight
            assert _emb.shape == (_w.shape[0] - 1, _w.shape[1]), \
                f"预训练书家表形状 {tuple(_emb.shape)} != 表 {tuple(_w.shape)} - null 行"
            with torch.no_grad():
                _w[:_emb.shape[0]].copy_(_emb.float())
            del _d
            log(f"callig 预训练表已加载: {tuple(_emb.shape)}")
    # 2) 冻结书家表: 会把 CFG null token **拆成独立 Parameter** `null_embed`
    #    —— 不做这一步, state_dict 里就少 `y_callig_embedder.null_embed`
    if getattr(a, "freeze_callig_table", False):
        assert _cep, "freeze_callig_table 需要 callig_emb_pretrained"
        model.y_callig_embedder.freeze_table()
        log("callig 表已冻结 (null_embed 已拆出)")

    # ★ 护栏: strict=True。字段漏一个/默认值错一个就会抛错, 不会静默用随机权重。
    #   实测已靠它抓出两个缺口: ① torch.compile 的 `_orig_mod.` 前缀
    #   ② freeze_table() 拆出的 null_embed。都用 strict=False 就会静默用随机权重。
    _n_stripped = 0
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
        _n_stripped = 1
        log("已剥离 torch.compile 的 `_orig_mod.` 前缀")
    model.load_state_dict(sd, strict=True)
    del ck, sd
    model.eval()
    # 双轴 CFG 开关 (与 train.py 一致, 挂模型属性)
    model.cfg_glyph_scale = (None if getattr(a, "cfg_glyph_scale", None) is None
                             else float(a.cfg_glyph_scale))
    model.cfg_w_inter = float(getattr(a, "cfg_w_inter", 0.0) or 0.0)
    if model.cfg_glyph_scale is not None:
        _gdp = float(getattr(a, "glyph_drop_prob", 0.0) or 0.0)
        log(f"双轴 CFG: cfg_glyph={model.cfg_glyph_scale}, w_inter={model.cfg_w_inter}, "
            f"glyph_drop_prob={_gdp}" + ("  ⚠ 内容轴未训练!" if _gdp <= 0 else ""))
    log(f"model loaded strict=True OK (params={sum(p.numel() for p in model.parameters()):,})")

    if args.sets != "skip":
        a.in_mem_eval_sets = args.sets
    a.in_mem_eval_save_samples = False
    a.eval_self_cond = False
    a.eval_blend_alpha = 0.0

    sets = []
    for spec in str(a.in_mem_eval_sets).split(","):
        if spec.strip():
            n, c, k = spec.strip().split(":")
            sets.append((n, c, int(k)))
    log(f"eval sets: {sets}")

    cfgs = [float(x) for x in args.cfgs.split(",") if x.strip()]
    steps_list = [int(x) for x in args.steps.split(",") if x.strip()]
    rows = []
    for st in steps_list:
        for cfg in cfgs:
            out = os.path.join(args.out_root, args.tag, f"cfg{cfg:g}_steps{st}")
            os.makedirs(out, exist_ok=True)
            a.eval_cfg = cfg
            a.eval_steps = st
            im._DIFF = None            # ★ 必须清: 模块级缓存按 steps 建, 不清会复用错的调度
            t0 = time.time()
            try:
                res = im.run_in_mem_eval(model, a, 0, args.device, results_dir=out,
                                         sets=sets, logger=lambda *x: None)
            except Exception as e:
                log(f"  cfg={cfg} steps={st}  FAILED: {type(e).__name__}: {e}")
                rows.append(dict(cfg=cfg, steps=st, failed=str(e)[:80]))
                continue
            r = dict(cfg=cfg, steps=st, dt=round(time.time() - t0, 1))
            for k, v in res.items():
                r[f"{k}_ssim"] = round(float(v), 4)
            # 取 LPIPS (in_mem_eval 已写 batch CSV)
            bp = os.path.join(out, "eval_stdskel_batch.csv")
            if os.path.exists(bp):
                for k in ("seen", "strict"):
                    vals = [float(x["lpips"]) for x in csv.DictReader(open(bp, encoding="utf-8"))
                            if x.get("set") == k and x.get("lpips")]
                    if vals:
                        r[f"{k}_lpips"] = round(sum(vals) / len(vals), 4)
            rows.append(r)
            log(f"  cfg={cfg:<4} steps={st:<4} " +
                "  ".join(f"{k}={r.get(k, '')}" for k in
                          ("seen_ssim", "strict_ssim", "strict_lpips", "dt")))

    out_csv = os.path.join(args.out_root, args.tag, "sweep.csv")
    keys = sorted({k for r in rows for k in r})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    log(f"DONE -> {out_csv}")
    for r in rows:
        log("  " + json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
