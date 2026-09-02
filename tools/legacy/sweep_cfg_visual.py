# -*- coding: utf-8 -*-
"""
sweep_cfg_visual.py — 扫描 cfg 并生成**同一批字 × 多个 cfg** 的对比网格图，供目检。

为什么必须目检而不能只看 SSIM
------------------------------
我们的关键观察是「cfg<1 显著更好」。但从 CFG 语义看，w<1 是让输出在
x0 空间从 x0_cond 向 x0_uncond（"平均字"）回拉。书法场景下存在一个陷阱：

  - 清晰但**写错**的字  -> SSIM 很低（结构完全不同）
  - **模糊**但接近平均轮廓的字 -> SSIM 中等（轮廓大致对）

即 cfg<1 有可能是「把字变模糊」从而骗到更高 SSIM，而不是真的提升质量。
SSIM / LPIPS 都无法区分这两种情况，**只有人眼能**。

本脚本生成 N 个字 × M 个 cfg 的网格：
  - 每一行 = 一个 cfg（纵向对比同一批字随 cfg 的变化）
  - 每一列 = 同一个字（横向对比不同 cfg 下同一字的字形是否正确）
  - 第一列额外放 GT，作为基准

判读方法
--------
1. 横向看某一列：cfg 从小到大，字是否从"清晰但可能错"变成"模糊但轮廓对"？
   - 若是 -> cfg<1 是**变模糊骗分**，指标虚高
   - 若字形在各 cfg 下都正确，只是笔画质量变化 -> 是真实质量提升
2. 纵向看某一行：该 cfg 下 8 个字里有几个字形正确？统计正确率
3. 特别关注 cfg=0（纯 uncond）：若它也不错，说明条件贡献很小

用法（远程）
-----------
  python tools/sweep_cfg_visual.py --ckpt <best.pt> --config <resolved_config.json> \
      --cfgs 0.0 0.3 0.5 0.7 1.0 --n 8 --out /tmp/cfg_grid.png
"""
import os, sys, json, argparse, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
from PIL import Image


# ---------- 复刻 train.py 的模型构造 + DINO 注入（见 tools/sweep_eval_cfg.py）----------
def build_model_and_args(config_path, device):
    from types import SimpleNamespace
    from src.model import DiT_2Cond_models
    cfg = json.load(open(config_path, encoding="utf-8"))
    args = SimpleNamespace(**cfg)
    _dt = str(getattr(args, 'diffusion_type', 'ddpm')).lower()
    _ls = getattr(args, 'learn_sigma', None)
    _learn_sigma = bool(_ls) if _ls is not None else _dt not in ('flow', 'flow_matching', 'fm')

    model = DiT_2Cond_models[cfg["model"]](
        input_size=cfg.get("image_size", 256) // 8,
        num_calligraphers=cfg["num_calligraphers"],
        num_characters=cfg["num_characters"],
        use_checkpoint=False,
        learn_sigma=_learn_sigma,
        condition_fusion=cfg["condition_fusion"],
        callig_embed_dim=cfg["callig_embed_dim"],
        char_embed_dim=cfg["char_embed_dim"],
        cond_drop_all_prob=0.0,          # 推理时不 dropout
        cond_drop_one_prob=0.0,
        cond_drop_which_glyph_prob=cfg.get("cond_drop_which_glyph_prob", 0.5),
        skel_head_enabled=False,
        use_glyph_cond=False,
        glyph_scale_init=cfg.get("glyph_scale_init", 0.4),
        in_channels=cfg.get("latent_channels", 4),
        char_proj_mode=cfg.get("char_proj_mode", "mlp"),
        freeze_char_table=cfg.get("freeze_char_table", False),
        norm_type=cfg.get("norm_type", "rms"),
        mlp_type=cfg.get("mlp_type", "swiglu"),
        qk_norm=bool(cfg.get("qk_norm", True)),
        rope=bool(cfg.get("rope", True)),
        rope_theta=cfg.get("rope_theta", 100.0),
        attn_impl=cfg.get("attn_impl", "sdpa"),
    ).to(device)
    _inject_dino(model, cfg, device)
    return model, args, cfg


def _inject_dino(model, cfg, device):
    import re
    emb_path = cfg.get("char_dino_embeddings")
    idx_path = cfg.get("char_dino_index")
    if not (emb_path and idx_path and os.path.isfile(emb_path) and os.path.isfile(idx_path)):
        print("[dino-init] WARNING: dino files missing -> char table random")
        return
    _NUM_CH = 7026
    _emb = np.load(emb_path)
    _glyphs = json.load(open(idx_path, encoding="utf-8"))
    _glyphs = _glyphs.get("glyphs", _glyphs)
    _table = model.y_char_embedder.embedding_table.weight
    loaded = 0
    rows = []
    for gi, glyph in enumerate(_glyphs):
        if gi >= _emb.shape[0]:
            break
        sid, cid = (int(glyph[0]), int(glyph[1])) if isinstance(glyph, (list, tuple)) \
            else divmod(int(glyph), _NUM_CH)
        gid = sid * _NUM_CH + cid
        if gid < _table.shape[0] - 1:
            rows.append(gid)
            loaded += 1
    if cfg.get("dino_per_script_center", 0):
        sids = np.array([int(g[0]) if isinstance(g, (list, tuple)) else int(g) // _NUM_CH
                         for g in _glyphs])
        for s in np.unique(sids):
            m = sids == s
            if m.sum() > 1:
                _emb[m] -= _emb[m].mean(0, keepdims=True)
        _emb = _emb / np.maximum(np.linalg.norm(_emb, axis=1, keepdims=True), 1e-12)
    with torch.no_grad():
        for gi, glyph in enumerate(_glyphs):
            if gi >= _emb.shape[0]:
                break
            sid, cid = (int(glyph[0]), int(glyph[1])) if isinstance(glyph, (list, tuple)) \
                else divmod(int(glyph), _NUM_CH)
            gid = sid * _NUM_CH + cid
            if gid < _table.shape[0] - 1:
                _table[gid] = torch.from_numpy(_emb[gi]).to(_table.dtype)
        if cfg.get("dino_fill_unknown", 1) and rows:
            n = model.y_char_embedder.num_classes
            mv = torch.from_numpy(_emb.mean(0)).to(_table.device, _table.dtype)
            mv = mv / (mv.norm() + 1e-12)
            known = set(rows)
            unk = [r for r in range(n) if r not in known]
            if unk:
                _table.index_copy_(0, torch.tensor(unk, device=_table.device),
                                   mv.unsqueeze(0).expand(len(unk), -1))
    print(f"[dino-init] injected {loaded} glyph embeddings")


def load_ckpt(model, path):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict):
        for k in ("model", "ema"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                print(f"[load] using '{k}'")
                break
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(miss)} unexpected={len(unexp)}")


@torch.no_grad()
def sample_rows(model, args, cache, cfg_scale, device, steps, dit_batch):
    from src.loss import create_diffusion_or_flow, flow_kwargs_from
    from src.eval.in_process_eval import load_eval_vae
    diff = create_diffusion_or_flow(str(steps),
                                    diffusion_type=getattr(args, 'diffusion_type', 'flow'),
                                    **flow_kwargs_from(args))
    vae = load_eval_vae(args, device)
    n, lc, ls, sf = cache["n"], cache["latent_channels"], cache["latent_spatial"], cache["scaling_factor"]
    conds, noise = cache["conds"], cache["noise"]
    model.eval()
    torch.manual_seed(0)
    lat = torch.zeros(n, lc, ls, ls)
    for i in range(0, n, dit_batch):
        j = min(i + dit_batch, n)
        z = noise[i:j].to(device)
        yc = torch.tensor([c[0] for c in conds[i:j]], device=device, dtype=torch.long)
        yh = torch.tensor([c[1] for c in conds[i:j]], device=device, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s = diff.ddim_sample_loop(model.forward_with_cfg, z.shape, z,
                                      clip_denoised=False,
                                      model_kwargs=dict(y_callig=yc, y_char=yh,
                                                        cfg_scale=cfg_scale),
                                      device=device)
        lat[i:j] = s.float().cpu()
        del z, s
        torch.cuda.empty_cache()
    outs = []
    for i in range(0, n, 16):
        d = vae.decode(lat[i:i+16].to(device) / sf).sample
        outs.append(d.float().cpu())
        torch.cuda.empty_cache()
    return torch.cat(outs).clamp(-1, 1)


def to_pil(t):
    a = ((t.clamp(-1, 1) + 1) / 2 * 255).clamp(0, 255).byte()
    a = a[0].permute(1, 2, 0).numpy()
    return Image.fromarray(a, "RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-csv", default="5script/eval_fame_strict.csv")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--cfgs", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--cell", type=int, default=160)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/tmp/cfg_grid.png")
    a = ap.parse_args()

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    from src.eval.in_process_eval import prepare_eval_cache
    model, args, cfgj = build_model_and_args(a.config, dev)
    load_ckpt(model, a.ckpt)
    steps = a.steps or int(cfgj.get("eval_steps", 25))
    cache = prepare_eval_cache(a.eval_csv, cfgj.get("img_root", "final_imgs_256"),
                               cfgj.get("image_size", 256), a.n,
                               int(cfgj.get("vae_downscale", 8)),
                               int(cfgj.get("latent_channels", 4)),
                               float(cfgj.get("vae_scaling_factor", 0.18215)))

    gts = cache["gts"]
    rows_imgs = {}          # label -> list of PIL
    rows_imgs["GT"] = [to_pil(gts[k:k+1]) for k in range(cache["n"])]
    for c in a.cfgs:
        t0 = time.time()
        dec = sample_rows(model, args, cache, c, dev, steps, a.batch)
        rows_imgs[f"cfg={c:g}"] = [to_pil(dec[k:k+1]) for k in range(dec.shape[0])]
        print(f"  cfg={c:<5} done ({time.time()-t0:.0f}s)", flush=True)

    # 拼网格：第一列 GT，其余列各 cfg
    ncol = cache["n"]
    nrow = len(rows_imgs)
    cell, pad = a.cell, 6
    W = ncol * cell + (ncol + 1) * pad
    H = nrow * cell + (nrow + 1) * pad
    canvas = Image.new("RGB", (W, H), (250, 250, 250))
    for r, label in enumerate(rows_imgs):
        for c in range(ncol):
            im = rows_imgs[label][c].resize((cell, cell), Image.BICUBIC)
            canvas.paste(im, (pad + c * (cell + pad), pad + r * (cell + pad)))
    canvas.save(a.out)
    print(f"\nsaved -> {a.out}  ({W}x{H})")
    print("  行 = cfg（第一行为 GT 基准）；列 = 同一个字")
    for i, label in enumerate(rows_imgs):
        print(f"    row{i}: {label}")


if __name__ == "__main__":
    main()
