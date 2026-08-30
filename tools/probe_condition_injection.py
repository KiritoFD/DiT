# -*- coding: utf-8 -*-
"""
probe_condition_injection.py — 量化「条件到底对模型产生了多大作用」。

要回答的问题
------------
「DINO 不应该是一个坏的信号，是不是注入不对？」

当前注入链路（factorized_add）:
    e_callig = y_callig_embedder(y_callig)      # (N, 128)
    e_char   = y_char_embedder(y_char)          # (N, 384)  <- 冻结的 DINO 表
    y_emb    = (callig_proj(e_callig) + char_proj(e_char)) / sqrt(2)
    c        = t_emb + y_emb                    # (N, D)
    shift, scale, gate = adaLN_modulation(c).chunk(3)

三个可测的失效假说
------------------
H1 **幅度淹没**: y_emb 的 norm 远小于 t_emb -> 条件信号被时间步信号淹没
H2 **空间不可达**: c 只走 AdaLN（对 1024 个 token 施加**相同**的
   scale/shift），架构上无法传递"哪个位置画哪一笔"的信息
H3 **加法混合不可分**: 书家与字相加后共享同一向量，模型需自行分离

本脚本用**真实训练好的 ckpt** 实测 H1，并给出 H2 的间接证据。

测量项
------
1. 各阶段 norm: e_char / char_proj 输出 / e_callig / y_emb / t_emb / c
2. y_emb 对 adaLN 三元组 (shift, scale, gate) 的**边际贡献**:
   Δ = ||mod(c_full) - mod(c_without_cond)|| / ||mod(c_full)||
3. 不同字符间 y_emb 的**区分度**（余弦相似度分布）:
   若不同字的 y_emb 几乎相同 -> 字符条件确实没起作用

用法（远程）
-----------
  python tools/probe_condition_injection.py \
      --ckpt <best.pt> --config <resolved_config.json> --n 64
"""
import os, sys, json, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn.functional as F


def build_and_load(config_path, ckpt_path, device):
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
        use_checkpoint=False, learn_sigma=_learn_sigma,
        condition_fusion=cfg["condition_fusion"],
        callig_embed_dim=cfg["callig_embed_dim"],
        char_embed_dim=cfg["char_embed_dim"],
        cond_drop_all_prob=0.0, cond_drop_one_prob=0.0,
        cond_drop_which_glyph_prob=0.5,
        skel_head_enabled=False, use_glyph_cond=False,
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

    # DINO 注入（复刻 train.py）
    emb_path = cfg.get("char_dino_embeddings")
    idx_path = cfg.get("char_dino_index")
    if emb_path and idx_path and os.path.isfile(emb_path) and os.path.isfile(idx_path):
        _NUM_CH = 7026
        _emb = np.load(emb_path)
        _glyphs = json.load(open(idx_path, encoding="utf-8"))
        _glyphs = _glyphs.get("glyphs", _glyphs)
        tbl = model.y_char_embedder.embedding_table.weight
        rows = []
        for gi, g in enumerate(_glyphs):
            if gi >= _emb.shape[0]:
                break
            sid, cid = (int(g[0]), int(g[1])) if isinstance(g, (list, tuple)) else divmod(int(g), _NUM_CH)
            gid = sid * _NUM_CH + cid
            if gid < tbl.shape[0] - 1:
                rows.append(gid)
        if cfg.get("dino_per_script_center", 0):
            sids = np.array([int(g[0]) if isinstance(g, (list, tuple)) else int(g) // _NUM_CH
                             for g in _glyphs])
            for s in np.unique(sids):
                m = sids == s
                if m.sum() > 1:
                    _emb[m] -= _emb[m].mean(0, keepdims=True)
            _emb = _emb / np.maximum(np.linalg.norm(_emb, axis=1, keepdims=True), 1e-12)
        with torch.no_grad():
            for gi, g in enumerate(_glyphs):
                if gi >= _emb.shape[0]:
                    break
                sid, cid = (int(g[0]), int(g[1])) if isinstance(g, (list, tuple)) else divmod(int(g), _NUM_CH)
                gid = sid * _NUM_CH + cid
                if gid < tbl.shape[0] - 1:
                    tbl[gid] = torch.from_numpy(_emb[gi]).to(tbl.dtype)
            if cfg.get("dino_fill_unknown", 1) and rows:
                n = model.y_char_embedder.num_classes
                mv = torch.from_numpy(_emb.mean(0)).to(tbl.device, tbl.dtype)
                mv = mv / (mv.norm() + 1e-12)
                known = set(rows)
                unk = [r for r in range(n) if r not in known]
                if unk:
                    tbl.index_copy_(0, torch.tensor(unk, device=tbl.device),
                                    mv.unsqueeze(0).expand(len(unk), -1))
        print("[dino-init] glyph embeddings injected")

    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict):
        for k in ("model", "ema"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                print(f"[load] using '{k}' weights")
                break
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(miss)} unexpected={len(unexp)}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, args, cfg


@torch.no_grad()
def measure(model, conds, device, t_val=0.5):
    """测量条件链路各阶段 norm 与边际贡献。"""
    import math
    yc = torch.tensor([c[0] for c in conds], device=device, dtype=torch.long)
    yh = torch.tensor([c[1] for c in conds], device=device, dtype=torch.long)
    N = yc.shape[0]
    t = torch.full((N,), t_val, device=device)

    e_callig = model.y_callig_embedder(yc, False)
    e_char = model.y_char_embedder(yh, False)
    p_callig = model.callig_proj(e_callig)
    p_char = model.char_proj(e_char)
    y_emb = (p_callig + p_char) / math.sqrt(2.0)
    t_emb = model.t_embedder(t)
    c_full = t_emb + y_emb

    out = {
        "e_callig_norm": e_callig.norm(dim=-1).mean().item(),
        "e_char_norm": e_char.norm(dim=-1).mean().item(),
        "p_callig_norm": p_callig.norm(dim=-1).mean().item(),
        "p_char_norm": p_char.norm(dim=-1).mean().item(),
        "y_emb_norm": y_emb.norm(dim=-1).mean().item(),
        "t_emb_norm": t_emb.norm(dim=-1).mean().item(),
        "c_norm": c_full.norm(dim=-1).mean().item(),
    }
    out["ratio_y_over_t"] = out["y_emb_norm"] / max(out["t_emb_norm"], 1e-8)

    # ---- 边际贡献：y_emb 对 adaLN 三元组的影响 ----
    blk = model.blocks[0]
    mod = blk.adaLN_modulation
    full = mod(c_full)
    no_cond = mod(t_emb)                      # 去掉 y_emb
    only_callig = mod(t_emb + p_callig / math.sqrt(2.0))
    only_char = mod(t_emb + p_char / math.sqrt(2.0))

    names = ["shift", "scale", "gate"]
    for i, nm in enumerate(names):
        D = model.hidden_size if hasattr(model, "hidden_size") else full.shape[-1] // 3
        f = full[:, i*D:(i+1)*D]
        n0 = no_cond[:, i*D:(i+1)*D]
        na = only_callig[:, i*D:(i+1)*D]
        nb = only_char[:, i*D:(i+1)*D]
        out[f"delta_{nm}_full"] = ((f - n0).norm(dim=-1) / f.norm(dim=-1).clamp_min(1e-8)).mean().item()
        out[f"delta_{nm}_callig"] = ((na - n0).norm(dim=-1) / f.norm(dim=-1).clamp_min(1e-8)).mean().item()
        out[f"delta_{nm}_char"] = ((nb - n0).norm(dim=-1) / f.norm(dim=-1).clamp_min(1e-8)).mean().item()

    # ---- 区分度：不同字符的 y_emb 余弦相似度 ----
    yc_norm = F.normalize(y_emb, dim=-1)
    sim = yc_norm @ yc_norm.T
    iu = torch.triu_indices(N, N, offset=1)
    off = sim[iu[0], iu[1]]
    out["yemb_cos_sim_mean"] = off.mean().item()
    out["yemb_cos_sim_std"] = off.std().item()

    # 只看不同字符之间的相似度（排除同字）
    yh_arr = yh.cpu().numpy()
    diff_mask = torch.tensor(
        [[yh_arr[i] != yh_arr[j] for j in range(N)] for i in range(N)],
        device=device, dtype=torch.bool)
    dm = diff_mask & torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), 1)
    if dm.any():
        out["yemb_cos_sim_diff_char"] = sim[dm].mean().item()

    # char_proj 输出本身（未经 callig 混合）的区分度
    pc = F.normalize(p_char, dim=-1)
    simc = pc @ pc.T
    if dm.any():
        out["pchar_cos_sim_diff_char"] = simc[dm].mean().item()
    out["echar_cos_sim_diff_char"] = (
        F.normalize(e_char, dim=-1) @ F.normalize(e_char, dim=-1).T)[dm].mean().item() if dm.any() else None

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-csv", default="5script/eval_fame_strict.csv")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="5script/condition_probe.json")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")

    import csv, re
    rows = []
    with open(a.eval_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    conds = []
    seen_chars = set()
    for r in rows:
        ch = r.get("character")
        if ch in seen_chars:
            continue          # 尽量取不同字，便于测区分度
        seen_chars.add(ch)
        conds.append((int(r["calligrapher_id"]),
                      int(r.get("glyph_id", r.get("character_id", 0)))))
        if len(conds) >= a.n:
            break
    print(f"# using {len(conds)} samples, {len(seen_chars)} distinct characters")

    model, args, cfg = build_and_load(a.config, a.ckpt, dev)

    print("\n" + "=" * 72)
    print("1) 各阶段 norm（H1: 幅度淹没？）")
    print("=" * 72)
    r = measure(model, conds, dev, t_val=0.5)
    for k in ("e_callig_norm", "p_callig_norm", "e_char_norm", "p_char_norm",
              "y_emb_norm", "t_emb_norm", "c_norm"):
        print(f"  {k:<18} {r[k]:>12.4f}")
    print(f"\n  y_emb / t_emb = {r['ratio_y_over_t']:.4f}   "
          f"{'<< 1 条件可能被淹没' if r['ratio_y_over_t'] < 0.3 else 'OK' if r['ratio_y_over_t'] < 3 else '>1 条件过强'}")

    print("\n" + "=" * 72)
    print("2) y_emb 对 adaLN 三元组的边际贡献（相对变化量）")
    print("=" * 72)
    print(f"  {'':<10}{'callig only':>14}{'char only':>14}{'full':>14}")
    for nm in ("shift", "scale", "gate"):
        print(f"  {nm:<10}{r[f'delta_{nm}_callig']:>14.4f}"
              f"{r[f'delta_{nm}_char']:>14.4f}{r[f'delta_{nm}_full']:>14.4f}")

    print("\n" + "=" * 72)
    print("3) 区分度：不同字符的条件向量有多不同？")
    print("=" * 72)
    print(f"  DINO 表输出 e_char  余弦相似度(不同字) = {r.get('echar_cos_sim_diff_char', float('nan')):.4f}")
    print(f"  char_proj 输出       余弦相似度(不同字) = {r.get('pchar_cos_sim_diff_char', float('nan')):.4f}")
    print(f"  y_emb (混合后)       余弦相似度(不同字) = {r.get('yemb_cos_sim_diff_char', float('nan')):.4f}")
    print("  (越接近 1 = 不同字的条件几乎相同 = 字符条件失效)")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
