#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""debug_sp2_callig.py — sp2 的 callig(书家风格) 通路诊断 (无数据依赖).

回答 "loss scale-up 后降不下去, 字=书家够不够, 是不是梯度流向/模型设计问题":
  1. 生命信号  : glyph_scale / callig_scale 学到多少 (两条条件通路的活性)
  2. embedding 塌缩 : 1013 书家 callig embedding 是否塌成单一向量 (风格信息是否丢失)
  3. 注入幅度  : callig 条件 add 进 t_emb 后与 t_emb 的相对幅度

用法 (远程): python tools/debug_sp2_callig.py --ckpt 0022500
"""
import argparse, glob, os, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.environ.get("DIIT_ROOT", "/root/Workspace/xy/DiT")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    specs = sorted(glob.glob("5script/results/v10b_stdskel_fame3_sp2/*/checkpoints/*.pt"),
                   key=lambda p: int(os.path.basename(p).split(".")[0]))
    ck_path = (next((p for p in specs if int(os.path.basename(p).split(".")[0]) == int(args.ckpt)), specs[-1])
               if args.ckpt.isdigit() else args.ckpt)
    step = int(os.path.basename(ck_path).split(".")[0])
    print(f"[ckpt] {ck_path} (step {step})", flush=True)

    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    a = ck.get("args", {}) or {}
    if not isinstance(a, dict):
        a = vars(a) if hasattr(a, "__dict__") else {}

    from src.model import DiT_2Cond_models
    arch = dict(norm_type=a.get("norm_type", "rms"), mlp_type=a.get("mlp_type", "swiglu"),
                qk_norm=bool(a.get("qk_norm", 1)), rope=bool(a.get("rope", 1)),
                rope_theta=float(a.get("rope_theta", 100.0)), attn_impl="sdpa")
    model = DiT_2Cond_models[a.get("model", "DiT-2Cond-Sp/2")](
        num_calligraphers=int(a.get("num_calligraphers", 1013)),
        num_characters=int(a.get("num_characters", 35130)),
        condition_fusion=a.get("condition_fusion", "factorized_add"),
        callig_embed_dim=int(a.get("callig_embed_dim", 128)),
        char_embed_dim=int(a.get("char_embed_dim", 384)),
        char_proj_mode=a.get("char_proj_mode", "mlp"),
        freeze_char_table=bool(a.get("freeze_char_table", False)),
        cond_drop_all_prob=0.1, cond_drop_one_prob=0.4,
        cond_drop_which_glyph_prob=0.85, use_checkpoint=False, learn_sigma=False,
        use_glyph_cond=True, use_char_cond=not bool(a.get("no_char_cond", False)),
        glyph_scale_init=float(a.get("glyph_scale_init", 0.6)),
        glyph_drop_prob=float(a.get("glyph_drop_prob", 0.1)),
        glyph_inject_layers=int(a.get("glyph_inject_layers", 4)),
        glyph_embedder_depth=int(a.get("glyph_embedder_depth", 2)), **arch)
    sd = ck.get("ema") or ck.get("model") or ck
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"[model] {a.get('model')} miss={len(miss)} unexp={len(unexp)}", flush=True)

    # ── 1. 生命信号 ──
    gs = getattr(model, "glyph_scale", None)
    cs = getattr(model, "callig_scale", None)
    print(f"\n[1] 生命信号")
    print(f"  glyph_scale   = {gs.item():.4f} (init {a.get('glyph_scale_init')})")
    if cs is not None:
        print(f"  callig_scale  = {cs.item():.4f} (init {a.get('callig_scale_init', 1.0)})")

    # ── 2. callig embedding 塌缩检查 ──
    print(f"\n[2] callig embedding 塌缩检查")
    emb = None
    for name, p in model.named_parameters():
        if "callig" in name and "embed" in name and "weight" in name:
            emb = p.detach().clone()
            break
    if emb is None:
        for name, p in model.named_parameters():
            if "y_callig" in name and "weight" in name:
                emb = p.detach().clone(); break
    if emb is not None:
        emb = emb.float()
        nc = emb.shape[0]
        mean_vec = emb.mean(0, keepdim=True)
        # 类间 vs 类内: 用"离均值距离的分布"衡量塌缩
        dist = (emb - mean_vec).norm(dim=1)
        total_var = float(((emb - mean_vec) ** 2).sum() / nc)
        # 归一化 pairwise 余弦
        e = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
        sim = e @ e.T
        off = sim[~torch.eye(nc, dtype=torch.bool)]
        print(f"  embedding 形状 = {tuple(emb.shape)}")
        print(f"  逐维方差之和(total_var) = {total_var:.4f}  (0=完全塌缩到单点)")
        print(f"  离均值距离 std/mean = {dist.std():.4f}/{dist.mean():.4f}  (std≈0=塌缩)")
        print(f"  归一化 pairwise 余弦 = mean {off.mean():.4f} / std {off.std():.4f}  (mean→1=全同)")
        print(f"  ||emb|| 分布 = mean {emb.norm(dim=1).mean():.3f} ± {emb.norm(dim=1).std():.3f}")
    else:
        print("  [未找到 callig embedding 权重]")

    # ── 3. 注入幅度 (callig 条件 vs t_emb 的幅度) ──
    print(f"\n[3] callig 注入幅度")
    try:
        with torch.no_grad():
            t_vec = torch.full((16,), 0.5)
            t_emb = model.t_embedder(t_vec * 1000.0)
            y = torch.randint(0, int(a.get("num_calligraphers", 1013)), (16,))
            e_callig = model.y_callig_embedder(y, False)
            c_callig = model.callig_proj(e_callig)
            if cs is not None:
                c_callig = cs * c_callig
            print(f"  ||t_emb||      = {t_emb.norm(dim=-1).mean():.3f}")
            print(f"  ||callig||     = {c_callig.norm(dim=-1).mean():.3f}")
            print(f"  占比 callig/t  = {c_callig.norm(dim=-1).mean() / t_emb.norm(dim=-1).mean():.4f}")
            # callig 对不同书家的区分度 (同一 t, 不同 y)
            y2 = torch.randint(0, int(a.get("num_calligraphers", 1013)), (16,))
            e2 = model.y_callig_embedder(y2, False)
            c2 = model.callig_proj(e2)
            d = (c_callig - c2).norm(dim=-1).mean()
            print(f"  随机两书家 callig 向量平均距离 = {d:.4f} (0=无法区分书家)")
    except Exception as ex:
        print(f"  [注入幅度诊断失败: {ex}]")

    print(f"\n== 解读 ==")
    print("  callig_scale 降/不动 + embedding 塌缩 + 两书家距离≈0 => 风格通路断裂(梯度/优化问题)")
    print("  embedding 分化好但风格仍统一          => 注入机制问题(信息算了但没进主干)")


if __name__ == "__main__":
    main()