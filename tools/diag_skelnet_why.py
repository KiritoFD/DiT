# -*- coding: utf-8 -*-
"""diag_skelnet_why.py — 定量回答 "SkelNet-DiT 为什么没学到"。

查四件事:
  A. 容量     : 参数量 / 训练 loss 是否已经平掉
  B. 信噪比   : x0(std) vs ε(std=1); 速度目标 v=ε-x0 里 x0 占多少方差
                -> 若 x0 占比极小, 则 flow loss 几乎"看不见"结构, 低 loss ≠ 学到
  C. latent 域是否学到: cos(生成, GT) 是否 > cos(输入std, GT)  (copy baseline)
  D. 输出是否塌成条件均值: std(生成) vs std(GT) (过平滑) + 墨量比

只读 + 推理, 不训练。
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch as th

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TIME_SCALE = 1000.0


def cos(a, b):
    a = a.ravel().float()
    b = b.ravel().float()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="assets/skelnet_dit_B_w3.pt.step024000")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()

    sd = th.load(a.ckpt, map_location="cpu", weights_only=False)
    A = sd.get("args", {})
    A = dict(A) if not isinstance(A, argparse.Namespace) else vars(A)
    n_slots = int(sd.get("n_slots", 0))
    print(f"[ckpt] step={sd.get('step')}  n_slots={n_slots}  tgt={A.get('tgt_shards')}")

    from src.model.dit import DiT_2Cond
    model = DiT_2Cond(
        input_size=32, patch_size=2, in_channels=4, out_channels=4,
        depth=int(A.get("depth", 6)), hidden_size=int(A.get("hidden", 256)),
        num_heads=int(A.get("heads", 4)), num_calligraphers=max(n_slots, 1),
        num_characters=1, use_char_cond=False, use_glyph_cond=True,
        glyph_in_channels=4, glyph_inject_layers=int(A.get("inject_layers", 2)),
        glyph_inject_mode="adaln", glyph_scale_init=0.6, glyph_drop_prob=0.0,
        glyph_embedder_depth=2, condition_fusion="factorized_cat",
        callig_embed_dim=128, glyph_vec_cond=True, glyph_vec_dim=128,
        cond_drop_all_prob=float(A.get("style_drop", 0.1)),
        cond_drop_one_prob=0.0, learn_sigma=False,
    ).cuda().eval()
    model.load_state_dict({k: v.cuda() for k, v in sd["ema"].items()}, strict=False)
    npar = sum(p.numel() for p in model.parameters())
    print(f"[A 容量] 参数量 = {npar/1e6:.2f}M  depth={A.get('depth')} "
          f"hidden={A.get('hidden')} inject={A.get('inject_layers')}")

    from src.utils.latent_dataset import MCCDLatentDataset
    csmap = None
    if os.path.exists(A.get("callig_map", "")):
        csmap = json.load(open(A["callig_map"], encoding="utf-8"))
    ds = MCCDLatentDataset(
        csv_file=A.get("val_csv"), latent_shards_dir=A.get("tgt_shards"),
        img_root="", image_size=256, is_train=False, preload=True,
        load_image=False, skel_latent_shards_dir=A.get("cond_shards"),
        callig_id_map=None, callig_script_map=csmap)
    n = min(a.n, len(ds))
    print(f"[data] val {len(ds)}, 取 {n}")

    # ── B. 信噪比 ──
    xs = th.stack([ds[i]["latent"] for i in range(min(n, 256))])
    sx = float(xs.std())
    vx = float(xs.var())
    print(f"\n[B 信噪比]")
    print(f"  x0 (GT骨架 latent) std = {sx:.4f}, var = {vx:.5f}")
    print(f"  eps ~ N(0,1)      std = 1.0000, var = 1.00000")
    print(f"  速度目标 v = eps - x0 : var = 1 + {vx:.5f} = {1+vx:.5f}")
    print(f"  ★ x0 占 v 的方差比 = {vx/(1+vx)*100:.2f}%")
    print(f"  -> 若模型完全不学 x0, 只把噪声学好, loss 也能降到接近 (噪声残差) 的水平;"
          f"  x0 最多只贡献 {vx:.3f} 的 MSE")

    # ── C/D. 生成 vs copy-input ──
    steps = int(A.get("sample_steps", a.steps))
    pred_mode = A.get("pred", "v")
    cg, cs, sg_, sx_, se = [], [], [], [], []
    mse_g, mse_s = [], []
    for i in range(n):
        b = ds[i]
        x0 = b["latent"].float().cuda()[None]
        g = b["skel_latent"].float().cuda()[None]
        y = th.tensor([int(b["y_callig"])], device="cuda")
        z = th.randn_like(x0)
        eps = z.clone()
        ts = th.linspace(1.0, 0.0, steps + 1, device="cuda")
        with th.no_grad():
            for k in range(steps):
                out = model(z, th.full((1,), float(ts[k]) * TIME_SCALE, device="cuda"),
                            y_callig=y, y_char=th.zeros_like(y), g=g)
                if isinstance(out, tuple):
                    out = out[0]
                if pred_mode == "x0":
                    z = (1 - ts[k + 1]) * out + ts[k + 1] * eps
                else:
                    z = z + (ts[k + 1] - ts[k]) * out
        cg.append(cos(z[0], x0[0]))
        cs.append(cos(g[0], x0[0]))
        sg_.append(float(z[0].std()))
        sx_.append(float(x0[0].std()))
        se.append(float(z[0].mean()))
        mse_g.append(float((z[0] - x0[0]).pow(2).mean()))
        mse_s.append(float((g[0] - x0[0]).pow(2).mean()))

    m = lambda v: float(np.mean(v))
    print(f"\n[C latent 域: 生成 vs 照抄输入]")
    print(f"  cos(生成, GT)   = {m(cg):+.4f}")
    print(f"  cos(输入std,GT) = {m(cs):+.4f}   <- copy baseline")
    print(f"  ★ 生成{'优于' if m(cg) > m(cs) else '**劣于**'}照抄输入 (差 {m(cg)-m(cs):+.4f})")
    print(f"  MSE(生成,GT)    = {m(mse_g):.5f}")
    print(f"  MSE(输入std,GT) = {m(mse_s):.5f}")

    print(f"\n[D 是否塌成条件均值(过平滑)]")
    print(f"  std: 生成 {m(sg_):.4f} | GT {m(sx_):.4f} | 比 {m(sg_)/m(sx_):.3f}")


if __name__ == "__main__":
    main()
