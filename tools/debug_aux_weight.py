# -*- coding: utf-8 -*-
"""aux 权重梯度 debug: 图像 4ch vs aux 8ch 的 loss/梯度占比 -> 定 aux_loss_weight.

用法: python tools/debug_aux_weight.py [--steps 30] [--batch 8]
输出: 每组 loss 均值 + 关键参数上的梯度范数 + 每输出通道梯度 + 推荐 w_aux (10%/20% 占比)
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from src.model import DiT_2Cond_models  # noqa: E402
from src.utils import MCCDLatentDataset  # noqa: E402
from src.utils.callig_map import load_callig_id_map  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="assets/train_fame3_e_full.csv")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--layers", type=int, default=4)
    args = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(0)
    m = DiT_2Cond_models["DiT-2Cond-S/2"](
        num_calligraphers=41, num_characters=35130, condition_fusion="factorized_add",
        callig_embed_dim=128, char_embed_dim=384, cond_drop_all_prob=0.05,
        cond_drop_one_prob=0.05, cond_drop_which_glyph_prob=0.85,
        use_glyph_cond=True, use_char_cond=False, glyph_scale_init=0.6,
        glyph_drop_prob=0.1, glyph_inject_layers=args.layers, glyph_embedder_depth=2,
        in_channels=12, learn_sigma=False).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.02)
    cmap, _ = load_callig_id_map("assets/callig_id_map.json")
    ds = MCCDLatentDataset(
        csv_file=args.csv, latent_shards_dir="data/latents/final_latents_fame_e",
        img_root="data/imgs/final_imgs_fame_e", preload=False, load_image=False,
        use_glyph_cond=True, skel_latent_shards_dir="data/skel/std_skel1_latents_fame_e",
        callig_id_map=cmap, is_train=True,
        aux_latent_shards_dirs=["data/aux/aux_skel_latents_fame_e", "data/aux/aux_canny_latents_fame_e"])
    print(f"dataset {len(ds)} rows; model in_ch={m.in_channels} layers={args.layers}")

    def get_batch(i0):
        items = [ds[i] for i in range(i0, i0 + args.batch)]
        x = torch.stack([it["latent"] for it in items]).to(dev)
        aux = torch.stack([it["aux_latents"] for it in items]).to(dev)
        y = torch.stack([it["y_callig"] for it in items]).to(dev)
        g = torch.stack([it["g"] for it in items]).to(dev).float()
        return torch.cat([x, aux], 1), y, g

    def probe(i0, tag):
        x0, y, g = get_batch(i0)
        t = torch.rand(x0.shape[0], 1, 1, 1, device=dev)
        noise = torch.randn_like(x0)
        x_t = (1 - t) * x0 + t * noise
        v = noise - x0
        pred = m(x_t, (t.flatten() * 1000.0), y, None, g=g)
        se = (pred - v).square()
        li = se[:, :4].mean()
        la = se[:, 4:].mean()
        # --- 梯度: 分别对 image / aux 反传, 比关键参数的梯度范数 ---
        def grad_norm(loss):
            opt.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            out = {}
            for n, p in m.named_parameters():
                if p.grad is None:
                    continue
                if n in ("final_layer.linear.weight", "x_embedder.proj.weight"):
                    out[n] = p.grad.detach().norm().item()
                    if n == "final_layer.linear.weight":
                        go = p.grad.detach()          # (out_dim, hidden)
                        per = go.reshape(12, 4, -1).norm(dim=(1, 2))
                        out["final_per_ch"] = per.tolist()
                elif "glyph_injections" in n and "proj.weight" in n:
                    out.setdefault("inj_proj", []).append(p.grad.detach().norm().item())
            return out
        gi = grad_norm(li)
        ga = grad_norm(la)
        opt.zero_grad(set_to_none=True)
        print(f"[{tag}] loss_img={li.item():.4f} loss_aux={la.item():.4f} "
              f"ratio_aux/img={la.item()/li.item():.3f}")
        print(f"      grad‖final_lin‖ img={gi['final_layer.linear.weight']:.3f} "
              f"aux={ga['final_layer.linear.weight']:.3f} "
              f"ratio={ga['final_layer.linear.weight']/gi['final_layer.linear.weight']:.3f}")
        print(f"      grad‖patch_embed‖ img={gi['x_embedder.proj.weight']:.3f} "
              f"aux={ga['x_embedder.proj.weight']:.3f}")
        if "final_per_ch" in ga:
            img_ch = np.mean(ga["final_per_ch"][:4])
            aux_ch = np.mean(ga["final_per_ch"][4:])
            print(f"      final per-channel grad ‖·‖ (aux反传): img={img_ch:.3f} aux={aux_ch:.3f}")
        if "inj_proj" in ga:
            print(f"      inj proj grad (aux反传) mean={np.mean(ga['inj_proj']):.4f} "
                  f"img={np.mean(gi.get('inj_proj',[float('nan')])):.4f}")
        return li.item(), la.item(), gi["final_layer.linear.weight"], ga["final_layer.linear.weight"]

    # 训练若干步 (看注入/主干是否在学 + 稳定)
    for step in range(1, args.steps + 1):
        i0 = ((step - 1) * args.batch) % max(len(ds) - args.batch, 1)
        x0, y, g = get_batch(i0)
        t = torch.rand(x0.shape[0], 1, 1, 1, device=dev)
        noise = torch.randn_like(x0)
        x_t = (1 - t) * x0 + t * noise
        v = noise - x0
        pred = m(x_t, (t.flatten() * 1000.0), y, None, g=g)
        loss = (pred - v).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step == 1:
            inj = [p.detach().abs().mean().item() for n, p in m.named_parameters()
                   if "glyph_injections" in n and "proj.weight" in n]
            print(f"  step{step:3d} loss={loss.item():.4f} gradnorm={gn:.3f} "
                  f"inj|w|mean={np.mean(inj):.2e} glyph_scale={m.glyph_scale.item():.3f}")

    li, la, gi_, ga_ = probe(0, "probe")
    print("\n=== 推荐 w_aux (按梯度占比) ===")
    for share in (0.10, 0.20, 0.30):
        w = share / max(ga_ / gi_, 1e-9)
        print(f"  aux 梯度占 {share*100:.0f}% -> w_aux ≈ {w:.3f}")
    print("DEBUG_DONE")


if __name__ == "__main__":
    main()
