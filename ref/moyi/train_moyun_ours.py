#!/usr/bin/env python
"""train_moyun_ours.py —— moyun(Rectified Flow) 在我们数据上的复现训练脚本。

## 与原版 train_moyun2_RF.py 的差异
1. **数据**: 不用 ImageFolder，用 `dataset_moyun.MultiLabelNestedDataset`
   （csv + 预编码 latent shards）—— 直接返回 latent，**跳过训练循环里的 VAE encode**
2. **12ch**: image/edge/skeleton 各 4ch，来自 shards_img / shards_aux_canny / shards_std_fixed
3. **单卡**: 不用 DDP（原版 torchrun 2 卡），DDP 分支保留但不启用
4. **冒烟**: --smoke N 跑 N 步就退出并打印显存

用法:
  PYTHONPATH=ref/moyi python ref/moyi/train_moyun_ours.py \
      --model moyun-12channel-B --csv assets/train_50k_v2_fixed.csv \
      --global-batch-size 24 --smoke 20
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "moyun"))

from dataset_moyun import MultiLabelNestedDataset  # noqa: E402
from moyun_2 import DiT_models  # noqa: E402
from utils.REPA_diffusion import create_diffusion  # noqa: E402


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="moyun-12channel-B")
    p.add_argument("--csv", default="assets/train_50k_v2_fixed.csv")
    p.add_argument("--img-shards", default="data/50k/shards_img")
    p.add_argument("--edge-shards", default="data/50k/shards_aux_canny")
    p.add_argument("--skel-shards", default="data/50k/shards_std_fixed")
    p.add_argument("--results-dir", default="assets/results/moyun_repro")
    p.add_argument("--global-batch-size", type=int, default=24)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-steps", type=int, default=155000)
    p.add_argument("--ckpt-every", type=int, default=5000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--num-classes", type=int, default=6000,
                   help="三个 Embedding 表的类别数（要 >= max(书家id, 书体id, 字id)+1）")
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--if-rope", type=int, default=0)
    p.add_argument("--without-t", type=int, default=0)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--smoke", type=int, default=0, help=">0 时跑这么多步就退出")
    # ★ 对齐 ref _ful.sh 的分维度条件 dropout
    p.add_argument("--charactor-x", type=float, default=0.08)
    p.add_argument("--font-x", type=float, default=0.08)
    p.add_argument("--calligrapher-x", type=float, default=0.16)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    a = get_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    os.makedirs(a.results_dir, exist_ok=True)

    print(f"[moyun] model={a.model} batch={a.global_batch_size} "
          f"steps={a.max_steps}", flush=True)

    # ── 数据 ─────────────────────────────────────────────────────────
    ds = MultiLabelNestedDataset(
        csv_file=a.csv, img_shards=a.img_shards,
        edge_shards=a.edge_shards, skel_shards=a.skel_shards,
        num_classes=a.num_classes)
    print(f"[moyun] dataset {len(ds):,} 条", flush=True)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=a.global_batch_size, shuffle=True,
        num_workers=a.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(a.num_workers > 0))

    # ── 模型 ─────────────────────────────────────────────────────────
    model = DiT_models[a.model](
        input_size=a.image_size, num_classes=a.num_classes,
        learn_sigma=True, if_rope=bool(a.if_rope)).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[moyun] params {n_par/1e6:.1f}M", flush=True)
    ema = torch.optim.swa_utils.AveragedModel(
        model, avg_fn=lambda e, m, n: a.ema_decay * e + (1 - a.ema_decay) * m)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0)
    # ★ 按 ref 生产脚本（_ful.sh -> train_moyun2_diffusion_repa.py）用 DDPM
    diffusion = create_diffusion(
        timestep_respacing="",
        learn_sigma=True,
        use_black_white_mse_loss=False,
        use_grey_mse_loss=False)

    model.train()
    step = 0
    t0 = time.time()
    loss_sum = 0.0
    for epoch in range(10 ** 6):
        for batch in loader:
            image, edge, skel, y, stroke, f1, f2, f3 = batch
            x = torch.cat([image, edge, skel], dim=1).to(dev, non_blocking=True)
            # ⚠ LabelEmbedder 内部 labels.T 后取 [0]/[1]/[2]，
            #   所以传进去的必须是 (B,3)（dataloader 给的是 (B,3,1)）。不要转置！
            y = y.squeeze(-1).contiguous().to(dev)
            stroke = stroke.to(dev)
            # ★ 分维度条件 dropout —— 必须对齐 ref 生产脚本 _ful.sh:
            #     --charactor-x 0.08 --font-x 0.08 --calligrapher-x 0.16
            #   (calligrapher 的丢弃率是内容的 2 倍)
            fdict = {"calligrapher": f1.to(dev), "font": f2.to(dev),
                     "charactor": f3.to(dev),
                     "calligrapher_x": a.calligrapher_x,
                     "font_x": a.font_x, "charactor_x": a.charactor_x}

            # ★ DDPM：随机采 t，training_losses(model, x, t, feature_dict, model_kwargs)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],),
                              device=dev)
            fdict["_num_classes"] = a.num_classes
            loss_dict = diffusion.training_losses(
                model, x, t, fdict, {"y": y, "stroke": stroke})
            loss = loss_dict["loss"].mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ema.update_parameters(model)

            loss_sum += loss.item()
            step += 1
            if step % a.log_every == 0:
                el = time.time() - t0
                mem = (torch.cuda.max_memory_allocated() / 1e9
                       if dev.type == "cuda" else 0)
                print(f"  step={step:07d} loss={loss_sum/a.log_every:.4f} "
                      f"{a.log_every/max(el,1e-9):.2f} step/s  "
                      f"peak_mem={mem:.1f}G", flush=True)
                loss_sum = 0.0
                t0 = time.time()

            if a.smoke and step >= a.smoke:
                print(f"\n[smoke] ✓ 跑通 {step} 步，无异常", flush=True)
                if dev.type == "cuda":
                    print(f"[smoke] 峰值显存 "
                          f"{torch.cuda.max_memory_allocated()/1e9:.2f}G", flush=True)
                return
            if step % a.ckpt_every == 0:
                p = os.path.join(a.results_dir, f"moyun_{step:07d}.pt")
                torch.save({"model": model.state_dict(),
                            "ema": ema.module.state_dict(), "step": step}, p)
                print(f"  [ckpt] {p}", flush=True)
            if step >= a.max_steps:
                print(f"[moyun] 到达 max_steps={a.max_steps}", flush=True)
                return


if __name__ == "__main__":
    main()
