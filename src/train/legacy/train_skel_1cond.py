# -*- coding: utf-8 -*-
"""
train_skel_1cond.py — 无 char 条件的骨架引导生成训练 (DiT-1CondSkel)。

独立入口 (不影响 train.py/train_controlnet.py):
  * 模型: src/model/legacy/dit_skel_1cond.py 的 DiT-1CondSkel (输入=img+skel concat, 条件=callig)
  * 数据: train_fame_clean_v8.csv, 需要 image latent + skel latent 两种 shards
  * loss: Flow Matching (logit_normal t, OT) + 可选 REPA (公共 src.loss.repa)
  * eval: 统一 eval_facade (同进程指标+落盘)
  * 早停: ssim (eval_auto_*)

用法:
  python src/train/legacy/train_skel_1cond.py --config src/train/configs/v10_skel1cond.json
"""
import os, sys, json, math, time, glob, datetime
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.model.legacy.dit_skel_1cond import DiT_1CondSkel
from src.loss import create_diffusion_or_flow, flow_kwargs_from
from src.eval.eval_facade import build_eval_cache, eval_run
from src.utils.logger import get_logger  # 若无则退化

try:
    logger = get_logger("train_skel_1cond")
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger("train_skel_1cond")


def build_dataloader(args):
    """MCCD latent dataset: image latent + skel latent (均 v8 shards), 无 char 条件。"""
    from src.utils.latent_dataset import MCCDLatentDataset as _DS
    ds = _DS(
        csv_file=args.csv, latent_shards_dir=args.latent_shards_dir,
        skel_latent_shards_dir=args.skel_latent_shards_dir,
        img_root=getattr(args, "img_root", None),
        image_size=int(getattr(args, "image_size", 256)),
        is_train=True,
        preload=bool(getattr(args, 'preload', True)),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True)
    return loader


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args, _ = ap.parse_known_args()
    if args.config:
        cfg = json.load(open(args.config, encoding="utf-8"))
        for k, v in cfg.items():
            setattr(args, k, v)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiT_1CondSkel(
        num_calligraphers=args.num_calligraphers,
        hidden_size=int(getattr(args, "hidden_size", 384)),
        depth=int(getattr(args, "depth", 12)),
        num_heads=int(getattr(args, "num_heads", 6)),
        patch_size=int(getattr(args, "patch_size", 2)),
        learn_sigma=False,
        cond_drop_prob=float(getattr(args, "cond_drop_prob", 0.1)),
        skel_drop_prob=float(getattr(args, "skel_drop_prob", 0.1)),
    ).to(device)
    logger.info(f"[model] DiT-1CondSkel-S/{args.patch_size} "
                f"({sum(p.numel() for p in model.parameters())/1e6:.1f}M, 无 char)")

    # 统一 REPA (可选)
    repa = None
    if getattr(args, "w_repa", 0) > 0:
        from src.loss.repa import build_repa_module
        repa = build_repa_module(
            student_dim=model.hidden_size,
            layers=tuple(int(x) for x in str(getattr(args, 'repa_layers', '8')).split(',') if x.strip()),
            teacher_ckpt=getattr(args, "repa_teacher_ckpt", "") or None,
            w_repa=float(args.w_repa),
            warmup_steps=int(getattr(args, "repa_warmup", 0) or 0),
            device=device)
        logger.info(f"[repa] w={args.w_repa} layers={repa.layers}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    if repa is not None:
        trainable += repa.trainable_params()
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    diffusion = create_diffusion_or_flow(
        timestep_respacing="", diffusion_type=args.diffusion_type,
        **flow_kwargs_from(args))
    loader = build_dataloader(args)

    # eval cache
    eval_cache = None
    eval_vae_batch = int(getattr(args, "eval_vae_batch", 16))
    if getattr(args, "gpu_eval_csv", "") and os.path.exists(args.gpu_eval_csv):
        eval_cache = build_eval_cache(
            args.gpu_eval_csv, args.gpu_eval_img_root, None, 256,
            int(getattr(args, "gpu_eval_n", 100)), 8, 4, 0.18215,
            skel_latent_shards_dir=args.gpu_eval_skel_latent_shards_dir or None)

    exp_dir = os.path.join(args.results_dir, f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{args.experiment_name}")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        prog = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
        return max(getattr(args, "min_lr_ratio", 0.1), 0.5 * (1 + math.cos(math.pi * prog)))
    sched = LambdaLR(opt, lr_lambda)

    train_steps = 0
    t0 = time.time()
    best_ssim = -1
    for epoch in range(args.epochs):
        for batch in loader:
            if train_steps >= args.max_steps:
                break
            x_latent = batch["latent"].to(device)
            skel_lat = batch["skel_latent"].to(device)
            y_callig = batch["y_callig"].to(device)
            img = batch.get("image")
            img = img.to(device) if img is not None and img.numel() else None

            t = diffusion.sample_t(x_latent.shape[0], device)
            model_kwargs = dict(y_callig=y_callig, skel=skel_lat)
            if repa is not None:
                if len(repa.layers) > 1:
                    model_kwargs["return_intermediate_layers"] = repa.layers
                else:
                    model_kwargs["return_intermediate_layer"] = repa.layers[0]

            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss_dict = diffusion.training_losses(model, x_latent, t, model_kwargs)
                loss = loss_dict["loss"].mean()
                if repa is not None and img is not None:
                    int_feats = loss_dict.get("intermediate_feats")
                    if int_feats is not None:
                        loss = loss + repa(int_feats, img, step=train_steps)

            if not torch.isfinite(loss):
                logger.warning(f"[nan] step {train_steps}; skip")
                train_steps += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            opt.step()
            sched.step()
            train_steps += 1

            if train_steps % args.log_every == 0:
                logger.info(f"(step={train_steps:07d}) Total: {loss.item():.4f} "
                            f"LR: {opt.param_groups[0]['lr']:.2e} "
                            f"Steps/Sec: {train_steps/(time.time()-t0):.2f} "
                            f"Mem: {torch.cuda.memory_reserved()/1024**3:.1f}G")

            # eval
            if eval_cache is not None and train_steps % args.gpu_eval_every == 0:
                try:
                    eval_run(model, diffusion, eval_cache, device, train_steps, ckpt_dir,
                             ddim_steps=int(getattr(args, "gpu_eval_steps", 50)),
                             cfg_scale=float(getattr(args, "gpu_eval_cfg", 0.7)),
                             dit_batch=int(getattr(args, "gpu_eval_dit_batch", 8)),
                             vae_batch=eval_vae_batch, with_skel=False)
                    import glob as _g
                    evs = sorted(_g.glob(os.path.join(ckpt_dir, "eval_auto_*.json")))
                    if evs:
                        d = json.load(open(evs[-1]))
                        s = d.get("ssim", -1)
                        if s > best_ssim:
                            best_ssim = s
                            torch.save({"model": model.state_dict(), "train_steps": train_steps},
                                       os.path.join(ckpt_dir, "best.pt"))
                            logger.info(f"[eval] best ssim={s:.4f} @{train_steps}")
                except Exception as e:
                    logger.warning(f"[eval] FAILED: {e}")

            if train_steps % args.ckpt_every == 0:
                torch.save({"model": model.state_dict(), "train_steps": train_steps},
                           os.path.join(ckpt_dir, f"{train_steps:07d}.pt"))

    logger.info("Done!")


if __name__ == "__main__":
    main()