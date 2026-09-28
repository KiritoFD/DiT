# -*- coding: utf-8 -*-
"""VQ-Font training on top10_style23 — thin re-implementation of the repo's
train.py with two changes:
  1) chars_sim_dict comes from SimMatrix (npy) instead of the missing JSON;
  2) 256x256 via cfg (input_size), everything else faithful (CombinedTrainer,
     Evaluator, GAN losses, codebook from pretrain_vqvae.py).

Run from baseline/VQ-Font:
  python ../adapter/vqfont/train_top10.py top10 ../adapter/vqfont/cfg_top10.yaml \
      [--resume work_dir/checkpoints/top10/xxxx.pth]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from sconf import Config, dump_args

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # adapter/vqfont
sys.path.insert(0, os.getcwd())                                 # VQ-Font repo

import utils  # noqa: E402
from utils import Logger  # noqa: E402
from datasets import (load_lmdb, load_json, read_data_from_lmdb,  # noqa: E402
                      get_comb_trn_loader, get_cv_comb_loaders)
from trainer import load_checkpoint, CombinedTrainer  # noqa: E402
from model import generator_dispatch, disc_builder  # noqa: E402
from model.modules import weights_init  # noqa: E402
from evaluator import Evaluator  # noqa: E402
from top10_sim import SimMatrix  # noqa: E402


def load_pretrain_vae_model(load_path, gen):
    vae_state_dict = torch.load(load_path, map_location="cuda:0")
    component_objects = vae_state_dict["_vq_vae._embedding.weight"]
    del_key = [key for key, _ in vae_state_dict.items() if "encoder" in key]
    i = 0
    for param in gen.content_encoder.parameters():
        param.data = vae_state_dict[del_key[i]]
        i += 1
        param.requires_grad = False
    return component_objects


def setup(args):
    cfg = Config(*args.config_paths, default="cfgs/defaults.yaml",
                 colorize_modified_item=True)
    cfg.argv_update(args.left_argv)
    cfg.work_dir = Path(cfg.work_dir)
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    cfg.unique_name = cfg.name = args.name
    (cfg.work_dir / "logs").mkdir(parents=True, exist_ok=True)
    (cfg.work_dir / "checkpoints" / args.name).mkdir(parents=True, exist_ok=True)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("config_paths", nargs="+")
    ap.add_argument("--resume", default=None)
    args, left_argv = ap.parse_known_args()
    args.left_argv = left_argv
    cfg = setup(args)

    logger = Logger.get(file_path=cfg.work_dir / "logs" / f"{cfg.unique_name}.log",
                        level="info", colorize=True)
    logger.info("Run Argv:\n> {}".format(" ".join(sys.argv)))
    logger.info("Configs:\n{}".format(cfg.dumps()))

    size = cfg.input_size
    t_tf = [transforms.Resize((size, size)), transforms.ToTensor()]
    if cfg.dset_aug.normalize:
        t_tf.append(transforms.Normalize([0.5], [0.5]))
        cfg.g_args.dec.out = "tanh"
    trn_transform = transforms.Compose(t_tf)
    val_transform = transforms.Compose(t_tf)

    env = load_lmdb(cfg.data_path)
    env_get = lambda env, x, y, transform: transform(
        read_data_from_lmdb(env, f"{x}_{y}")["img"])  # noqa: E731
    data_meta = load_json(cfg.data_meta)

    trn_dset, trn_loader = get_comb_trn_loader(env, env_get, cfg, data_meta["train"],
                                               trn_transform, num_workers=cfg.n_workers,
                                               shuffle=True, drop_last=True)
    cv_loaders = get_cv_comb_loaders(env, env_get, cfg, data_meta, val_transform,
                                     num_workers=0, shuffle=False, drop_last=True)

    logger.info("Build Few-shot model ...")
    g_kwargs = cfg.get("g_args", {})
    g_cls = generator_dispatch()
    gen = g_cls(1, cfg.C, 1, cfg, **g_kwargs)
    gen.cuda()
    gen.apply(weights_init(cfg.init))

    logger.info("Load pre-train model ...")
    component_objects = load_pretrain_vae_model(cfg.vae_pth, gen)

    if cfg.gan_w > 0.:
        d_kwargs = cfg.get("d_args", {})
        disc = disc_builder(cfg.C, trn_dset.n_fonts, trn_dset.n_unis, **d_kwargs)
        disc.cuda()
        disc.apply(weights_init(cfg.init))
    else:
        disc = None

    g_optim = optim.Adam(gen.parameters(), lr=cfg.g_lr, betas=cfg.adam_betas)
    d_optim = optim.Adam(disc.parameters(), lr=cfg.d_lr, betas=cfg.adam_betas)
    gen_scheduler = torch.optim.lr_scheduler.StepLR(g_optim, step_size=cfg["step_size"],
                                                    gamma=cfg["gamma"])
    dis_scheduler = torch.optim.lr_scheduler.StepLR(d_optim, step_size=cfg["step_size"],
                                                    gamma=cfg["gamma"]) if disc is not None else None

    if os.environ.get("VQ_COMPILE") == "1":
        # optional torch.compile; checkpoints then carry _orig_mod.* key
        # prefixes (sample_top10.py strips them on load; keep the flag
        # consistent across resume runs)
        gen = torch.compile(gen)
        if disc is not None:
            disc = torch.compile(disc)

    st_step = 1
    if args.resume:
        st_step, _ = load_checkpoint(args.resume, gen, disc, g_optim, d_optim,
                                     gen_scheduler, dis_scheduler)
        logger.info(f"Resumed from {args.resume} at step {st_step}")

    writer_path = cfg.work_dir / "runs" / cfg.unique_name
    eval_image_path = cfg.work_dir / "images" / cfg.unique_name
    writer = utils.TBDiskWriter(writer_path, eval_image_path, scale=0.6)

    envaluator = Evaluator(env, env_get, cfg, logger, writer, cfg.batch_size,
                           val_transform, cfg.content_font, use_half=cfg.use_half)
    trainer = CombinedTrainer(gen, disc, g_optim, d_optim, gen_scheduler, dis_scheduler,
                              logger, envaluator, cv_loaders, cfg)

    chars_sim_dict = SimMatrix(cfg.sim_npy, cfg.sim_unis)
    logger.info(f"SimMatrix {chars_sim_dict.sim.shape} loaded")

    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    trainer.train(trn_loader, st_step, cfg["iter"], component_objects, chars_sim_dict)


if __name__ == "__main__":
    main()
