# -*- coding: utf-8 -*-
"""
train_pixel.py — Pixel-space DiT training (no VAE).

直接对 256x256 RGB 像素做扩散训练, 结构损失 (EdgeGradientLoss/SkeletonLoss)
在模型预测的像素 x0 上无损计算, 无需 VAE decode 桥 —— 无跨流形梯度、
无 decode 显存开销, batch 可开大 (4090 @ 24G: patch8/1024token, batch 32+)。

与 src/train.py (latent 路线) 的差异:
  - x = batch['image'] ([-1,1] 256x256x3) 直接进 diffusion
  - pred_xstart 即像素 x0_pred, 结构损失直接作用其上 (无 vae.decode)
  - 模型 DiT_2Cond_S_8: input_size=256, patch_size=8, in_channels=3

用法 (tmux 内, 脱离 ssh):
  /opt/conda/bin/python src/train_pixel.py --config exp_px_XXX.json
"""
import os
os.environ["XFORMERS_DISABLED"] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import numpy as np
import sys
from glob import glob
from time import time
import argparse
import logging
import json
import datetime
import copy
import re
import math
import random
from collections import OrderedDict

from models import DiT_2Cond_models, DiT_2Cond
from diffusion import create_diffusion
from losses import EdgeGradientLoss, SkeletonLoss
from samplers import DistributedFactorBalancedSampler

# ----------------------------------------------------------------------------
# 日志
# ----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format='[\033[34m%(asctime)s\033[0m] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


# ----------------------------------------------------------------------------
# Pixel 数据集: csv + 图片 (可选 canny/skel map), 无 latent shards
# ----------------------------------------------------------------------------
class PixelDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_root, canny_root=None, skel_root=None,
                 image_size=256, load_canny=False, load_skel=False,
                 preload=False, preload_dir=None, num_preload_workers=16):
        import csv as _csv
        self.samples = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            for row in _csv.DictReader(f):
                self.samples.append(row)
        self.img_root = img_root
        self.canny_root = canny_root
        self.skel_root = skel_root
        self.image_size = image_size
        self.load_canny = load_canny
        self.load_skel = load_skel
        self.preload = preload
        self.preload_dir = preload_dir
        self._imgs = None
        self._cannys = None
        self._skels = None
        if preload:
            self._preload_all(num_preload_workers)

    def _preload_all(self, workers=16):
        # ---- mmap 大文件打包 (5script/pixmap/*.npy, 顺序建好, 随机访问零解码) ----
        t0 = time()
        if self.preload_dir and os.path.isdir(self.preload_dir):
            _imgs_found = False
            imgp = os.path.join(self.preload_dir, "imgs.npy")
            if os.path.exists(imgp):
                self._imgs = np.load(imgp, mmap_mode="c")
                _imgs_found = True
                logger.info(f"[Preload] mmap imgs {imgp} {self._imgs.shape} "
                            f"({os.path.getsize(imgp)/1024**3:.1f}G) in {time()-t0:.1f}s")
            if self.load_canny:
                cp = os.path.join(self.preload_dir, "cannys.npy")
                if os.path.exists(cp):
                    self._cannys = np.load(cp, mmap_mode="c")
            if self.load_skel:
                sp = os.path.join(self.preload_dir, "skels.npy")
                if os.path.exists(sp):
                    self._skels = np.load(sp, mmap_mode="c")
            if _imgs_found:
                return
            logger.warning("[Preload] pixmap dir missing imgs.npy, falling back to PNG decode")
        import multiprocessing as mp
        from concurrent.futures import ThreadPoolExecutor
        from PIL import Image
        n = len(self.samples)
        self._imgs = np.empty((n, 256, 256, 3), dtype=np.uint8)
        self._cannys = np.empty((n, 256, 256), dtype=np.uint8) if self.load_canny else None
        self._skels = np.empty((n, 256, 256), dtype=np.uint8) if self.load_skel else None
        ids = [int(re.search(r"(\d+)\.png", r["image_path"]).group(1)) for r in self.samples]

        def _load(i):
            img_id = ids[i]
            with Image.open(os.path.join(self.img_root, f"{img_id}.png")) as im:
                self._imgs[i] = np.asarray(im.convert('RGB'))
            if self.load_canny:
                with Image.open(os.path.join(self.canny_root, f"{img_id}.png")) as c:
                    self._cannys[i] = np.asarray(c.convert('L'))
            if self.load_skel:
                with Image.open(os.path.join(self.skel_root, f"{img_id}.png")) as sk:
                    self._skels[i] = np.asarray(sk.convert('L'))

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_load, range(n)))
        ram = (self._imgs.nbytes
               + (self._cannys.nbytes if self._cannys is not None else 0)
               + (self._skels.nbytes if self._skels is not None else 0))
        logger.info(f"[Preload] {n} images (+can/sk) loaded, RAM {ram/1024**3:.1f}G")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        from PIL import Image
        m = re.search(r"(\d+)\.png", row['image_path'])
        if not m:
            raise ValueError(f"Cannot parse img_id from {row['image_path']}")
        img_id = int(m.group(1))

        if self.preload and self._imgs is not None:
            img_t = torch.from_numpy(self._imgs[idx].astype(np.float32) / 255.0 * 2.0 - 1.0).permute(2, 0, 1)
        else:
            with Image.open(os.path.join(self.img_root, f"{img_id}.png")) as im:
                img = im.convert('RGB')
            img_t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1) * 2.0 - 1.0

        canny_t = torch.empty(0)
        if self.load_canny:
            if self.preload:
                c = self._cannys[idx].astype(np.float32) / 255.0
            else:
                with Image.open(os.path.join(self.canny_root, f"{img_id}.png")) as cim:
                    c = np.asarray(cim.convert('L'), dtype=np.float32) / 255.0
            canny_t = (torch.from_numpy(c) > 0.5).float().unsqueeze(0)

        skel_t = torch.empty(0)
        if self.load_skel:
            if self.preload:
                s = self._skels[idx].astype(np.float32) / 255.0
            else:
                with Image.open(os.path.join(self.skel_root, f"{img_id}.png")) as sim:
                    s = np.asarray(sim.convert('L'), dtype=np.float32) / 255.0
            skel_t = (torch.from_numpy(s) > 0.5).float().unsqueeze(0)

        return {
            'image': img_t,
            'canny': canny_t,
            'skeleton': skel_t,
            'y_callig': torch.tensor(int(row['calligrapher_id']), dtype=torch.long),
            'y_char': torch.tensor(int(row.get('glyph_id', row['character_id'])), dtype=torch.long),
        }


# ----------------------------------------------------------------------------
# 解析
# ----------------------------------------------------------------------------
def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("true", "1", "yes", "on")


parser = argparse.ArgumentParser()
parser.add_argument("--config", default=None, help="JSON config path (optional).")
parser.add_argument("--data-csv", default="5script/train_top6.csv")
parser.add_argument("--img-root", default="final_imgs_256")
parser.add_argument("--canny-root", default="final_canny")
parser.add_argument("--skel-root", default="final_skeleton")
parser.add_argument("--results-dir", default="5script/results/px")
parser.add_argument("--experiment-name", default="pixel-diT-S-8")
parser.add_argument("--image-size", type=int, default=256)
parser.add_argument("--patch-size", type=int, default=8)
parser.add_argument("--in-channels", type=int, default=3)
parser.add_argument("--epochs", type=int, default=100000)
parser.add_argument("--max-steps", type=int, default=200000)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--lr-schedule", default="cosine")
parser.add_argument("--warmup-steps", type=int, default=1000)
parser.add_argument("--min-lr-ratio", type=float, default=0.2)
parser.add_argument("--weight-decay", type=float, default=0.02)
parser.add_argument("--global-batch-size", type=int, default=32)
parser.add_argument("--global-seed", type=int, default=0)
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--use-ema", type=_str_to_bool, default=True)
parser.add_argument("--ema-decay", type=float, default=0.9999)
parser.add_argument("--log-every", type=int, default=20)
parser.add_argument("--ckpt-every", type=int, default=5000)
parser.add_argument("--ckpt-keep", type=int, default=40)
parser.add_argument("--preload", type=_str_to_bool, default=True)
parser.add_argument("--preload-workers", type=int, default=24)
parser.add_argument("--preload-dir", default=None,
                    help="打包好的 pixmap 目录 (imgs.npy/cannys.npy/skels.npy, mmap 随机访问)")
# 结构损失
parser.add_argument("--use-canny", type=_str_to_bool, default=False)
parser.add_argument("--use-skel", type=_str_to_bool, default=False)
parser.add_argument("--w-canny", type=float, default=0.5)
parser.add_argument("--w-skel", type=float, default=1.0)
parser.add_argument("--struct-subset", type=int, default=0,
                    help="结构损失计算的子集大小 (0=全 batch；像素域无 decode,显存富余,常设 0)")
parser.add_argument("--struct-max-t", type=int, default=0,
                    help="只在 t<=struct_max_t 的低噪声步施加结构损失 (0=全部)")
parser.add_argument("--struct-warmup-steps", type=int, default=0,
                    help="结构损失权重从 0 线性渐入的步数 (0=直接全额)")
# 条件
parser.add_argument("--cond-mode", default="2cond")
parser.add_argument("--condition-fusion", default="factorized_add")
parser.add_argument("--callig-embed-dim", type=int, default=128)
parser.add_argument("--char-embed-dim", type=int, default=256)
parser.add_argument("--cond-drop-all", type=float, default=0.05)
parser.add_argument("--cond-drop-one", type=float, default=0.25)
parser.add_argument("--num-calligraphers", type=int, default=1011)
parser.add_argument("--num-characters", type=int, default=35130)
# 评估 / 早停
parser.add_argument("--eval-csv", default="5script/eval100_top6.csv")
parser.add_argument("--eval-n", type=int, default=100)
parser.add_argument("--eval-steps", type=int, default=50)
parser.add_argument("--eval-cfg", type=float, default=4.0)
parser.add_argument("--eval-seed", type=int, default=0)
parser.add_argument("--eval-batch", type=int, default=8)
parser.add_argument("--early-stop", type=_str_to_bool, default=True)
parser.add_argument("--early-stop-metric", default="ssim", choices=["ssim", "mse"])
parser.add_argument("--early-stop-patience", type=int, default=6)
parser.add_argument("--early-stop-min-steps", type=int, default=30000)
# resume
parser.add_argument("--resume-full", default=None,
                    help="Pixel 版 full checkpoint (delta/opt/ema/scheduler/train_steps) 无损续训")


def parse_args():
    args = parser.parse_args()
    if args.config:
        cfg = json.load(open(args.config, encoding="utf-8"))
        # config json 覆盖 argparse 默认 (不存在的键忽略)
        known = {a.dest: a for a in parser._actions if a.dest != "help"}
        for k, v in cfg.items():
            if k in known:
                setattr(args, k, v)
        # 将未知 config 键挂到 args, 供 getattr 使用
        for k, v in cfg.items():
            if not hasattr(args, k):
                setattr(args, k, v)
    return args


# ----------------------------------------------------------------------------
# EMA / checkpoint
# ----------------------------------------------------------------------------
def _ema_update(ema, model, decay):
    with torch.no_grad():
        for ep, p in zip(ema.parameters(), model.parameters()):
            ep.mul_(decay).add_(p.detach(), alpha=1.0 - decay)


def main():
    # mmap 数组 + fork = 页表 COW 共享, 零复制且无 spawn 开销; 见 gaussian 训练主流程同款
    # (不用 spawn: spawn 会重新 import 且 mmap 句柄在子进程重建, 8 workers 时极慢)
    try:
        torch.multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass
    args = parse_args()
    print("=== train_pixel args ===")
    for k, v in vars(args).items():
        if v not in (None, False, ""):
            print(f"  {k}: {v}")

    torch.manual_seed(args.global_seed + np.random.randint(0, 10000))
    device = torch.device("cuda")
    torch.cuda.set_device(0)

    # ---- 数据集 ----
    preload_dir = getattr(args, "preload_dir", None) or None
    ds = PixelDataset(
        csv_file=args.data_csv, img_root=args.img_root,
        canny_root=args.canny_root if args.use_canny else None,
        skel_root=args.skel_root if args.use_skel else None,
        image_size=args.image_size, load_canny=args.use_canny,
        load_skel=args.use_skel, preload=args.preload,
        preload_dir=preload_dir,
        num_preload_workers=args.preload_workers)
    logger.info(f"Dataset {len(ds)} samples")

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=args.global_batch_size,
                        shuffle=True, num_workers=args.num_workers, drop_last=True)

    # ---- 模型 (S, patch8) ----
    model = DiT_2Cond(
        input_size=args.image_size, patch_size=args.patch_size,
        in_channels=args.in_channels,
        hidden_size=384, depth=12, num_heads=6,
        num_calligraphers=args.num_calligraphers,
        num_characters=args.num_characters,
        condition_fusion=args.condition_fusion,
        callig_embed_dim=args.callig_embed_dim,
        char_embed_dim=args.char_embed_dim,
        cond_drop_all_prob=args.cond_drop_all,
        cond_drop_one_prob=args.cond_drop_one,
        use_checkpoint=False,
        learn_sigma=True,
    ).to(device)
    logger.info(f"DiT-S/8: params={sum(p.numel() for p in model.parameters()):,}")

    n_tokens = (args.image_size // args.patch_size) ** 2
    logger.info(f"pixel tokens per image: {n_tokens} "
                f"(attention {n_tokens**2/1e6:.1f}M elems/image)")

    ema_model = None
    if args.use_ema:
        ema_model = copy.deepcopy(model).eval()
        requires_grad(ema_model, False)
        logger.info(f"[EMA] enabled decay={args.ema_decay}")

    # ---- 恢复 (resume-full) ----
    resume_start_step = 0
    if args.resume_full:
        ck = torch.load(args.resume_full, map_location="cpu", weights_only=False)
        raw = model
        sd = ck.get("delta", ck.get("model", None))
        missing, unexpected = raw.load_state_dict(sd, strict=False)
        logger.info(f"[resume-full] weights loaded (missing={len(missing)}, unexpected={len(unexpected)})")
        resume_start_step = int(ck.get("train_steps", 0))
        if ema_model is not None and ck.get("ema") is not None:
            ema_model.load_state_dict(ck["ema"], strict=True)
            logger.info("[EMA] restored")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.lr_schedule == "cosine":
        from torch.optim.lr_scheduler import LambdaLR
        def _lr_fn(step):
            if step < args.warmup_steps:
                return float(step + 1) / float(max(args.warmup_steps, 1))
            total = max(args.max_steps, 1)
            progress = (step - args.warmup_steps) / max(total - args.warmup_steps, 1)
            return max(args.min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
        scheduler = LambdaLR(optimizer, lr_lambda=_lr_fn)

    diffusion = create_diffusion(timestep_respacing="")
    canny_loss_fn = EdgeGradientLoss().to(device)
    skel_loss_fn = SkeletonLoss().to(device)

    # ---- 结果目录 ----
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir = os.path.join(args.results_dir, f"{ts}-{args.experiment_name}")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger.info(f"results: {exp_dir}")
    log_path = os.path.join(exp_dir, "log.txt")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter('[\033[34m%(asctime)s\033[0m] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(fh)
    # 写 active ckpt dir 供 auto_eval_cpu 定位
    with open(os.path.join(args.results_dir, "_active_ckpt_dir.txt"), "w") as f:
        f.write(ckpt_dir + "\n")

    # ---- 训练 ----
    train_steps = resume_start_step
    early_stop_best = None
    early_stop_stale = 0
    early_stop_last_eval_step = -1

    def _check_early_stop():
        nonlocal early_stop_best, early_stop_stale, early_stop_last_eval_step
        if not args.early_stop:
            return False
        if train_steps < int(getattr(args, "early_stop_min_steps", 0)):
            return False
        ev_files = sorted(glob(os.path.join(ckpt_dir, "eval_auto_*.json")))
        if not ev_files:
            return False
        last_ev = ev_files[-1]
        step = int(os.path.basename(last_ev).replace("eval_auto_", "").replace(".json", ""))
        if step <= early_stop_last_eval_step:
            return False
        early_stop_last_eval_step = step
        d = json.load(open(last_ev, encoding="utf-8"))
        val = float(d.get(args.early_stop_metric))
        if val is None:
            return False
        better = (val > early_stop_best) if args.early_stop_metric == "ssim" else (val < early_stop_best)
        if early_stop_best is None or better:
            early_stop_best = val
            early_stop_stale = 0
            logger.info(f"[early-stop] eval step {step}: {args.early_stop_metric}={val:.4f} (new best)")
        else:
            early_stop_stale += 1
            logger.info(f"[early-stop] eval step {step}: {args.early_stop_metric}={val:.4f} "
                        f"(best {early_stop_best:.4f}, stale {early_stop_stale}/{args.early_stop_patience})")
            if early_stop_stale >= args.early_stop_patience:
                logger.info("[early-stop] no improvement; early stop.")
                return True
        return False

    # 结构损失权重渐入
    def _struct_scale():
        if args.struct_warmup_steps <= 0:
            return 1.0
        ft = max(0, train_steps - resume_start_step)
        return min(1.0, ft / float(args.struct_warmup_steps))

    optimizer.zero_grad()
    early_stopped = False
    t_start = time()

    for epoch in range(args.epochs):
        if early_stopped:
            break
        for batch in loader:
            x = batch['image'].to(device)
            y_callig = batch['y_callig'].to(device)
            y_char = batch['y_char'].to(device)
            canny_gt = batch['canny'].to(device) if args.use_canny else None
            skel_gt = batch['skeleton'].to(device) if args.use_skel else None
            model_kwargs = dict(y_callig=y_callig, y_char=y_char)

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss_diff = loss_dict["loss"].mean()

            pred_x0 = loss_dict.get("pred_xstart", None)  # 像素域 x0 预测

            # ---- 结构损失: 直接 on 像素 x0, 无 VAE decode ----
            loss_canny = torch.tensor(0.0, device=device)
            loss_skel = torch.tensor(0.0, device=device)
            if (args.use_canny or args.use_skel) and pred_x0 is not None:
                B_ = pred_x0.shape[0]
                _ss = int(getattr(args, "struct_subset", 0))
                _use = True
                _idx = None
                if int(getattr(args, "struct_max_t", 0)) > 0:
                    _gi = torch.nonzero(t <= int(getattr(args, "struct_max_t", 0))).view(-1)
                    if _gi.numel() == 0:
                        _use = False
                    elif _ss > 0 and _gi.numel() > _ss:
                        _idx = _gi[torch.randperm(_gi.numel(), device=device)[:_ss]]
                    else:
                        _idx = _gi
                elif _ss > 0 and _ss < B_:
                    _idx = torch.randperm(B_, device=device)[:_ss]
                if _use:
                    sub = pred_x0 if _idx is None else pred_x0[_idx]
                    if args.use_canny:
                        gt_c = canny_gt if _idx is None else canny_gt[_idx]
                        # EdgeGradientLoss 需要 GT 图: 这里退化为像素直接监督梯度
                        # 对 pixel diffusion, canny 用预测图梯度 vs GT 图梯度更合理,
                        # 这里直接对预测的 x0 与 GT 图算 Edge 损失
                        loss_canny = canny_loss_fn(sub, x[_idx] if _idx is not None else x)
                    if args.use_skel:
                        gt_s = skel_gt if _idx is None else skel_gt[_idx]
                        loss_skel = skel_loss_fn(sub, gt_s)

            scl = _struct_scale()
            loss = (loss_diff
                    + args.w_canny * scl * loss_canny
                    + args.w_skel * scl * loss_skel)

            if not torch.isfinite(loss):
                logger.warning(f"[nan] step {train_steps}: loss={loss.item()}, skip")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if ema_model is not None:
                _ema_update(ema_model, model, args.ema_decay)

            train_steps += 1

            if train_steps % args.log_every == 0:
                dt = time() - t_start
                sps = args.log_every / max(dt, 1e-6)
                t_start = time()
                lr_now = optimizer.param_groups[0]['lr']
                logger.info(
                    f"(step={train_steps:07d}) Total: {loss.item():.4f} | "
                    f"Diff: {loss_diff.item():.4f} | "
                    f"Canny: raw {loss_canny.item():.4f} x {args.w_canny*scl:.2f} | "
                    f"Skel: raw {loss_skel.item():.4f} x {args.w_skel*scl:.2f} | "
                    f"LR: {lr_now:.2e} | Steps/Sec: {sps:.2f} | Mem: {torch.cuda.memory_reserved()/1024**3:.2f}G")

            # ---- 周期保存 (训练自己不 eval; 评估由独立 CPU eval 进程完成) ----
            if (args.ckpt_every > 0 and train_steps % args.ckpt_every == 0):
                _save_ckpt(model, ema_model, optimizer, scheduler, train_steps, ckpt_dir, args)

            # ---- 早停 ----
            _es_every = 5000
            if (train_steps >= int(getattr(args, "early_stop_min_steps", 0))
                    and train_steps % _es_every == 0 and _check_early_stop()):
                early_stopped = True
                logger.info(f"[early-stop] triggered at step {train_steps}")
                break

        if train_steps >= args.max_steps:
            logger.info(f"Reached max_steps={args.max_steps}; stopping.")
            break

    logger.info("Done!")
    with open(os.path.join(args.results_dir, "_active_ckpt_dir.txt"), "w") as f:
        f.write("")


def _save_ckpt(model, ema_model, optimizer, scheduler, step, ckpt_dir, args):
    raw = model if not hasattr(model, "module") else model.module
    ck = {
        "delta": {k: v.detach().cpu() for k, v in raw.state_dict().items()},
        "opt": optimizer.state_dict(),
        "ema": ema_model.state_dict() if ema_model is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "train_steps": step,
        "args": args,
        "pixel_mode": True,
        "saved_at": datetime.datetime.now().isoformat(),
    }
    path = os.path.join(ckpt_dir, f"{step:07d}.pt")
    torch.save(ck, path)
    logger.info(f"Saved checkpoint to {path}")
    # 清理旧 ckpt
    if args.ckpt_keep > 0:
        pts = sorted(glob(os.path.join(ckpt_dir, "*.pt")))
        for p in pts[:-args.ckpt_keep]:
            os.remove(p)


if __name__ == "__main__":
    main()