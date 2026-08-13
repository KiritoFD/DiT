import os
os.environ["XFORMERS_DISABLED"] = "1"
import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import sys
from glob import glob
from time import time
import argparse
import logging
import json

from models import DiT_2Cond_models, DiT_3Cond_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model

from dataset import MCCDDataset
from losses import SobelCannyLoss, SkeletonLoss, REPALoss
from torch.utils.checkpoint import checkpoint as grad_ckpt

def _coerce(value, template):
    """Coerce a config.json value to the type of the argparse default."""
    if isinstance(template, bool):
        return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes", "on")
    if isinstance(template, int):
        return int(value)
    if isinstance(template, float):
        return float(value)
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in ("none", "null", ""):
        return None
    return str(value)


def _str_to_bool(value):
    """Single-arg bool parser for argparse type=."""
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "on")


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    dist.destroy_process_group()

def create_logger(logging_dir):
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

    # gloo is the only backend available on Windows; prefer nccl on Linux for speed.
    backend = "nccl" if (dist.is_nccl_available() and sys.platform != "win32") else "gloo"
    dist.init_process_group(backend)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(device)
    
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    assert args.image_size % 8 == 0
    latent_size = args.image_size // 8
    
    cond_mode = args.cond_mode
    if cond_mode == "3cond":
        if args.model not in DiT_3Cond_models:
            raise ValueError(f"cond_mode=3cond but model '{args.model}' is not a 3Cond model. "
                             f"Use one of {list(DiT_3Cond_models.keys())}.")
        model = DiT_3Cond_models[args.model](
            input_size=latent_size,
            num_calligraphers=args.num_calligraphers,
            num_scripts=args.num_scripts,
            num_characters=args.num_characters,
            use_checkpoint=args.use_checkpoint
        )
        logger.info(f"Building 3-Cond model: {args.model} "
                    f"(callig={args.num_calligraphers}, script={args.num_scripts}, char={args.num_characters})")
    else:
        if args.model not in DiT_2Cond_models:
            raise ValueError(f"cond_mode=2cond but model '{args.model}' is not a 2Cond model. "
                             f"Use one of {list(DiT_2Cond_models.keys())}.")
        model = DiT_2Cond_models[args.model](
            input_size=latent_size,
            num_calligraphers=args.num_calligraphers,
            num_characters=args.num_characters,
            use_checkpoint=args.use_checkpoint
        )
        logger.info(f"Building 2-Cond model: {args.model} "
                    f"(callig={args.num_calligraphers}, char={args.num_characters})")

    # Load order (fixed): pretrained body -> reset cond head -> inject LoRA -> load delta.
    # The checkpoint `delta` contains only the "changed" part (LoRA + condition head +
    # adaLN/final_layer), so the frozen pretrained body is ALWAYS loaded from disk first.
    _resume_full_ckpt = None

    # 1) pretrained body (shared, not stored per-ckpt)
    if args.pretrained is not None:
        ckpt_path = args.pretrained
        state_dict = find_model(ckpt_path)
        # Filter out ALL label-embedding / conditioning keys that don't match DiT_2Cond.
        # The pretrained DiT-XL checkpoint has a single 'y_embedder'; DiT_2Cond/3Cond have
        # separate calligrapher/character(/script) embedders plus cond_fusion. Keep only the
        # transformer body (x_embedder / pos_embed / t_embedder / blocks / final_layer).
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.startswith(("y_embedder", "y_callig", "y_script", "y_char", "cond_fusion"))}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded pre-trained weights from {ckpt_path}.")
        logger.info(f"Missing keys (expected for 2/3-cond): {missing}")
        logger.info(f"Unexpected keys (filtered): {unexpected}")

    # 2) re-init the conditioning head (adaLN & final_layer) after loading pretrained body.
    # The pretrained adaLN was trained on ImageNet's single y_embedder; our 3-cond fused
    # condition vector c is out-of-distribution for it, which can blow up to NaN early on.
    # Re-initializing adaLN/final_layer to a small std (like the successful overfit run)
    # keeps the pretrained transformer body while letting the new condition head learn.
    # (Skipped on full resume: the delta already carries the learned adaLN.)
    if getattr(args, 'reset_cond_head', True) and getattr(args, 'resume_full', None) is None:
        import torch.nn as _nn
        for _b in model.blocks:
            _nn.init.normal_(_b.adaLN_modulation[-1].weight, std=0.02)
            _nn.init.normal_(_b.adaLN_modulation[-1].bias, std=0.02)
        _nn.init.normal_(model.final_layer.adaLN_modulation[-1].weight, std=0.02)
        _nn.init.normal_(model.final_layer.adaLN_modulation[-1].bias, std=0.02)
        _nn.init.normal_(model.final_layer.linear.weight, std=0.02)
        logger.info("Re-initialized adaLN/final_layer (std=0.02) to fit new 3-cond condition head.")

    from lora import inject_lora, upgrade_lora_rank, extract_full_inference
    if getattr(args, 'use_lora', True):
        _r = getattr(args, 'lora_r', 16)
        _alpha = getattr(args, 'lora_alpha', None)
        if _alpha is None:
            _alpha = _r  # default scaling = 1
        logger.info(f"Injecting LoRA (r={_r}, alpha={_alpha}, scaling={_alpha/_r:.2f}) into DiT blocks...")
        model = inject_lora(model, r=_r, lora_alpha=_alpha)

        # 3) full resume: load the delta (LoRA + condition head + adaLN/final_layer)
        if getattr(args, 'resume_full', None) is not None:
            import torch as _torch
            _rf = _torch.load(args.resume_full, map_location="cpu", weights_only=False)
            _resume_full_ckpt = _rf
            _sd = _rf.get("delta", _rf.get("model", _rf))
            missing, unexpected = model.load_state_dict(_sd, strict=False)
            logger.info(f"[resume-full] Loaded delta from {args.resume_full} "
                        f"(missing={len(missing)}, unexpected={len(unexpected)}).")
        
        # Freeze everything first
        requires_grad(model, False)

        # LoRA + newly-added condition embedders/fusion are always trainable.
        # adaLN / final_layer: `reset_cond_head` re-initializes them (std=0.02) to
        #   fit the new 3-cond head. If they stay frozen afterwards, the model is left
        #   with a *random, never-trained* modulation — the root cause of blurry output
        #   and per-ckpt adaLN divergence. `train_cond_head` (default True) makes them
        #   trainable so they can actually learn, matching the original design intent.
        train_cond_head = getattr(args, 'train_cond_head', True)
        for name, param in model.named_parameters():
            if ('lora_' in name or 'y_callig_embedder' in name or 'y_char_embedder' in name
                    or 'cond_fusion' in name or 'y_script_embedder' in name):
                param.requires_grad = True
            elif train_cond_head and ('adaLN' in name or 'final_layer' in name):
                param.requires_grad = True

        # Calculate trainable params (should be far smaller than full model now)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        logger.info(f"LoRA Trainable Parameters: {trainable_params:,}")
        logger.info(f"Frozen Parameters: {frozen_params:,} (trainable ratio: {trainable_params/(trainable_params+frozen_params)*100:.2f}%)")

        # Optional: upgrade LoRA rank from a previous run's checkpoint, preserving
        # the learned low-rank deltas. See lora.upgrade_lora_rank for the strategy.
        if getattr(args, 'resume_lora', None):
            import torch as _torch
            _sd = _torch.load(args.resume_lora, map_location="cpu")
            _sd = _sd.get("state_dict", _sd) if isinstance(_sd, dict) else _sd
            old_r = getattr(args, 'old_lora_r', 16)
            model = upgrade_lora_rank(model, _r, _alpha, _sd, old_r)
            # recompute trainable counts after rebuild
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
            logger.info(f"[resume_lora] Upgraded from {args.resume_lora} (old_r={old_r}). "
                        f"Trainable now: {trainable_params:,} ({trainable_params/(trainable_params+frozen_params)*100:.2f}%)")

    model = model.to(device)
    if dist.get_world_size() > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    diffusion = create_diffusion(timestep_respacing="")
    class MockVAE(torch.nn.Module):
        def __init__(self, device):
            super().__init__()
            self.device = device
        def encode(self, x):
            class Dist:
                def sample(self):
                    return torch.randn(x.shape[0], 4, x.shape[2]//8, x.shape[3]//8, device=x.device)
            class Output:
                latent_dist = Dist()
            return Output()
        def decode(self, z):
            class Output:
                sample = torch.randn(z.shape[0], 3, z.shape[2]*8, z.shape[3]*8, device=z.device)
            return Output()

    try:
        if getattr(args, 'vae_path', None) is not None and os.path.exists(args.vae_path):
            logger.info(f"Loading VAE from local path: {args.vae_path}")
            vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
        else:
            vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
        requires_grad(vae, False)
    except Exception as e:
        logger.warning(f"Failed to load AutoencoderKL due to network/path error: {e}")
        logger.warning("Using MockVAE (random latents) for testing purposes!")
        vae = MockVAE(device)

    
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    raw_model = model.module if hasattr(model, 'module') else model
    student_hidden_size = raw_model.x_embedder.proj.weight.shape[0]
    
    # Initialize structural and REPA losses
    canny_loss_fn = SobelCannyLoss().to(device)
    skel_loss_fn = SkeletonLoss(lambda_bg=1.0).to(device)
    
    repa_loss_fn = None
    if args.w_repa > 0:
        teacher_ckpt = getattr(args, "repa_teacher_ckpt", "") or None
        logger.info(f"Initializing REPA Loss (Teacher: dinov2_vits14, ckpt={teacher_ckpt or 'auto'}, Student Dim: {student_hidden_size})")
        repa_loss_fn = REPALoss(student_dim=student_hidden_size, teacher_backbone="dinov2_vits14",
                                teacher_ckpt=teacher_ckpt).to(device)

    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    if repa_loss_fn is not None:
        trainable_params_list.extend([p for p in repa_loss_fn.proj.parameters() if p.requires_grad])
        
    opt = torch.optim.AdamW(trainable_params_list, lr=args.lr, weight_decay=0)

    # Restore optimizer state + step counter for full resume. If --resume-lr is given,
    # override the LR so we can test whether a smaller LR avoids the NaN.
    resume_start_step = 0
    if _resume_full_ckpt is not None:
        _opt_sd = _resume_full_ckpt.get("opt", None)
        if _opt_sd is not None:
            try:
                opt.load_state_dict(_opt_sd)
                logger.info(f"[resume-full] Restored optimizer state.")
            except Exception as _e:
                logger.warning(f"[resume-full] Failed to restore optimizer state: {_e}")
        if getattr(args, 'resume_lr', None) is not None:
            for _pg in opt.param_groups:
                _pg["lr"] = args.resume_lr
            logger.info(f"[resume-full] Overrode LR -> {args.resume_lr}")
        # Recover the step counter. `args` (saved Namespace) has no train_steps field,
        # so prefer the checkpoint filename (e.g. 0010000.pt -> 10000); fall back to a
        # stored args.train_steps if present.
        import re as _re
        _fname = os.path.basename(str(args.resume_full))
        _digits = _re.findall(r"\d+", _fname)
        if _digits:
            resume_start_step = int(_digits[-1])
            logger.info(f"[resume-full] Inferred start step={resume_start_step} from filename {_fname}")
        _ckpt_args = _resume_full_ckpt.get("args", None)
        if _ckpt_args is not None and getattr(_ckpt_args, "train_steps", None) is not None:
            resume_start_step = int(_ckpt_args.train_steps)
            logger.info(f"[resume-full] Resuming from train_steps={resume_start_step}")

    # bf16 training: run the model in bf16 autocast (no loss scaling needed — bf16 has
    # the same exponent range as fp32, so it does not overflow like fp16 AMP). VAE and
    # structural losses stay fp32 for numerical stability.
    dataset = MCCDDataset(csv_file=args.data_csv, root_dir=args.data_dir, image_size=args.image_size,
                          load_canny=args.use_canny, load_skel=args.use_skel)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images")

    model.train()

    train_steps = resume_start_step
    log_steps = 0
    running_loss = 0
    running_diff = 0
    running_canny = 0
    running_skel = 0
    running_repa = 0
    nan_steps = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        if rank == 0:
            logger.info(f"Beginning epoch {epoch}...")
        
        try:
            for batch_idx, batch in enumerate(loader):
                x = batch['image'].to(device)
                canny_gt = batch['canny'].to(device)
                skel_gt = batch['skeleton'].to(device)
                y_callig = batch['y_callig'].to(device)
                y_char = batch['y_char'].to(device)
                if cond_mode == "3cond":
                    y_script = batch['y_script'].to(device)

                # VAE encode stays in fp32 for numerical stability (VAE is sensitive to low precision).
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float32):
                    x_latent = vae.encode(x).latent_dist.sample().mul_(0.18215)
                    x_latent = x_latent.float()

                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                if cond_mode == "3cond":
                    model_kwargs = dict(y_callig=y_callig, y_script=y_script, y_char=y_char)
                else:
                    model_kwargs = dict(y_callig=y_callig, y_char=y_char)
                
                # If REPA is enabled, request intermediate layer 8 features
                if args.w_repa > 0:
                    model_kwargs['return_intermediate_layer'] = 8

                # Forward pass under bf16 autocast (same exponent range as fp32, no overflow).
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss_dict = diffusion.training_losses(model, x_latent, t, model_kwargs)
                    loss_diff = loss_dict["loss"].mean()

                loss_canny = torch.tensor(0.0, device=device)
                loss_skel = torch.tensor(0.0, device=device)
                loss_repa = torch.tensor(0.0, device=device)

                pred_xstart_latent = loss_dict.get("pred_xstart", None)
                if pred_xstart_latent is not None and (args.use_canny or args.use_skel):
                    # VAE decode in fp32 via gradient checkpointing to save memory:
                    # the VAE is frozen, so on backward we re-run decode instead of
                    # keeping its large up-sampling activations resident.
                    def _decode(z):
                        return vae.decode(z).sample
                    with torch.autocast("cuda", dtype=torch.float32):
                        x0_pred = grad_ckpt(_decode, pred_xstart_latent.float() / 0.18215,
                                             use_reentrant=False)
                    # Structural losses computed in fp32 (outside autocast) for stability.
                    if args.use_canny:
                        loss_canny = canny_loss_fn(x0_pred, canny_gt)
                    if args.use_skel:
                        loss_skel = skel_loss_fn(x0_pred, skel_gt)

                intermediate_feats = loss_dict.get("intermediate_feats", None)
                if intermediate_feats is not None and repa_loss_fn is not None and args.w_repa > 0:
                    # original 'x' is ground truth x_0 [-1, 1]
                    loss_repa = repa_loss_fn(intermediate_feats, x)

                loss = loss_diff + args.w_canny * loss_canny + args.w_skel * loss_skel + args.w_repa * loss_repa

                opt.zero_grad()
                # NaN guard: skip the step if loss is not finite (e.g. a bad sample).
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params_list, max_norm=1.0)
                    opt.step()
                else:
                    nan_steps += 1
                    if rank == 0:
                        logger.warning(
                            f"Step {train_steps} skipped: non-finite loss "
                            f"(diff={loss_diff.item():.4f}, canny={loss_canny.item():.4f}, "
                            f"skel={loss_skel.item():.4f}). Accumulated skips: {nan_steps}"
                        )
                    # No optimizer update on NaN; just skip (grads are discarded).

                if torch.isfinite(loss):
                    running_loss += loss.item()
                    running_diff += loss_diff.item()
                    running_canny += loss_canny.item()
                    running_skel += loss_skel.item()
                    running_repa += loss_repa.item()
                    log_steps += 1
                train_steps += 1
                
                if train_steps % args.log_every == 0:
                    torch.cuda.synchronize()
                    end_time = time()
                    # Guard against a logging window in which every step was skipped due to non-finite loss.
                    divisor = max(log_steps, 1)
                    steps_per_sec = log_steps / max(end_time - start_time, 1e-9)
                    
                    avg_l = torch.tensor(running_loss / divisor, device=device)
                    avg_d = torch.tensor(running_diff / divisor, device=device)
                    avg_c = torch.tensor(running_canny / divisor, device=device)
                    avg_s = torch.tensor(running_skel / divisor, device=device)
                    avg_r = torch.tensor(running_repa / divisor, device=device)
                    ws = dist.get_world_size()
                    if ws > 1:
                        dist.all_reduce(avg_l, op=dist.ReduceOp.SUM)
                        dist.all_reduce(avg_d, op=dist.ReduceOp.SUM)
                        dist.all_reduce(avg_c, op=dist.ReduceOp.SUM)
                        dist.all_reduce(avg_s, op=dist.ReduceOp.SUM)
                        dist.all_reduce(avg_r, op=dist.ReduceOp.SUM)
                        avg_l, avg_d, avg_c, avg_s, avg_r = avg_l.item()/ws, avg_d.item()/ws, avg_c.item()/ws, avg_s.item()/ws, avg_r.item()/ws
                    else:
                        avg_l, avg_d, avg_c, avg_s, avg_r = avg_l.item(), avg_d.item(), avg_c.item(), avg_s.item(), avg_r.item()
                    
                    if rank == 0:
                        wc, ws, wr = args.w_canny, args.w_skel, args.w_repa
                        c_contrib, s_contrib, r_contrib = wc * avg_c, ws * avg_s, wr * avg_r
                        logger.info(
                            f"(step={train_steps:07d}) Total: {avg_l:.4f} | "
                            f"Diff: {avg_d:.4f} | "
                            f"Canny: raw {avg_c:.4f} x {wc:.2f} = {c_contrib:.4f} | "
                            f"Skel: raw {avg_s:.4f} x {ws:.2f} = {s_contrib:.4f} | "
                            f"REPA: raw {avg_r:.4f} x {wr:.2f} = {r_contrib:.4f} | "
                            f"Steps/Sec: {steps_per_sec:.2f}"
                        )
                    
                    running_loss = running_diff = running_canny = running_skel = running_repa = 0
                    log_steps = 0
                    start_time = time()

                if train_steps % args.ckpt_every == 0 and train_steps > 0:
                    if rank == 0:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        # Store only the "changed" part (LoRA + condition head + adaLN/final_layer).
                        # The frozen pretrained body is shared and loaded from disk at restore time,
                        # so it is NOT stored per-checkpoint.
                        delta = extract_full_inference(model_to_save)
                        checkpoint = {
                            "delta": delta,
                            "opt": opt.state_dict(),
                            "args": args
                        }
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
        except Exception as e:
            import traceback
            logger.error(f"Error during training loop: {e}")
            logger.error(traceback.format_exc())
            break
        
        if dist.get_world_size() > 1:
            dist.barrier()

    model.eval()
    logger.info("Done!")
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", type=str, default="train.csv",
                        help="Path to the training CSV (default from config.json).")
    parser.add_argument("--data-dir", type=str, default="", help="Root dataset directory if CSV has relative paths")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained DiT checkpoint")
    parser.add_argument("--reset-cond-head", type=_str_to_bool, default=True,
                        help="After loading pretrained body, re-init adaLN/final_layer (std=0.02) "
                             "to fit new multi-cond head. Prevents early NaN from OOD conditioning.")
    parser.add_argument("--train-cond-head", type=_str_to_bool, default=True,
                        help="Whether adaLN/final_layer (reset by --reset-cond-head) should be "
                             "trainable. True (default) lets them learn after reset; False keeps "
                             "them frozen at their reset random values (legacy behavior).")
    parser.add_argument("--model", type=str, choices=list(DiT_2Cond_models.keys()) + list(DiT_3Cond_models.keys()), default="DiT-2Cond-XL/2")
    parser.add_argument("--cond-mode", type=str, choices=["2cond", "3cond"], default="2cond",
                        help="Conditioning mode: 2cond (callig+char) or 3cond (callig+script+char).")
    parser.add_argument("--num-scripts", type=int, default=12,
                        help="Number of script classes (only used in 3cond mode).")
    parser.add_argument("--use-checkpoint", type=_str_to_bool, default=True,
                        help="Enable gradient checkpointing on DiT blocks (cuts activation memory).")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-calligraphers", type=int, default=2021)
    parser.add_argument("--num-characters", type=int, default=7765)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--global-batch-size", type=int, default=16) # default small batch for laptop GPU
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--vae-path", type=str, default="pretrained_models/sd-vae-ft-ema", help="Local path to VAE weights")
    parser.add_argument("--use-lora", type=_str_to_bool, default=True, help="Use LoRA for fine-tuning DiT blocks")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=None,
                        help="LoRA alpha (scaling = alpha/r). Default: same as r (scaling=1).")
    parser.add_argument("--resume-lora", type=str, default=None,
                        help="Path to a previous LoRA checkpoint to upgrade from (rank up, preserving learned deltas).")
    parser.add_argument("--old-lora-r", type=int, default=16,
                        help="Rank of the LoRA checkpoint given by --resume-lora.")
    parser.add_argument("--resume-full", type=str, default=None,
                        help="Path to a training checkpoint (our own, with delta/opt/args) to resume from. "
                             "Loads the delta (LoRA + condition head + adaLN), optimizer state and step "
                             "counter; the pretrained body is still loaded from --pretrained (delta stores "
                             "only the changed part).")
    parser.add_argument("--resume-lr", type=float, default=None,
                        help="If set with --resume-full, override the learning rate from the checkpoint "
                             "(e.g. lower LR to test whether NaN was numerical).")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=10_000)
    parser.add_argument("--w-canny", type=float, default=0.05, help="Weight for canny structural loss")
    parser.add_argument("--w-skel", type=float, default=0.05, help="Weight for skeleton structural loss")
    parser.add_argument("--w-repa", type=float, default=1.0, help="Weight for Representation Alignment (REPA) Loss")
    parser.add_argument("--repa-teacher-ckpt", type=str, default="",
                        help="Local path to DINOv2 teacher weights (ModelScope safetensors). "
                             "Empty = auto-detect pretrained_models/dinov2_vits14_pretrain.safetensors or $DINO_WEIGHTS.")
    parser.add_argument("--use-canny", type=_str_to_bool, default=False,
                        help="Enable Canny structural loss (requires canny maps in dataset/canny).")
    parser.add_argument("--use-skel", type=_str_to_bool, default=False,
                        help="Enable Skeleton structural loss (requires skeleton maps in dataset/skeleton).")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to JSON config file with default args (CLI overrides).")

    # Apply config-file defaults first, then CLI overrides.
    config_defaults = {}
    cfg_path = parser.parse_known_args()[0].config
    if cfg_path and os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            config_defaults = json.load(f)
    for action in parser._actions:
        if action.dest in ("help", "config"):
            continue
        if action.dest in config_defaults:
            # config supplies a value: use it as default and drop "required"
            action.default = _coerce(config_defaults[action.dest], action.default)
            action.required = False

    args = parser.parse_args()
    main(args)
