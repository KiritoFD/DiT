import os
os.environ["XFORMERS_DISABLED"] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
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
import datetime
import copy
import re
import hashlib
import platform
import math

from models import DiT_2Cond_models, DiT_3Cond_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model

from dataset import MCCDDataset
from latent_dataset import MCCDLatentDataset
from losses import SobelCannyLoss, SkeletonLoss, SkeletonLossV2, EdgeGradientLoss, REPALoss
from torch.utils.checkpoint import checkpoint as grad_ckpt
from lora import inject_lora, upgrade_lora_rank, extract_full_inference
from samplers import DistributedFactorBalancedSampler
from latent_structure import LatentStructureLoss, LatentStructureProbe

def _coerce(value, template, target_type=None):
    """Coerce a config.json value to the type of the argparse default."""
    if value is None or (isinstance(value, str) and value.lower() in ("none", "null", "")):
        return None
    if isinstance(template, bool):
        return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes", "on")
    if target_type is not None:
        return target_type(value)
    if isinstance(template, int):
        return int(value)
    if isinstance(template, float):
        return float(value)
    return str(value)


def _str_to_bool(value):
    """Single-arg bool parser for argparse type=."""
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "on")


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


@torch.no_grad()
def update_ema(ema_model, model, decay):
    """Update a full-precision model EMA, including floating-point buffers."""
    source = model.module if hasattr(model, "module") else model
    source_params = dict(source.named_parameters())
    for name, ema_param in ema_model.named_parameters():
        ema_param.mul_(decay).add_(source_params[name].detach(), alpha=1.0 - decay)
    source_buffers = dict(source.named_buffers())
    for name, ema_buffer in ema_model.named_buffers():
        source_buffer = source_buffers[name].detach()
        if torch.is_floating_point(ema_buffer):
            ema_buffer.mul_(decay).add_(source_buffer, alpha=1.0 - decay)
        else:
            ema_buffer.copy_(source_buffer)

def _state_to_cpu(obj):
    """Recursively move tensors in a (possibly nested) state dict to CPU.

    opt.state_dict() nests dicts two levels deep (state -> param_idx -> tensors)
    and lists (param_groups), so a flat .detach().cpu() pass is not enough.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _state_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_state_to_cpu(v) for v in obj]
    return obj

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
        model_string_name = args.model.replace("/", "-")
        # Timestamp-named experiment dir (unique per launch, never collides or overwrites).
        _ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        _name = getattr(args, "experiment_name", "") or model_string_name
        _name = re.sub(r"[^A-Za-z0-9._-]+", "-", _name).strip("-")
        experiment_dir = f"{args.results_dir}/{_ts}-{_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        # 供 auto_eval_cpu（独立 CPU 进程）定位当前活动实验的 ckpt 目录。
        with open(f"{args.results_dir}/_active_ckpt_dir.txt", "w", encoding="utf-8") as _m:
            _m.write(checkpoint_dir + "\n")
        # log.txt lives inside this experiment dir (created first), never overwritten.
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        with open(f"{experiment_dir}/resolved_config.json", "w", encoding="utf-8") as _cf:
            json.dump(vars(args), _cf, ensure_ascii=False, indent=2)
        _sources = {}
        for _path in ("models.py", "train.py", "losses.py", "latent_dataset.py",
                      "latent_structure.py", "samplers.py", "eval_auto.py"):
            if os.path.isfile(_path):
                with open(_path, "rb") as _sf:
                    _sources[_path] = hashlib.sha256(_sf.read()).hexdigest()
        _probe_path = getattr(args, "latent_structure_probe", None)
        if _probe_path and os.path.isfile(_probe_path):
            with open(_probe_path, "rb") as _pf:
                _sources[f"probe:{_probe_path}"] = hashlib.sha256(_pf.read()).hexdigest()
        with open(f"{experiment_dir}/source_manifest.json", "w", encoding="utf-8") as _mf:
            json.dump({
                "created_at": datetime.datetime.now().isoformat(),
                "hostname": platform.node(),
                "python": sys.version,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "sha256": _sources,
            }, _mf, ensure_ascii=False, indent=2)
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
            use_checkpoint=args.use_checkpoint,
            condition_fusion=args.condition_fusion,
            callig_embed_dim=args.callig_embed_dim,
            script_embed_dim=args.script_embed_dim,
            char_embed_dim=args.char_embed_dim,
            cond_drop_all_prob=args.cond_drop_all_prob,
            cond_drop_one_prob=args.cond_drop_one_prob,
        )
        logger.info(f"Building 3-Cond model: {args.model} "
                    f"(callig={args.num_calligraphers}, script={args.num_scripts}, char={args.num_characters}, "
                    f"fusion={args.condition_fusion}, dims={args.callig_embed_dim}/"
                    f"{args.script_embed_dim}/{args.char_embed_dim}, dropout="
                    f"all:{args.cond_drop_all_prob}, one:{args.cond_drop_one_prob})")
    else:
        if args.model not in DiT_2Cond_models:
            raise ValueError(f"cond_mode=2cond but model '{args.model}' is not a 2Cond model. "
                             f"Use one of {list(DiT_2Cond_models.keys())}.")
        model = DiT_2Cond_models[args.model](
            input_size=latent_size,
            num_calligraphers=args.num_calligraphers,
            num_characters=args.num_characters,
            use_checkpoint=args.use_checkpoint,
            condition_fusion=args.condition_fusion,
            callig_embed_dim=args.callig_embed_dim,
            char_embed_dim=args.char_embed_dim,
            cond_drop_all_prob=args.cond_drop_all_prob,
            cond_drop_one_prob=args.cond_drop_one_prob,
            skel_head_enabled=getattr(args, 'w_skel_head', 0) > 0,
            use_glyph_cond=getattr(args, 'w_glyph_cond', 0) > 0,
            glyph_scale_init=getattr(args, 'glyph_scale_init', 0.4),
        )
        logger.info(f"Building 2-Cond model: {args.model} "
                    f"(callig={args.num_calligraphers}, glyph/char={args.num_characters}, "
                    f"fusion={args.condition_fusion}, dims={args.callig_embed_dim}/"
                    f"{args.char_embed_dim}, dropout=all:{args.cond_drop_all_prob}, "
                    f"one:{args.cond_drop_one_prob}, skel_head={getattr(args, 'w_skel_head', 0) > 0}, "
                    f"glyph_cond={getattr(args, 'w_glyph_cond', 0) > 0}, glyph_scale_init={getattr(args, 'glyph_scale_init', 0.4)})")

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
    # 条件调制层总是重置从头学（无论 legacy/factorized_add/xl_highdim）。
    # 关键认知：ImageNet 预训练 adaLN/final_layer/y_embedder 学的是"1000类自然物分类
    # →调制"，与书法(callig×glyph)条件完全正交。保留它= 强迫模型用 ImageNet 分类
    # 眼光生成，导致乱码/跑偏。我们只保留通用扩散引擎（t_embedder/x_embedder/attn/mlp），
    # 条件调制一律从头学，由训练目标(结构loss+diff)自行建立"条件→生成"耦合。
    if getattr(args, 'resume_full', None) is None:
        import torch.nn as _nn
        for _b in model.blocks:
            _nn.init.normal_(_b.adaLN_modulation[-1].weight, std=0.02)
            _nn.init.normal_(_b.adaLN_modulation[-1].bias, std=0.02)
        _nn.init.normal_(model.final_layer.adaLN_modulation[-1].weight, std=0.02)
        _nn.init.normal_(model.final_layer.adaLN_modulation[-1].bias, std=0.02)
        _nn.init.normal_(model.final_layer.linear.weight, std=0.02)
        logger.info("[cond-head] reset adaLN/final_layer to std=0.02 (retain task-agnostic "
                    "transformer engine, drop ImageNet class-condition coupling).")

    if getattr(args, 'use_lora', True):
        _r = getattr(args, 'lora_r', 16)
        _alpha = getattr(args, 'lora_alpha', None)
        if _alpha is None:
            _alpha = _r  # default scaling = 1
        logger.info(f"Injecting LoRA (r={_r}, alpha={_alpha}, scaling={_alpha/_r:.2f}, target={getattr(args, 'lora_target', 'all')}) into DiT blocks...")
        model = inject_lora(model, r=_r, lora_alpha=_alpha, target=getattr(args, 'lora_target', 'all'))

        # Optional: upgrade LoRA rank from a previous run's checkpoint, preserving
        # the learned low-rank deltas. See lora.upgrade_lora_rank for the strategy.
        if getattr(args, 'resume_lora', None):
            import torch as _torch
            _sd = _torch.load(args.resume_lora, map_location="cpu")
            _sd = _sd.get("state_dict", _sd) if isinstance(_sd, dict) else _sd
            old_r = getattr(args, 'old_lora_r', 16)
            model = upgrade_lora_rank(model, _r, _alpha, _sd, old_r)

    # 3) full resume works for both LoRA and full-from-scratch checkpoints.
    if getattr(args, 'resume_full', None) is not None:
        import torch as _torch
        _rf = _torch.load(args.resume_full, map_location="cpu", weights_only=False)
        _resume_full_ckpt = _rf
        _sd = _rf.get("delta", _rf.get("model", _rf))
        missing, unexpected = model.load_state_dict(_sd, strict=False)
        logger.info(f"[resume-full] Loaded weights from {args.resume_full} "
                    f"(missing={len(missing)}, unexpected={len(unexpected)}).")

    # ---- freeze / trainable policy (independent of use_lora) -----------------
    # Two regimes:
    #   1) pretrained body (+ optional LoRA): freeze the pretrained transformer body,
    #      train only the *new* condition head + adaLN/final_layer (+ lora_* if injected).
    #      adaLN is reset to std=0.02 by `reset_cond_head`, so it MUST be trainable
    #      (`train_cond_head=true`), otherwise the model is stuck on random modulation.
    #   2) from-scratch (pretrained=None and use_lora=false): keep all params trainable.
    _has_pretrained = args.pretrained is not None
    if getattr(args, 'use_lora', True) or _has_pretrained:
        requires_grad(model, False)
        train_cond_head = getattr(args, 'train_cond_head', True)
        for name, param in model.named_parameters():
            if ('lora_' in name or 'y_callig_embedder' in name or 'y_char_embedder' in name
                    or 'cond_fusion' in name or 'y_script_embedder' in name
                    or 'callig_proj' in name or 'script_proj' in name or 'char_proj' in name
                    or 'y_scale' in name or 'skel_head' in name or 'glyph_scale' in name
                    or 'glyph_embedder' in name):
                param.requires_grad = True
            elif train_cond_head and ('adaLN' in name or 'final_layer' in name):
                param.requires_grad = True

    # report trainable counts
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info(f"Trainable Parameters: {trainable_params:,}")
    logger.info(f"Frozen Parameters: {frozen_params:,} (trainable ratio: {trainable_params/(trainable_params+frozen_params)*100:.2f}%)")

    model = model.to(device)
    ema_model = None
    if getattr(args, "use_ema", False):
        ema_model = copy.deepcopy(model).eval()
        requires_grad(ema_model, False)
        if _resume_full_ckpt is not None and _resume_full_ckpt.get("ema") is not None:
            ema_model.load_state_dict(_resume_full_ckpt["ema"], strict=True)
            logger.info("[EMA] restored EMA weights from checkpoint")
        logger.info(f"[EMA] enabled with decay={args.ema_decay}")
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
    # use_struct_loss_v2: 修复版损失 (骨架只做正向牵引 + 边缘梯度场直接匹配 GT)
    if getattr(args, 'use_struct_loss_v2', False):
        canny_loss_fn = EdgeGradientLoss().to(device)
        skel_loss_fn = SkeletonLossV2().to(device)
        logger.info("[struct] use_struct_loss_v2: EdgeGradientLoss + SkeletonLossV2 "
                    "(骨架 recall-only, 边缘梯度场匹配)")
    else:
        canny_loss_fn = SobelCannyLoss().to(device)
        skel_loss_fn = SkeletonLoss(lambda_bg=1.0).to(device)

    latent_structure_loss_fn = None
    if args.w_latent_canny > 0 or args.w_latent_skel > 0:
        structure_probe = None
        if args.w_latent_skel > 0:
            if not args.latent_structure_probe:
                raise ValueError("w_latent_skel > 0 requires --latent-structure-probe")
            probe_ckpt = torch.load(
                args.latent_structure_probe, map_location="cpu", weights_only=False)
            probe_args = probe_ckpt.get("args", {})
            structure_probe = LatentStructureProbe(
                width=int(probe_args.get("width", 32)),
                depth=int(probe_args.get("depth", 2)))
            structure_probe.load_state_dict(probe_ckpt["model"], strict=True)
            structure_probe.to(device)
            logger.info(
                f"[latent-structure] frozen probe={args.latent_structure_probe} "
                f"metrics={probe_ckpt.get('metrics', {})}")
        latent_structure_loss_fn = LatentStructureLoss(
            probe=structure_probe, max_timestep=args.latent_struct_max_t).to(device)
        logger.info(
            f"[latent-structure] enabled: canny={args.w_latent_canny}, "
            f"skeleton={args.w_latent_skel}, max_t={args.latent_struct_max_t}")
    
    repa_loss_fn = None
    if args.w_repa > 0:
        teacher_ckpt = getattr(args, "repa_teacher_ckpt", "") or None
        logger.info(f"Initializing REPA Loss (Teacher: dinov2_vits14, ckpt={teacher_ckpt or 'auto'}, Student Dim: {student_hidden_size})")
        repa_loss_fn = REPALoss(student_dim=student_hidden_size, teacher_backbone="dinov2_vits14",
                                teacher_ckpt=teacher_ckpt).to(device)

    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    if repa_loss_fn is not None:
        trainable_params_list.extend([p for p in repa_loss_fn.proj.parameters() if p.requires_grad])
        
    opt = torch.optim.AdamW(trainable_params_list, lr=args.lr, weight_decay=args.weight_decay)

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
    use_latent = bool(getattr(args, "latent_shards_dir", None))
    need_canny_map = args.use_canny or args.w_latent_canny > 0
    need_skel_map = args.use_skel or args.w_latent_skel > 0 or getattr(args, 'w_skel_head', 0) > 0
    if use_latent:
        dataset = MCCDLatentDataset(csv_file=args.data_csv,
                                    latent_shards_dir=args.latent_shards_dir,
                                    img_root=args.img_root,
                                    canny_root=args.canny_root if need_canny_map else None,
                                    image_size=args.image_size,
                                    load_canny=need_canny_map,
                                    load_skel=need_skel_map,
                                    skel_root=args.skel_root if need_skel_map else None,
                                    preload=bool(getattr(args, 'preload', False)),
                                    load_image=(args.w_repa > 0 or getattr(args, 'use_struct_loss_v2', False)),
                                    num_preload_workers=int(getattr(args, 'preload_workers', 16)),
                                    structure_size=(32 if (latent_structure_loss_fn is not None or getattr(args, 'w_skel_head', 0) > 0) else 256),
                                    use_glyph_cond=getattr(args, 'w_glyph_cond', False))
        logger.info("Using latent-cached dataset (skip on-the-fly VAE encode)."
                    + (" preload=ON" if getattr(args, 'preload', False) else ""))
    else:
        dataset = MCCDDataset(csv_file=args.data_csv, root_dir=args.data_dir, image_size=args.image_size,
                              load_canny=need_canny_map, load_skel=need_skel_map)
    if args.sampler == "factor_balanced":
        sampler = DistributedFactorBalancedSampler(
            dataset, num_replicas=dist.get_world_size(), rank=rank, seed=args.global_seed,
            char_alpha=args.balance_char_alpha,
            callig_alpha=args.balance_callig_alpha)
        logger.info(f"Using factor-balanced sampler: {sampler.summary()} "
                    f"(char_alpha={args.balance_char_alpha}, "
                    f"callig_alpha={args.balance_callig_alpha})")
    else:
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
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else 2,
    )
    logger.info(f"Dataset contains {len(dataset):,} images")

    total_planned_steps = args.max_steps if args.max_steps > 0 else args.epochs * len(loader)
    if getattr(args, 'fresh_scheduler', False) and _resume_full_ckpt is not None and args.max_steps > 0:
        total_planned_steps = max(args.max_steps - resume_start_step, 1)
        logger.info(f"[LR] fresh-scheduler: fine-tune horizon = {total_planned_steps} steps "
                    f"(max_steps {args.max_steps} - resume {resume_start_step})")
    scheduler = None
    if args.lr_schedule == "cosine":
        warmup_steps = min(args.warmup_steps, max(total_planned_steps - 1, 0))

        def _lr_scale(step):
            if warmup_steps > 0 and step < warmup_steps:
                return max((step + 1) / warmup_steps, 1e-8)
            progress = ((step - warmup_steps)
                        / max(total_planned_steps - warmup_steps, 1))
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_scale)
        if (_resume_full_ckpt is not None and _resume_full_ckpt.get("scheduler") is not None
                and not getattr(args, 'fresh_scheduler', False)):
            scheduler.load_state_dict(_resume_full_ckpt["scheduler"])
            logger.info("[LR] restored scheduler state")
        elif getattr(args, 'fresh_scheduler', False):
            logger.info("[LR] fresh-scheduler: starting from base lr "
                        f"{opt.param_groups[0]['lr']:.2e} (old scheduler state ignored)")
        logger.info(f"[LR] cosine schedule: warmup={warmup_steps}, "
                    f"total={total_planned_steps}, min_ratio={args.min_lr_ratio}")

    model.train()

    train_steps = resume_start_step
    log_steps = 0
    running_loss = 0
    running_diff = 0
    running_canny = 0
    running_skel = 0
    running_latent_canny = 0
    running_latent_skel = 0
    running_repa = 0
    running_x0lat = 0
    running_skel_head = 0
    running_std_mid = 0
    nan_steps = 0
    current_ema_decay = args.ema_decay
    start_time = time()

    # ---- 早停状态 (基于 CPU eval 的 eval_auto_*.json mse/ssim) ----
    early_stop_best = None      # 最佳 metric 值 (ssim 越大越好 / mse 越小越好)
    early_stop_stale = 0        # 连续未改善的 eval 次数
    early_stop_last_eval_step = -1
    early_stop_stopped = False
    _es_metric = getattr(args, 'early_stop_metric', 'ssim')
    _es_better = ((lambda a, b: a > b) if _es_metric == 'ssim' else (lambda a, b: a < b))
    _es_check_every = int(getattr(args, 'early_stop_check_every', 0))
    if _es_check_every <= 0:
        _es_check_every = max(int(getattr(args, 'ckpt_every', 5000)) // 2, 1000)

    def _early_stop_check(force=False):
        """读 ckpt 目录最新 eval_auto json, 更新 best/stale; 达到 patience 返回 True 表示停。"""
        nonlocal early_stop_best, early_stop_stale, early_stop_last_eval_step
        if not getattr(args, 'early_stop', False):
            return False
        ev_files = sorted(glob(os.path.join(checkpoint_dir, "eval_auto_*.json")))
        if not ev_files:
            return False
        last_ev = ev_files[-1]
        ev_step = int(os.path.basename(last_ev).replace("eval_auto_", "").replace(".json", ""))
        if ev_step <= early_stop_last_eval_step:
            return False
        early_stop_last_eval_step = ev_step
        try:
            with open(last_ev, "r", encoding="utf-8") as _f:
                d = json.load(_f)
            m, s = d.get("mse"), d.get("ssim")
            if _es_metric == 'ssim':
                val = float(s) if s is not None else None
            else:
                val = float(m) if m is not None else None
        except Exception:
            return False
        if val is None:
            return False
        if early_stop_best is None or _es_better(val, early_stop_best):
            early_stop_best = val
            early_stop_stale = 0
            logger.info(f"[early-stop] eval step {ev_step}: {_es_metric}={val:.4f} (new best)")
        else:
            early_stop_stale += 1
            logger.info(f"[early-stop] eval step {ev_step}: {_es_metric}={val:.4f} "
                        f"(best {early_stop_best:.4f}, stale {early_stop_stale}/"
                        f"{args.early_stop_patience})")
            if early_stop_stale >= int(getattr(args, 'early_stop_patience', 5)):
                logger.info(f"[early-stop] {_es_metric} no improvement for "
                            f"{early_stop_stale} evals; early stopping.")
                return True
        return False

    logger.info(f"Training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        if rank == 0:
            logger.info(f"Beginning epoch {epoch}...")
        
        try:
            for batch_idx, batch in enumerate(loader):
                y_callig = batch['y_callig'].to(device)
                y_char = batch['y_char'].to(device)
                if cond_mode == "3cond":
                    y_script = batch['y_script'].to(device)

                if 'latent' in batch:
                    # Latent-cached training: latent pre-encoded (scaled by 0.18215); image kept for gt-losses.
                    x_latent = batch['latent'].to(device)
                    x = batch['image'].to(device)
                    canny_gt = batch['canny'].to(device)
                    skel_gt = batch['skeleton'].to(device) if need_skel_map else None
                else:
                    x = batch['image'].to(device)
                    canny_gt = batch['canny'].to(device)
                    skel_gt = batch['skeleton'].to(device)
                    # VAE encode stays in fp32 for numerical stability (VAE is sensitive to low precision).
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float32):
                        x_latent = vae.encode(x).latent_dist.sample().mul_(0.18215)
                        x_latent = x_latent.float()

                t = torch.randint(0, diffusion.num_timesteps, (x_latent.shape[0],), device=device)
                if cond_mode == "3cond":
                    model_kwargs = dict(y_callig=y_callig, y_script=y_script, y_char=y_char)
                else:
                    model_kwargs = dict(y_callig=y_callig, y_char=y_char)
                # 标准字形条件 g(甲2 token-add): batch 由 dataset 提供, None=禁用对应项
                if getattr(args, 'w_glyph_cond', False) and 'g' in batch and batch['g'].numel() > 0:
                    model_kwargs['g'] = batch['g'].to(device)   # (N,4,32,32)
                
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
                loss_latent_canny = torch.tensor(0.0, device=device)
                loss_latent_skel = torch.tensor(0.0, device=device)
                # X0Lat：模型预测的干净 latent 与 GT latent 的 MSE（直接回答
                # “输出和 GT latent 比”的指标；注意它随 t 混合了高噪声步，
                # 高噪声步 x0 预测天然不准，所以数值比 eps-MSE 大是正常的）。
                loss_x0lat = torch.tensor(0.0, device=device)

                pred_xstart_latent = loss_dict.get("pred_xstart", None)
                if pred_xstart_latent is not None:
                    loss_x0lat = ((pred_xstart_latent - x_latent) ** 2).mean()

                # ---- MIDSTEP_STD: 中间噪声水平, 让去噪结果 x0_pred 逼近标准字形 latent g。
                # 主损失从 GT x0 学报内容+风格; 此项在 sqrt(alpha_cumprod)∈[alo,ahi] 的中段噪声,
                # 额外把模型预测的 clean latent 拉向标准字形 latent g, 使字形结构在去噪中段被锚定。
                # 权重须明显小于主 loss, 避免抹掉书家风格。仅当使用 glyph 条件时生效。采样端不变。
                loss_std_mid = torch.tensor(0.0, device=device)
                if (getattr(args, 'w_std_mid', 0.0) > 0
                        and pred_xstart_latent is not None
                        and model_kwargs.get('g') is not None):
                    _sqrt_a = torch.as_tensor(diffusion.sqrt_alphas_cumprod, device=device)
                    _a_t = _sqrt_a[t]                       # (N,)
                    _alo = float(getattr(args, 'std_mid_alo', 0.35))
                    _ahi = float(getattr(args, 'std_mid_ahi', 0.75))
                    _mid = (_a_t >= _alo) & (_a_t <= _ahi)  # (N,) bool: 中间噪声水平子集
                    if bool(_mid.any()):
                        _g = model_kwargs['g'].float()      # (N,4,32,32) 标准字形 latent
                        _p = pred_xstart_latent.float()
                        # 归一化到该子集作均值 (不按全 batch, 排除无监督噪声步)
                        loss_std_mid = ((_p[_mid] - _g[_mid]) ** 2).mean()

                # 骨架辅助头监督（latent 空间，训练引导 / 推理不用）：
                # forward 在 skel_head 启用时返回 (主输出, skel_pred)，
                # gaussian_diffusion.training_losses 把第二元素存进 loss_dict['intermediate_feats']。
                loss_skel_head = torch.tensor(0.0, device=device)
                if getattr(args, 'w_skel_head', 0) > 0:
                    skel_pred = loss_dict.get("intermediate_feats", None)
                    if skel_pred is not None and skel_gt is not None and skel_gt.numel() > 0:
                        # batch 可能取整后被 drop_last 截断，对齐批次
                        _n = min(skel_pred.shape[0], skel_gt.shape[0])
                        _skel_gt = skel_gt[:_n].float()
                        _skel_pred = skel_pred[:_n].float()
                        # BCE with logits（骨架头输出未 sigmoid）
                        loss_skel_head = torch.nn.functional.binary_cross_entropy_with_logits(
                            _skel_pred, _skel_gt).mean()
                if x is not None and pred_xstart_latent is not None and (args.use_canny or args.use_skel):
                    # Infra 优化：pixel 结构损失只需要在 batch 的一个随机子集上做
                    # differentiable VAE decode（结构监督是低频辅助信号，子集采样
                    # 是无偏估计）。默认 32 张，decode 显存从"全 batch"降为固定小量。
                    # t 门控 (struct_max_t>0)：只在低噪声步 (t<=tmax) 施加结构损失。
                    # 高噪声步的 x0 预测本来就是一团糊，逼它在此刻"成像"会让 x0
                    # 整体漂出 VAE 流形（正是旧实验 X0Lat 30~50 的直接原因）。
                    _ss = int(getattr(args, 'struct_subset', 32))
                    _struct_tmax = int(getattr(args, 'struct_max_t', 0))
                    _B = pred_xstart_latent.shape[0]
                    _use_this_step = True
                    if _struct_tmax > 0:
                        _gidx = torch.nonzero(t <= _struct_tmax).view(-1)
                        if _gidx.numel() == 0:
                            _use_this_step = False
                        elif _ss > 0 and _gidx.numel() > _ss:
                            _perm = torch.randperm(_gidx.numel(), device=t.device)[:_ss]
                            _idx = _gidx[_perm]
                        else:
                            _idx = _gidx
                    else:
                        if _ss > 0 and _ss < _B:
                            _idx = torch.randperm(_B, device=pred_xstart_latent.device)[:_ss]
                        else:
                            _idx = None
                    if _use_this_step and _idx is None:
                        pred_xstart_sub = pred_xstart_latent
                        canny_gt_sub = canny_gt if args.use_canny else None
                        skel_gt_sub = skel_gt if args.use_skel else None
                        x_sub = x
                    elif _use_this_step:
                        pred_xstart_sub = pred_xstart_latent[_idx]
                        x0_pred = None
                        canny_gt_sub = canny_gt[_idx] if args.use_canny else None
                        skel_gt_sub = skel_gt[_idx] if args.use_skel else None
                        x_sub = x[_idx]
                    if _use_this_step:
                        # VAE decode: optional bf16 autocast + optional lower-resolution decode.
                        #  - struct_decode_bf16: bf16 has the same exponent range as fp32, so the
                        #    SD-VAE decoder cannot overflow like fp16 AMP; the coarser mantissa only
                        #    adds mild noise to an auxiliary structural loss. Output is cast back to
                        #    fp32 before the losses for stability.
                        #  - struct_decode_scale<1: feed a proportionally smaller latent into the
                        #    fully-convolutional decoder so it emits a lower-res image (e.g. 0.5 ->
                        #    128x128, ~4x cheaper); GT maps are resized to match.
                        _dscale = float(getattr(args, 'struct_decode_scale', 1.0))
                        _decode_dtype = (torch.bfloat16 if getattr(args, 'struct_decode_bf16', False)
                                         else torch.float32)
                        _decode_in = pred_xstart_sub.float() / 0.18215
                        if _dscale < 1.0:
                            _decode_in = F.interpolate(
                                _decode_in, scale_factor=_dscale, mode="area")

                        def _decode(z):
                            return vae.decode(z).sample
                        with torch.autocast("cuda", dtype=_decode_dtype):
                            x0_pred = grad_ckpt(_decode, _decode_in, use_reentrant=False)
                        x0_pred = x0_pred.float()
                        if _dscale < 1.0:
                            if canny_gt_sub is not None:
                                canny_gt_sub = F.interpolate(
                                    canny_gt_sub, scale_factor=_dscale, mode="nearest")
                            if skel_gt_sub is not None:
                                skel_gt_sub = F.interpolate(
                                    skel_gt_sub, scale_factor=_dscale, mode="nearest")
                            x_sub = F.interpolate(x_sub, scale_factor=_dscale, mode="area")
                        # Structural losses computed in fp32 (outside autocast) for stability.
                        if args.use_canny:
                            if getattr(args, 'use_struct_loss_v2', False):
                                loss_canny = canny_loss_fn(x0_pred, x_sub)
                            else:
                                loss_canny = canny_loss_fn(x0_pred, canny_gt_sub)
                        if args.use_skel:
                            loss_skel = skel_loss_fn(x0_pred, skel_gt_sub)

                if latent_structure_loss_fn is not None and pred_xstart_latent is not None:
                    latent_structure_losses = latent_structure_loss_fn(
                        pred_xstart_latent, x_latent, t,
                        canny=canny_gt if need_canny_map else None,
                        skeleton=skel_gt if need_skel_map else None)
                    loss_latent_canny = latent_structure_losses["canny"]
                    loss_latent_skel = latent_structure_losses["skeleton"]

                intermediate_feats = loss_dict.get("intermediate_feats", None)
                if x is not None and intermediate_feats is not None and repa_loss_fn is not None and args.w_repa > 0:
                    # original 'x' is ground truth x_0 [-1, 1]
                    loss_repa = repa_loss_fn(intermediate_feats, x)

                # Struct-loss weight ramp: linearly bring canny/skel from 0 to target over
                # --struct-warmup-steps fine-tune steps (counted from the resume point) so a
                # converged diff-only checkpoint adapts gradually instead of being jolted.
                _struct_scale = 1.0
                if int(getattr(args, 'struct_warmup_steps', 0)) > 0:
                    _steps_ft = max(0, train_steps - resume_start_step)
                    _struct_scale = min(1.0, _steps_ft / float(args.struct_warmup_steps))
                loss = (loss_diff
                        + args.w_canny * _struct_scale * loss_canny
                        + args.w_skel * _struct_scale * loss_skel
                        + args.w_latent_canny * loss_latent_canny
                        + args.w_latent_skel * loss_latent_skel
                        + args.w_repa * loss_repa
                        + getattr(args, 'w_skel_head', 0) * loss_skel_head
                        + getattr(args, 'w_std_mid', 0.0) * loss_std_mid)

                opt.zero_grad()
                # NaN guard: skip the step if loss is not finite (e.g. a bad sample).
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params_list, max_norm=1.0)
                    opt.step()
                    if scheduler is not None:
                        scheduler.step()
                    if ema_model is not None:
                        if args.ema_warmup:
                            # Early fixed 0.9999 EMA retains mostly random initialization
                            # for thousands of steps. This update-count cap is the standard
                            # warm-start schedule: 0.1 initially, ~0.991 at 1k, ~0.9991 at 10k.
                            current_ema_decay = min(
                                args.ema_decay, (1.0 + train_steps) / (10.0 + train_steps))
                        else:
                            current_ema_decay = args.ema_decay
                        update_ema(ema_model, model, current_ema_decay)
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
                    running_latent_canny += loss_latent_canny.item()
                    running_latent_skel += loss_latent_skel.item()
                    running_repa += loss_repa.item()
                    running_skel_head += loss_skel_head.item()
                    running_std_mid += loss_std_mid.item()
                    running_x0lat += loss_x0lat.item()
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
                    avg_lc = torch.tensor(running_latent_canny / divisor, device=device)
                    avg_ls = torch.tensor(running_latent_skel / divisor, device=device)
                    avg_r = torch.tensor(running_repa / divisor, device=device)
                    avg_x0 = torch.tensor(running_x0lat / divisor, device=device)
                    avg_skel_h = torch.tensor(running_skel_head / divisor, device=device)
                    avg_std_mid = torch.tensor(running_std_mid / divisor, device=device)
                    world_size = dist.get_world_size()
                    if world_size > 1:
                        dist.all_reduce(avg_l, op=dist.ReduceOp.SUM)
                        dist.all_reduce(avg_d, op=dist.ReduceOp.SUM)
                        dist.all_reduce(avg_c, op=dist.ReduceOp.SUM)
                        dist.all_reduce(avg_s, op=dist.ReduceOp.SUM)
                        dist.all_reduce(avg_lc, op=dist.ReduceOp.SUM)
                        dist.all_reduce(avg_ls, op=dist.ReduceOp.SUM)
                        dist.all_reduce(avg_r, op=dist.ReduceOp.SUM)
                        dist.all_reduce(avg_x0, op=dist.ReduceOp.SUM)
                        dist.all_reduce(avg_skel_h, op=dist.ReduceOp.SUM)
                        dist.all_reduce(avg_std_mid, op=dist.ReduceOp.SUM)
                        avg_l, avg_d = avg_l.item()/world_size, avg_d.item()/world_size
                        avg_c, avg_s = avg_c.item()/world_size, avg_s.item()/world_size
                        avg_lc, avg_ls = avg_lc.item()/world_size, avg_ls.item()/world_size
                        avg_r = avg_r.item()/world_size
                        avg_x0 = avg_x0.item()/world_size
                        avg_skel_h = avg_skel_h.item()/world_size
                        avg_std_mid = avg_std_mid.item()/world_size
                    else:
                        avg_l, avg_d = avg_l.item(), avg_d.item()
                        avg_c, avg_s = avg_c.item(), avg_s.item()
                        avg_lc, avg_ls = avg_lc.item(), avg_ls.item()
                        avg_r = avg_r.item()
                        avg_x0 = avg_x0.item()
                        avg_skel_h = avg_skel_h.item()
                        avg_std_mid = avg_std_mid.item()
                    
                    if rank == 0:
                        wc = args.w_canny * _struct_scale
                        ws = args.w_skel * _struct_scale
                        wr = args.w_repa
                        c_contrib, s_contrib, r_contrib = wc * avg_c, ws * avg_s, wr * avg_r
                        latent_c_contrib = args.w_latent_canny * avg_lc
                        latent_s_contrib = args.w_latent_skel * avg_ls
                        ema_log = (f"EMA: {current_ema_decay:.6f} | "
                                   if ema_model is not None else "")
                        logger.info(
                            f"(step={train_steps:07d}) Total: {avg_l:.4f} | "
                            f"Diff: {avg_d:.4f} | "
                            f"Canny: raw {avg_c:.4f} x {wc:.2f} = {c_contrib:.4f} | "
                            f"Skel: raw {avg_s:.4f} x {ws:.2f} = {s_contrib:.4f} | "
                            f"LatC: raw {avg_lc:.4f} x {args.w_latent_canny:.3f} = {latent_c_contrib:.4f} | "
                            f"LatS: raw {avg_ls:.4f} x {args.w_latent_skel:.3f} = {latent_s_contrib:.4f} | "
                            f"REPA: raw {avg_r:.4f} x {wr:.2f} = {r_contrib:.4f} | "
                            f"SkelH: raw {avg_skel_h:.4f} | "
                            f"StdMid: raw {avg_std_mid:.4f} | "
                            f"X0Lat: raw {avg_x0:.4f} | "
                            f"LR: {opt.param_groups[0]['lr']:.2e} | {ema_log}"
                            f"Steps/Sec: {steps_per_sec:.2f} | "
                            f"Mem: {torch.cuda.memory_reserved() / 1024 ** 3:.2f}G/"
                            f"{torch.cuda.max_memory_reserved() / 1024 ** 3:.2f}G"
                        )
                    
                    running_loss = running_diff = running_canny = running_skel = 0
                    running_latent_canny = running_latent_skel = running_repa = running_x0lat = running_skel_head = 0
                    running_std_mid = 0
                    log_steps = 0
                    start_time = time()

                _save_ckpt = args.ckpt_every > 0 and train_steps % args.ckpt_every == 0
                if train_steps <= 5000:
                    _save_ckpt = args.ckpt_every > 0 and train_steps % 1000 == 0
                elif train_steps > 5000:
                    _save_ckpt = args.ckpt_every > 0 and (train_steps - 5000) % 5000 == 0

                if _save_ckpt and train_steps > 0:
                    if rank == 0:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        # LoRA mode: store only the "changed" part (LoRA + condition head +
                        # adaLN/final_layer) since the frozen body loads from disk at restore time.
                        # Full-pretrain mode (use_lora=false): store the complete state dict.
                        if getattr(args, 'use_lora', True):
                            delta = extract_full_inference(model_to_save)
                        else:
                            delta = model_to_save.state_dict()
                        # Move tensors to CPU before serialize so torch.save never
                        # allocates extra GPU memory (avoids save-time VRAM spikes).
                        delta = _state_to_cpu(delta)
                        _opt_cpu = _state_to_cpu(opt.state_dict())
                        checkpoint = {
                            "delta": delta,
                            "opt": _opt_cpu,
                            "args": args,
                            "train_steps": train_steps,
                        }
                        if ema_model is not None:
                            checkpoint["ema"] = _state_to_cpu(ema_model.state_dict())
                        if scheduler is not None:
                            checkpoint["scheduler"] = scheduler.state_dict()
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        open(checkpoint_path + ".done", "w").close()
                        logger.info(f"Saved checkpoint to {checkpoint_path}")

                        # Rotation: keep only the most recent ckpt_keep checkpoints
                        # (and their eval dirs) to bound disk usage on long runs.
                        ckpt_keep = int(getattr(args, 'ckpt_keep', 0))
                        if ckpt_keep > 0:
                            import shutil as _sh
                            _pts = sorted(glob(f"{checkpoint_dir}/*.pt"))
                            for _old in _pts[:-ckpt_keep]:
                                _base = os.path.basename(_old)[:-3]
                                os.remove(_old)
                                for _suf in (".done",):
                                    if os.path.exists(_old + _suf):
                                        os.remove(_old + _suf)
                                _eval_dir = f"{checkpoint_dir}/eval_{_base}"
                                if os.path.isdir(_eval_dir):
                                    _sh.rmtree(_eval_dir, ignore_errors=True)
                            if len(_pts) > ckpt_keep:
                                logger.info(f"[ckpt-keep] pruned {len(_pts) - ckpt_keep} old checkpoint(s), keeping {ckpt_keep}")

                if args.max_steps > 0 and train_steps >= args.max_steps:
                    logger.info(f"Reached max_steps={args.max_steps}; stopping cleanly.")
                    break

                if (getattr(args, 'early_stop', False)
                        and train_steps >= int(getattr(args, 'early_stop_min_steps', 0))
                        and args.ckpt_every > 0
                        and train_steps % _es_check_every == 0
                        and rank == 0
                        and _early_stop_check()):
                    early_stop_stopped = True
                    break
        except Exception as e:
            import traceback
            logger.error(f"Error during training loop: {e}")
            logger.error(traceback.format_exc())
            break

        if args.max_steps > 0 and train_steps >= args.max_steps:
            break

        if early_stop_stopped:
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
    parser.add_argument("--experiment-name", type=str, default="",
                        help="Meaningful experiment slug appended after the launch timestamp.")
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
    parser.add_argument("--condition-fusion", type=str,
                        choices=["legacy", "factorized_add", "xl_highdim"], default="legacy",
                        help="Cond fusion: legacy joint MLP | factorized_add (low-dim additive) | "
                             "xl_highdim (high-dim, XL-aligned, preserves pretrained adaLN).")
    parser.add_argument("--callig-embed-dim", type=int, default=None)
    parser.add_argument("--script-embed-dim", type=int, default=None)
    parser.add_argument("--char-embed-dim", type=int, default=None)
    parser.add_argument("--cond-drop-all-prob", type=float, default=0.05,
                        help="Probability of dropping all factors for CFG.")
    parser.add_argument("--cond-drop-one-prob", type=float, default=0.0,
                        help="Probability of dropping exactly one uniformly selected factor.")
    parser.add_argument("--num-scripts", type=int, default=12,
                        help="Number of script classes (only used in 3cond mode).")
    parser.add_argument("--use-checkpoint", type=_str_to_bool, default=True,
                        help="Enable gradient checkpointing on DiT blocks (cuts activation memory).")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-calligraphers", type=int, default=2021)
    parser.add_argument("--num-characters", type=int, default=7765)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Optional clean stop after N optimizer steps (0 disables).")
    parser.add_argument("--early-stop", type=_str_to_bool, default=False,
                        help="Enable early stopping based on CPU eval (eval_auto_*.json mse/ssim).")
    parser.add_argument("--early-stop-metric", type=str, choices=["ssim", "mse"], default="ssim",
                        help="Metric to monitor for early stopping (ssim higher better, mse lower better).")
    parser.add_argument("--early-stop-patience", type=int, default=5,
                        help="Stop after this many consecutive evals without improvement.")
    parser.add_argument("--early-stop-min-steps", type=int, default=0,
                        help="Do not early-stop before this many total steps (train_steps).")
    parser.add_argument("--early-stop-check-every", type=int, default=0,
                        help="Check eval_auto json every N training steps (0 = ckpt_every//2, min 1000).")
    parser.add_argument("--struct-warmup-steps", type=int, default=0,
                        help="Linearly ramp w_canny/w_skel from 0 to their target over N "
                             "fine-tune steps counted from the resume point (0 = full weight "
                             "immediately). Lets a converged diff-only checkpoint adapt to "
                             "structural losses gradually.")
    parser.add_argument("--fresh-scheduler", type=_str_to_bool, default=False,
                        help="With --resume-full: ignore the restored scheduler state and "
                             "rebuild the LR schedule over the remaining fine-tune horizon "
                             "(max_steps - resume step) instead of continuing the old one.")
    parser.add_argument("--use-struct-loss-v2", type=_str_to_bool, default=False,
                        help="Use fixed structural losses: EdgeGradientLoss (gradient-profile "
                             "matching vs GT image, no per-image norm / no binary canny match) "
                             "and SkeletonLossV2 (recall-only, no off-skeleton penalization).")
    parser.add_argument("--struct-max-t", type=int, default=0,
                        help="Only apply pixel structural losses on noise steps t<=struct-max-t "
                             "(0 = apply at all timesteps). High-noise x0 predictions are blurry "
                             "mush; forcing structure there drifts x0 off the VAE manifold.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="AdamW weight decay (sparse-condition training benefits from 0.01-0.05)")
    parser.add_argument("--global-batch-size", type=int, default=16) # default small batch for laptop GPU
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--sampler", type=str, choices=["random", "factor_balanced"],
                        default="random")
    parser.add_argument("--balance-char-alpha", type=float, default=0.5,
                        help="Tempered inverse character-frequency exponent.")
    parser.add_argument("--balance-callig-alpha", type=float, default=0.25,
                        help="Tempered inverse calligrapher-frequency exponent.")
    parser.add_argument("--use-ema", type=_str_to_bool, default=False,
                        help="Maintain and evaluate a full-model exponential moving average.")
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--ema-warmup", type=_str_to_bool, default=True,
                        help="Cap early EMA decay by update count to avoid random-init lag.")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--vae-path", type=str, default="pretrained_models/sd-vae-ft-ema", help="Local path to VAE weights")
    parser.add_argument("--use-lora", type=_str_to_bool, default=True, help="Use LoRA for fine-tuning DiT blocks")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=None,
                        help="LoRA alpha (scaling = alpha/r). Default: same as r (scaling=1).")
    parser.add_argument("--lora-target", type=str, choices=["all", "attn", "mlp"], default="all",
                        help="Which linear layers to inject LoRA into: all (qkv+proj+fc1+fc2), "
                             "attn (qkv+proj), or mlp (fc1+fc2).")
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
    parser.add_argument("--ckpt-keep", type=int, default=0,
                        help="Keep only the N most recent checkpoints (0 = keep all). "
                             "Old checkpoints and their eval_* dirs are pruned after each save.")
    parser.add_argument("--preload", type=_str_to_bool, default=False,
                        help="Preload latents/canny/skeleton (and GT image when REPA is on) "
                             "into RAM at startup for zero-disk-IO training.")
    parser.add_argument("--preload-workers", type=int, default=16,
                        help="Parallel PNG-decode workers used by preload.")
    parser.add_argument("--auto-eval", type=_str_to_bool, default=False,
                        help="Run in-memory eval (MSE/SSIM on N test samples) after each checkpoint save.")
    parser.add_argument("--eval-csv", type=str, default="test.csv",
                        help="CSV for auto-eval (only used when --auto-eval is true).")
    parser.add_argument("--eval-n", type=int, default=100,
                        help="Number of test samples for auto-eval (free-sampling).")
    parser.add_argument("--eval-steps", type=int, default=50,
                        help="DDIM steps for free-sampling auto-eval.")
    parser.add_argument("--eval-cfg", type=float, default=4.0,
                        help="CFG scale for free-sampling auto-eval.")
    parser.add_argument("--eval-seed", type=int, default=0,
                        help="Seed for free-sampling auto-eval noise.")
    parser.add_argument("--eval-batch", type=int, default=16,
                        help="Sampling batch for free-sampling auto-eval.")
    parser.add_argument("--show5-csv", type=str, default=None,
                        help="固定跨书体展示样本 CSV(如 eval5)。设后每次都采样这 N 个(不算指标, "
                             "仅生成 eval_latest.png/eval_samples 展示, 与海报 GT 行同一批保证对照)。")
    parser.add_argument("--w-canny", type=float, default=0.05, help="Weight for canny structural loss")
    parser.add_argument("--w-skel", type=float, default=0.05, help="Weight for skeleton structural loss")
    parser.add_argument("--w-skel-head", type=float, default=0.0,
                        help="Weight for latent skel_head aux supervision (train-only guide; "
                             "inference uses pure ID conditions). 0=disabled. ")
    parser.add_argument("--w-glyph-cond", type=_str_to_bool, default=False,
                        help="Enable 甲2 standard-glyph token-add conditioning (use_glyph_cond).")
    parser.add_argument("--glyph-scale-init", type=float, default=0.4,
                        help="Initial glyph_scale (standard-glyph token-add strength).")
    parser.add_argument("--glyph-init-mix", type=float, default=0.0,
                        help="HYBRID 初始点 alpha∈[0,1]: xT=alpha*randn+(1-alpha)*std字形latent。"
                             "0=纯噪声(现状); (0,1)=混合; 默认 0 保持当前行为, 收敛后按需设 e.g.0.6。"
                             "见 HYBRID_INIT_PLAN.md。")
    parser.add_argument("--w-std-mid", type=float, default=0.0,
                        help="MIDSTEP_STD 权重: 在中间噪声水平 sqrt(alpha_cumprod)∈[alo,ahi] 时,"
                             "额外监督 模型预测 clean latent 逼近标准字形 latent g, 让字形中段锚定。"
                             "需 w-glyph-cond 开启。权重明显小于主 loss(如 0.1~0.5), 防抹掉风格。0=关。")
    parser.add_argument("--std-mid-alo", type=float, default=0.35,
                        help="中间噪声带下界(sqrt_alpha_cumprod), 默认 0.35。")
    parser.add_argument("--std-mid-ahi", type=float, default=0.75,
                        help="中间噪声带上界(sqrt_alpha_cumprod), 默认 0.75。")
    parser.add_argument("--struct-subset", type=int, default=32,
                        help="Random per-step subset of the batch used for pixel canny/skel "
                             "loss decode (infra optimization: bounds VAE-decode VRAM; "
                             "0 = full batch).")
    parser.add_argument("--struct-decode-bf16", type=_str_to_bool, default=False,
                        help="Run the differentiable VAE decode for pixel structural losses "
                             "under bf16 autocast. bf16 shares fp32's exponent range so the "
                             "SD-VAE decoder cannot overflow (unlike fp16); the coarser "
                             "mantissa only adds mild noise to an auxiliary structural loss. "
                             "Output is cast back to fp32 before the losses.")
    parser.add_argument("--struct-decode-scale", type=float, default=1.0,
                        help="Downscale the decoded-image resolution for pixel structural "
                             "losses by this factor (feed a proportionally smaller latent "
                             "into the fully-convolutional decoder, e.g. 0.5 -> 128x128, "
                             "~4x cheaper decode). GT canny/skel maps are resized to match. "
                             "1.0 = full 256x256 decode.")
    parser.add_argument("--w-latent-canny", type=float, default=0.0,
                        help="Weight for decoder-free Canny-weighted latent gradient loss.")
    parser.add_argument("--w-latent-skel", type=float, default=0.0,
                        help="Weight for decoder-free frozen-probe skeleton loss.")
    parser.add_argument("--latent-structure-probe", type=str, default=None,
                        help="Checkpoint from train_latent_structure_probe.py (required for latent skeleton loss).")
    parser.add_argument("--latent-struct-max-t", type=int, default=500,
                        help="Apply latent structural losses only at diffusion timesteps <= this value.")
    parser.add_argument("--w-repa", type=float, default=1.0, help="Weight for Representation Alignment (REPA) Loss")
    parser.add_argument("--repa-teacher-ckpt", type=str, default="",
                        help="Local path to DINOv2 teacher weights (ModelScope safetensors). "
                             "Empty = auto-detect pretrained_models/dinov2_vits14_pretrain.safetensors or $DINO_WEIGHTS.")
    parser.add_argument("--use-canny", type=_str_to_bool, default=False,
                        help="Enable Canny structural loss (requires canny maps in dataset/canny).")
    parser.add_argument("--use-skel", type=_str_to_bool, default=False,
                        help="Enable Skeleton structural loss (requires skeleton maps in dataset/skeleton).")
    parser.add_argument("--latent-shards-dir", type=str, default=None,
                        help="Dir of pre-built latent shards (shard_XXXXX.npz). If set, training reads "
                             "pre-encoded VAE latents instead of on-the-fly VAE encode.")
    parser.add_argument("--img-root", type=str, default="final_imgs_256",
                        help="Root dir of 256x256 gt images (used with latent-cached training for gt-losses).")
    parser.add_argument("--canny-root", type=str, default="final_canny",
                        help="Directory of precomputed canny images (img_id.png)")
    parser.add_argument("--skel-root", type=str, default="final_skeleton",
                        help="Directory of precomputed skeleton images (img_id.png)")
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
            action.default = _coerce(config_defaults[action.dest], action.default, action.type)
            action.required = False

    args = parser.parse_args()
    main(args)
