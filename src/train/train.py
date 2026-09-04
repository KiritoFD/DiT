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

from src.model import DiT_2Cond_models
from src.loss import create_diffusion_or_flow, flow_kwargs_from
from diffusers.models import AutoencoderKL
from src.utils import find_model

from src.utils import MCCDDataset
from src.utils import MCCDLatentDataset
from src.loss import EdgeGradientLoss, SkeletonLoss, REPALoss, StructDecoder, LatentStructLoss
from torch.utils.checkpoint import checkpoint as grad_ckpt
from src.utils import DistributedFactorBalancedSampler
from src.utils import LatentStructureLoss, LatentStructureProbe

# In-process GPU eval (bf16 sampling → VAE decode → save PNGs).
# Metrics computed by eval_ctrl_metrics_daemon.py (CPU, separate process).
try:
    from src.eval.in_process_eval import (
        prepare_eval_cache, run_gpu_eval, prepare_small_cache, run_show5,
    )
    _HAS_IN_PROCESS_EVAL = True
except ImportError:
    _HAS_IN_PROCESS_EVAL = False

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

    # Latent spatial size: default f8 (sd-vae), but supports f4 (kl-f4) via vae_downscale arg.
    vae_downscale = getattr(args, 'vae_downscale', 8)
    assert args.image_size % vae_downscale == 0, f"image_size {args.image_size} not divisible by vae_downscale {vae_downscale}"
    latent_size = args.image_size // vae_downscale
    
    # 注：历史上曾有 cond_mode=3cond（callig + script + char 三条件，模型 DiT_3Cond）。
    # 2026-08-31 清理时删除 —— 三条件模型已废弃，当前只保留 2cond（callig + char）。
    cond_mode = args.cond_mode
    if cond_mode == "3cond":
        raise ValueError(
            "cond_mode=3cond 已废弃：DiT_3Cond 模型类于 2026-08-31 清理时删除。"
            "当前只支持 cond_mode=2cond（书家 + 字）。若确需三条件，"
            "需从 git 历史恢复 src/model/dit.py 中的 DiT_3Cond。")
    else:
        if args.model not in DiT_2Cond_models:
            raise ValueError(f"cond_mode=2cond but model '{args.model}' is not a 2Cond model. "
                             f"Use one of {list(DiT_2Cond_models.keys())}.")
        # flow matching 没有方差头：learn_sigma=False 时 out_channels == in_channels。
        # 若为 True（旧默认），final_layer 会多输出 C 个通道，而
        # FlowMatching.training_losses 只取前 C 个 —— 后 C 个通道零初始化且
        # **永远收不到梯度**，白白浪费参数并污染 out_channels 语义。
        _diffusion_type = str(getattr(args, 'diffusion_type', 'ddpm')).lower()
        _ls = getattr(args, 'learn_sigma', None)
        _learn_sigma = bool(_ls) if _ls is not None else \
            _diffusion_type not in ('flow', 'flow_matching', 'fm')
        # 0/1 -> bool（argparse 用 int 以便 config JSON 里写 0/1）
        _qk_norm = bool(getattr(args, 'qk_norm', True))
        _rope = bool(getattr(args, 'rope', True))
        _heun_batch = bool(getattr(args, 'heun_batch', True))

        # IDS 组件码本: 构建 char_id -> char 映射
        _ids_char_id_to_char = None
        if getattr(args, 'use_ids_char_embedder', False):
            _ids_csv = getattr(args, 'ids_char_map_csv', None)
            if _ids_csv and os.path.isfile(_ids_csv):
                from src.model.ids_embedder import build_char_id_map_from_csv
                _ids_char_id_to_char = build_char_id_map_from_csv(_ids_csv)
                logger.info(f"[ids] loaded char_id->char map from {_ids_csv}: "
                            f"{len(_ids_char_id_to_char)} entries")
            else:
                logger.warning(f"[ids] ids_char_map_csv not found ({_ids_csv!r}), "
                               f"assuming char_id == Unicode codepoint")

        model = DiT_2Cond_models[args.model](
            input_size=latent_size,
            num_calligraphers=args.num_calligraphers,
            num_characters=args.num_characters,
            use_checkpoint=args.use_checkpoint,
            learn_sigma=_learn_sigma,
            condition_fusion=args.condition_fusion,
            callig_embed_dim=args.callig_embed_dim,
            char_embed_dim=args.char_embed_dim,
            cond_drop_all_prob=args.cond_drop_all_prob,
            cond_drop_one_prob=args.cond_drop_one_prob,
            cond_drop_which_glyph_prob=getattr(args, 'cond_drop_which_glyph_prob', 0.5),
            skel_head_enabled=getattr(args, 'w_skel_head', 0) > 0,
            use_glyph_cond=getattr(args, 'w_glyph_cond', 0) > 0,
            glyph_scale_init=getattr(args, 'glyph_scale_init', 0.4),
            in_channels=getattr(args, 'latent_channels', 4),
            char_proj_mode=getattr(args, 'char_proj_mode', 'full'),
            freeze_char_table=getattr(args, 'freeze_char_table', False),
            # ---- IDS 组件码本字嵌入 ----
            use_ids_char_embedder=getattr(args, 'use_ids_char_embedder', False),
            ids_file=getattr(args, 'ids_file', None),
            char_id_to_char=_ids_char_id_to_char,
            # ---- 标准字形 DINO 字嵌入 (冻结查表) ----
            use_std_dino_char_embedder=getattr(args, 'use_std_dino_char_embedder', False),
            std_dino_table_path=getattr(args, 'std_dino_table_path', None),
            # ---- 现代化骨干开关 ----
            norm_type=getattr(args, 'norm_type', 'rms'),
            mlp_type=getattr(args, 'mlp_type', 'swiglu'),
            qk_norm=_qk_norm,
            rope=_rope,
            rope_theta=getattr(args, 'rope_theta', 100.0),
            attn_impl=getattr(args, 'attn_impl', 'sdpa'),
        )
        logger.info(f"Building 2-Cond model: {args.model} "
                    f"(learn_sigma={_learn_sigma}, diffusion_type={_diffusion_type}, "
                    f"arch={getattr(args, 'norm_type', 'rms')}/"
                    f"{getattr(args, 'mlp_type', 'swiglu')}/"
                    f"qknorm={_qk_norm}/rope={_rope}, "
                    f"attn={getattr(args, 'attn_impl', 'sdpa')})")
        logger.info(f"Building 2-Cond model: {args.model} "
                    f"(callig={args.num_calligraphers}, glyph/char={args.num_characters}, "
                    f"fusion={args.condition_fusion}, dims={args.callig_embed_dim}/"
                    f"{args.char_embed_dim}, dropout=all:{args.cond_drop_all_prob}, "
                    f"one:{args.cond_drop_one_prob}, skel_head={getattr(args, 'w_skel_head', 0) > 0}, "
                    f"glyph_cond={getattr(args, 'w_glyph_cond', 0) > 0}, glyph_scale_init={getattr(args, 'glyph_scale_init', 0.4)}, "
                    f"char_proj_mode={getattr(args, 'char_proj_mode', 'full')}, "
                    f"freeze_char_table={getattr(args, 'freeze_char_table', False)})")

    # ── DINO glyph-embedding init for y_char_embedder ───────────────────────
    # glyph_id = script_id * 7026 + character_id (每 script 7026 个字符, 见
    # tools/remote_sync/_add_glyph_col.py). DINO vocab 是对"字*书体"(glyph) 取平均的:
    # 同一 glyph 的所有书写样本的 CLS token 平均后 L2 归一化, 维度必须 == char_embed_dim
    # (768), 之后 char_proj 直接 LayerNorm(768)->Linear(768,H) 投影, 不再经过中间 256 层。
    #
    # ⚠ 当 use_ids_char_embedder=True 时跳过 DINO 初始化:
    # IDSCharEmbedder 用部件嵌入池化, 不需要 DINO 初始化。
    _use_ids = getattr(args, 'use_ids_char_embedder', False)
    _use_std_dino = getattr(args, 'use_std_dino_char_embedder', False)
    _dino_emb_path = getattr(args, "char_dino_embeddings", None)
    _dino_idx_path = getattr(args, "char_dino_index", None)
    if _use_ids:
        logger.info(f"[ids] using IDSCharEmbedder, skipping DINO init. "
                    f"coverage={model.y_char_embedder.coverage:.2%}, "
                    f"num_components={model.y_char_embedder.num_components}")
    elif _use_std_dino:
        logger.info(f"[std-dino] using StdDinoCharEmbedder (frozen standard-glyph DINO table), "
                    f"skipping DINO init. table={tuple(model.y_char_embedder.char_table.shape)}")
    elif _dino_emb_path and _dino_idx_path and os.path.isfile(_dino_emb_path) and os.path.isfile(_dino_idx_path):
        _NUM_CH = 7026  # 与 _add_glyph_col.py 的 glyph_id 编码一致
        _emb = np.load(_dino_emb_path)
        with open(_dino_idx_path, "r", encoding="utf-8") as f:
            _idx_data = json.load(f)
        _glyphs = _idx_data.get("glyphs", _idx_data)
        _table = model.y_char_embedder.embedding_table.weight
        if _emb.ndim != 2 or _emb.shape[1] != _table.shape[1]:
            logger.warning(f"[dino-init] shape mismatch: dino={_emb.shape} vs "
                           f"char_embed_dim={_table.shape[1]} — skipping DINO init.")
        else:
            _emb = _emb.astype(np.float32)

            # 未知行的填充向量必须在 centering **之前**算：centering 后每个 script
            # 内部均值为 0，全体均值也趋近 0（实测 norm 仅 0.023），是个退化向量。
            # 用未中心化的 DINO 均值并 L2 归一化 -> norm=1.0，与已知行同量级，
            # 与已知行的平均余弦 +0.315（已知行两两之间平均 +0.115），
            # 即"一个居中的典型字形"，比 N(0,0.02) 随机噪声(余弦≈0, 等同于随机字)好得多。
            _fill_vec = _emb.mean(0)
            _fill_vec = _fill_vec / max(float(np.linalg.norm(_fill_vec)), 1e-12)

            # ---- (1) per-script centering（可选，实测有效）--------------------
            # 冻结 DINO 表被"书体"主导：有效秩只有 34.1/384（PC1 占 26.3% 能量），
            # 83% 的最近邻是同一书体，跨书体字符检索 top-1 仅 1.9%。
            # 而书体信息本该由 y_callig_embedder 提供，char 分支里的书体分量
            # 既是冗余也是噪声。减去每个书体的均值后：
            #   有效秩 34.1 → 57.0，ret@1 1.9% → 2.6%，ret@5 4.2% → 6.8%，
            #   书体泄漏 83.0% → 77.9%。
            if getattr(args, 'dino_per_script_center', 0):
                _sids = np.array([int(g[0]) for g in _glyphs])
                for _s in np.unique(_sids):
                    _m = _sids == _s
                    if _m.sum() > 1:
                        _emb[_m] -= _emb[_m].mean(0, keepdims=True)
                _n = np.linalg.norm(_emb, axis=1, keepdims=True)
                _emb = _emb / np.maximum(_n, 1e-12)
                logger.info(f"[dino-init] per-script centering applied "
                            f"({len(np.unique(_sids))} scripts) + L2 renormalized")

            _loaded = 0
            _dropped = 0
            _filled_rows = []
            with torch.no_grad():
                for _gi, (_sid, _cid) in enumerate(_glyphs):
                    _gid = int(_sid) * _NUM_CH + int(_cid)
                    if 0 <= _gid < _table.shape[0] and _gi < _emb.shape[0]:
                        _table[_gid].copy_(torch.from_numpy(_emb[_gi]).float())
                        _loaded += 1
                        _filled_rows.append(_gid)
                    else:
                        _dropped += 1

                # ---- (2) 未命中行：用 DINO 均值填充，而不是留随机噪声 ----------
                # y_char_embedder 有 num_characters=35130 行，而 DINO 只覆盖 20468 个
                # glyph。未命中的行停留在 nn.Embedding 默认 N(0, 0.02) 且被冻结，
                # 对模型来说就是一个"随机字符"。
                # 更糟：char_proj='ln_only' 时 LayerNorm 逐样本归一化，把
                # "已知行范数=1.0" 和 "未知行范数≈0.39" 这个唯一可辨的线索也抹掉了
                # —— 模型在数值上无法区分。用 DINO 均值填充至少给出一个
                # "平均字形"的合理先验（eval_unseen 上有 6.6% 的 glyph 落在这里）。
                _fill = getattr(args, 'dino_fill_unknown', 1)
                if _fill and _filled_rows:
                    # 注意：embedding_table 有 num_classes + 1 行，最后一行是
                    # LabelEmbedder 的 CFG null token，**绝不能覆盖**（否则 CFG 失效）。
                    _n_classes = model.y_char_embedder.num_classes
                    _mean = torch.from_numpy(_fill_vec).to(_table.device, _table.dtype)
                    _known = set(_filled_rows)
                    _n_unknown = _n_classes - len(_known)
                    if _n_unknown > 0:
                        # 只填充 [0, num_classes) 区间内未被 DINO 命中的行
                        _unknown_rows = [r for r in range(_n_classes) if r not in _known]
                        _rows_t = torch.as_tensor(_unknown_rows, device=_table.device)
                        _table.index_copy_(0, _rows_t,
                                           _mean[None].expand(len(_unknown_rows), -1))
                        logger.info(f"[dino-init] filled {_n_unknown} unknown rows "
                                    f"(of {_n_classes} classes) with the L2-normalized DINO "
                                    f"mean vector (norm=1.0, was: frozen N(0,0.02) noise with "
                                    f"~0 cosine to all real glyphs); "
                                    f"CFG null token (row {_n_classes}) untouched.")

            logger.info(f"[dino-init] injected {_loaded} glyph embeddings into "
                        f"y_char_embedder ({_emb.shape[0]} in vocab, {_dropped} out-of-range), "
                        f"table={tuple(_table.shape)}, L2-normalized DINO (glyph-averaged), "
                        f"char_proj_mode={getattr(args, 'char_proj_mode', 'full')}, "
                        f"freeze_char_table={getattr(args, 'freeze_char_table', False)}.")
    else:
        logger.warning(f"[dino-init] char_dino_embeddings/index not found "
                       f"({_dino_emb_path!r}, {_dino_idx_path!r}) — y_char_embedder stays random init.")

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

    # 注：LoRA 支持（inject_lora / upgrade_lora_rank / extract_full_inference）
    # 已于 2026-08-31 随 src/model/lora.py 一并删除 —— 当前所有配置均为
    # use_lora=false，ControlNet 训练用的是「冻结主干 + 只训 ctrl 分支」，
    # 不需要 LoRA。若配置里仍带 use_lora=true，这里显式报错而不是静默忽略。
    if getattr(args, 'use_lora', False):
        raise ValueError(
            "use_lora=true 已不支持：src/model/lora.py 已于 2026-08-31 删除。"
            "请改用 use_lora=false（全参数训练）或 --pretrained 冻结主干模式。"
            "若确实需要 LoRA，需从 git 历史恢复 src/model/lora.py。")

    # 3) full resume works for both LoRA and full-from-scratch checkpoints.
    if getattr(args, 'resume_full', None) is not None:
        import torch as _torch
        _rf = _torch.load(args.resume_full, map_location="cpu", weights_only=False)
        _resume_full_ckpt = _rf
        _sd = _rf.get("delta", _rf.get("model", _rf))
        missing, unexpected = model.load_state_dict(_sd, strict=False)
        logger.info(f"[resume-full] Loaded weights from {args.resume_full} "
                    f"(missing={len(missing)}, unexpected={len(unexpected)}).")

    # ---- freeze / trainable policy --------------------------------------------
    # Two regimes:
    #   1) pretrained body: freeze the pretrained transformer body,
    #      train only the *new* condition head + adaLN/final_layer.
    #      adaLN is reset to std=0.02 by `reset_cond_head`, so it MUST be trainable
    #      (`train_cond_head=true`), otherwise the model is stuck on random modulation.
    #   2) from-scratch (pretrained=None): keep all params trainable.
    _has_pretrained = args.pretrained is not None
    if _has_pretrained:
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

        # 上面的白名单用 `'y_char_embedder' in name` 匹配，会**顺带把已冻结的
        # 字符表重新解冻**（embedding_table.weight 也在 y_char_embedder 名下）。
        # 这里显式恢复冻结，只保留 LabelEmbedder.null_embed 可训练。
        if getattr(model, '_char_table_frozen', False) and hasattr(model, 'y_char_embedder'):
            _ye = model.y_char_embedder
            # IDSCharEmbedder 用 comp_embedding 而非 embedding_table
            if hasattr(_ye, 'comp_embedding'):
                _ye.comp_embedding.weight.requires_grad_(False)
                _frozen_params = _ye.comp_embedding.weight.numel()
            else:
                _ye.embedding_table.weight.requires_grad_(False)
                _frozen_params = _ye.embedding_table.weight.numel()
            if _ye.null_embed is not None:
                _ye.null_embed.requires_grad_(True)
            logger.info("[freeze-char-table] kept y_char_embedder frozen "
                        f"({_frozen_params:,} params); null_embed stays trainable.")

    # ── 诊断开关：只训练字 embedding（冻结主干）────────────────────────────
    # 目的：验证「base SSIM 卡在 0.50 的瓶颈是不是 char 条件」。
    # 除 char embedding + char_proj 外全部冻结，因此指标的任何变化都可以
    # 直接归因到字条件，而不会被「主干也顺便多训了一会儿」污染。
    # 背景：DINO glyph 表实测判别力极弱——库内检索 top-1 84%，换成同字
    # 不同书家的库外检索立刻掉到 4%（见 docs/system/14_glyph_condition_probe.md），
    # 说明 CLS 特征编码的是「这张图长什么样」而非「这个字是什么字」。
    if getattr(args, 'train_only_char_embed', False):
        requires_grad(model, False)
        _n = 0
        for _name, _p in model.named_parameters():
            if 'y_char_embedder.embedding_table' in _name or 'char_proj' in _name:
                _p.requires_grad = True
                _n += _p.numel()
        # CFG null token 必须保持可训练，否则 uncond 分支失效、eval 指标失真
        _ye = model.y_char_embedder
        if getattr(_ye, 'null_embed', None) is not None:
            _ye.null_embed.requires_grad_(True)
            _n += _ye.null_embed.numel()
        # IDSCharEmbedder 还有 fallback_embed
        if getattr(_ye, 'fallback_embed', None) is not None:
            _ye.fallback_embed.requires_grad_(True)
            _n += _ye.fallback_embed.numel()
        logger.info(f"[train-only-char-embed] backbone frozen; trainable={_n:,} "
                    f"(char embedder + char_proj + null/fallback). "
                    f"NOTE: pair with freeze_char_table=false, else the table stays frozen "
                    f"(_char_table_frozen={getattr(model, '_char_table_frozen', False)}).")

    # report trainable counts
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info(f"Trainable Parameters: {trainable_params:,}")
    logger.info(f"Frozen Parameters: {frozen_params:,} (trainable ratio: {trainable_params/(trainable_params+frozen_params)*100:.2f}%)")

    model = model.to(device)
    # torch.compile (torch>=2.0, cu121 env): 在 DDP 之前编译整个模型。
    # 默认 mode="default"；250 steps 预热区间的首步会花数十秒编译，之后每步走
    # inductor 缓存的 kernel。EMA 模型是 eval-only 深拷贝，不编译（省编译时间/
    # 显存），权重同步走普通张量拷贝，与编译无关。
    if getattr(args, "compile", False):
        _compile_mode = getattr(args, "compile_mode", "default")
        if float(torch.__version__[:3]) < 2.0:
            logger.warning("[compile] torch %s 不支持 torch.compile，忽略 --compile",
                           torch.__version__)
        else:
            logger.info(f"[compile] torch.compile(mode={_compile_mode}) 注入（DDP 之前）...")
            model = torch.compile(model, mode=_compile_mode)
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
    # flow 的 t 分布 / 求解器 / shift 由 config 指定，训练与 eval 共用同一份
    # （通过 flow_kwargs_from 抽取），避免两侧静默不一致。
    _flow_kw = flow_kwargs_from(args)
    diffusion = create_diffusion_or_flow(timestep_respacing="",
                                         diffusion_type=getattr(args, 'diffusion_type', 'ddpm'),
                                         **_flow_kw)
    _is_flow = getattr(diffusion, 'is_flow', False)
    if _is_flow:
        logger.info(f"[flow] {diffusion.describe()}")
    if _is_flow:
        # Flow-Matching mode: disable DDPM-timestep-dependent auxiliaries.
        # Flow trains on t in [0,1] with a velocity target; DDPM-gated structural/
        # latent auxiliaries would be semantically wrong. Hard-disable them (with a
        # log) so a flow run never silently mixes incompatible objectives.
        # NOTE (2026-09-05): w_repa 从禁用列表移除 —— REPA 对齐主模型 block 特征到
        # DINO (与 t 无关, 只需 GT 图 + teacher), 已在 train_repa/train_controlnet 的
        # flow 下验证有效 (v8c/v8e SOTA 0.767/0.776)。flow 禁用它是过度限制。
        _flow_disabled = []
        for _attr in ('use_canny', 'use_skel', 'w_skel_head', 'w_std_mid',
                      'w_latent_skel', 'w_latent_canny'):
            if getattr(args, _attr, 0):
                setattr(args, _attr, 0 if not isinstance(getattr(args, _attr, 0), bool) else False)
                _flow_disabled.append(_attr)
        if _flow_disabled:
            logger.info(f"[flow] disabled DDPM-only auxiliaries: {', '.join(_flow_disabled)}")
        logger.info(f"[flow] Flow-Matching enabled (velocity target, Euler ODE sampling, t in [0,1])")
    _vae_ds = getattr(args, 'vae_downscale', 8)
    _vae_lc = getattr(args, 'latent_channels', 4)
    _vae_ic = getattr(args, 'vae_in_channels', 3)
    _vae_oc = getattr(args, 'vae_out_channels', 3)
    _vae_sf = getattr(args, 'vae_scaling_factor', 0.18215)
    class MockVAE(torch.nn.Module):
        def __init__(self, device):
            super().__init__()
            self.device = device
        def encode(self, x):
            class Dist:
                def sample(self):
                    return torch.randn(x.shape[0], _vae_lc, x.shape[2]//_vae_ds, x.shape[3]//_vae_ds, device=x.device)
            class Output:
                latent_dist = Dist()
            return Output()
        def decode(self, z):
            class Output:
                sample = torch.randn(z.shape[0], _vae_oc, z.shape[2]*_vae_ds, z.shape[3]*_vae_ds, device=z.device)
            return Output()

    try:
        _need_vae = (not bool(getattr(args, "latent_shards_dir", None))) or args.w_repa > 0 or args.use_canny
        if _need_vae:
            if getattr(args, 'vae_path', None) is not None and os.path.exists(args.vae_path):
                logger.info(f"Loading VAE from local path: {args.vae_path}")
                vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
            else:
                vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
            requires_grad(vae, False)
        else:
            vae = MockVAE(device)
            logger.info("[infra] VAE skipped (latent-only mode, no pixel struct/REPA) -> saves ~500MB VRAM")
    except Exception as e:
        logger.warning(f"Failed to load AutoencoderKL due to network/path error: {e}")
        logger.warning("Using MockVAE (random latents) for testing purposes!")
        vae = MockVAE(device)
    #     （不做逐图归一化、不做二值 Canny 拟合，保留灰度过渡/抗锯齿）
    #   - SkeletonLoss: 只做正向牵引（recall-only），骨架上必须含墨，
    #     绝不惩罚骨架以外的墨水（笔画粗细/飞白交给扩散损失决定）
    canny_loss_fn = EdgeGradientLoss().to(device)
    skel_loss_fn = SkeletonLoss().to(device)
    logger.info("[struct] EdgeGradientLoss (gradient-profile, no per-image norm) "
                "+ SkeletonLoss (recall-only, no off-skel penalization)")

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

    # ---- LatentStructLoss: 冻结 StructDecoder (latent→skel/canny) + BCE ----
    latent_struct_loss_fn = None
    if getattr(args, 'w_latent_struct_skel', 0) > 0 or getattr(args, 'w_latent_struct_canny', 0) > 0:
        _lsd = getattr(args, 'latent_struct_decoder', '')
        if not _lsd:
            raise ValueError("w_latent_struct_skel/canny > 0 requires --latent-struct-decoder (path to skel_best.pt/canny_best.pt)")
        if getattr(args, 'w_latent_struct_skel', 0) > 0:
            latent_struct_skel_fn = LatentStructLoss(
                _lsd.replace("canny", "skel") if "canny" in _lsd else _lsd,
                decoder_type="skel",
                pos_weight=float(getattr(args, 'latent_struct_pos_weight', 15.0)),
                use_checkpoint=True).to(device)
            logger.info(f"[latent-struct-decoder] skel decoder loaded from {_lsd}")
        else:
            latent_struct_skel_fn = None
        if getattr(args, 'w_latent_struct_canny', 0) > 0:
            latent_struct_canny_fn = LatentStructLoss(
                _lsd.replace("skel", "canny") if "skel" in _lsd else _lsd,
                decoder_type="canny",
                pos_weight=float(getattr(args, 'latent_struct_pos_weight', 8.0)),
                use_checkpoint=True).to(device)
            logger.info(f"[latent-struct-decoder] canny decoder loaded from {_lsd}")
        else:
            latent_struct_canny_fn = None
        latent_struct_loss_fn = True  # marker
        logger.info(f"[latent-struct-decoder] enabled: "
                    f"skel_w={getattr(args, 'w_latent_struct_skel', 0)}, "
                    f"canny_w={getattr(args, 'w_latent_struct_canny', 0)}, "
                    f"max_t={getattr(args, 'latent_struct_max_t', 500)}")
    else:
        latent_struct_skel_fn = None
        latent_struct_canny_fn = None
    
    repa_loss_fn = None
    if args.w_repa > 0:
        teacher_ckpt = getattr(args, "repa_teacher_ckpt", "") or None
        # Student hidden dim = transformer hidden_size, derived from the patch
        # embedder projection (S/2 -> 384, B/2 -> 768, ...).
        try:
            student_hidden_size = int(model.x_embedder.proj.out_features)
        except Exception:
            student_hidden_size = 384
        logger.info(f"Initializing REPA Loss (Teacher: dinov2_vits14, ckpt={teacher_ckpt or 'auto'}, Student Dim: {student_hidden_size})")
        repa_loss_fn = REPALoss(student_dim=student_hidden_size, teacher_backbone="dinov2_vits14",
                                teacher_ckpt=teacher_ckpt).to(device)

    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    if repa_loss_fn is not None:
        trainable_params_list.extend([p for p in repa_loss_fn.proj.parameters() if p.requires_grad])

    _opt_name = getattr(args, "optimizer", "adamw")
    if _opt_name == "muon":
        # Muon: 矩阵权重 NS-正交化 (独立 muon_lr, 典型 0.02-0.05), 向量/embedding 走 AdamW (adamw_lr)
        try:
            from src.optim.muon import Muon as _Muon
        except Exception as e:
            raise RuntimeError(f"src.optim.muon import failed: {e}")
        _muon_lr = float(getattr(args, "muon_lr", 0.02))
        _adamw_lr = float(getattr(args, "lr", 3e-4))
        _adamw_wd = float(getattr(args, "weight_decay", 0.01))
        # REPA proj 是矩阵 dim=2 -> 自动进 muon 组, 会用它; 向量(embedding标量)进 adamw
        opt = _Muon(
            trainable_params_list, lr=_muon_lr, weight_decay=0.0,
            adamw_lr=_adamw_lr, adamw_weight_decay=_adamw_wd)
        logger.info(f"[optim] Muon: 矩阵组 lr={_muon_lr} (NS正交) / 向量+embedding组 AdamW lr={_adamw_lr} wd={_adamw_wd}")
    else:
        opt = torch.optim.AdamW(trainable_params_list, lr=args.lr, weight_decay=args.weight_decay)
        logger.info(f"[optim] AdamW lr={args.lr} wd={args.weight_decay}")

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
    need_canny_map = args.use_canny or args.w_latent_canny > 0 or getattr(args, 'w_latent_struct_canny', 0) > 0
    need_skel_map = args.use_skel or args.w_latent_skel > 0 or getattr(args, 'w_skel_head', 0) > 0 or getattr(args, 'w_latent_struct_skel', 0) > 0

    # Re-decide VAE need: if we're in latent-only mode (no pixel struct/REPA), use MockVAE.
    # VAE is only needed for on-the-fly encode (non-latent) or pixel structural losses.
    _need_vae_now = (not use_latent) or args.w_repa > 0 or args.use_canny
    if not _need_vae_now:
        vae = MockVAE(device)
        logger.info("[infra] VAE skipped (latent-only mode, no pixel struct/REPA) -> saves ~500MB VRAM")
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
                                    load_image=(args.w_repa > 0 or args.use_canny),
                                    num_preload_workers=int(getattr(args, 'preload_workers', 16)),
                                    structure_size=256,
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
        prefetch_factor=4 if args.num_workers > 0 else None,
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
    running_latent_struct_skel = 0
    running_latent_struct_canny = 0
    running_repa = 0
    running_x0lat = 0
    running_skel_head = 0
    running_std_mid = 0
    nan_steps = 0
    current_ema_decay = args.ema_decay
    start_time = time()

    # ---- 早停状态 (基于 CPU eval 的 eval_auto_*.json mse/ssim/skel_iou) ----
    early_stop_best = None      # 最佳 metric 值 (ssim 越大越好 / mse 越小越好)
    early_stop_stale = 0        # 连续未改善的 eval 次数
    early_stop_last_eval_step = -1
    early_stop_stopped = False
    _es_metric = getattr(args, 'early_stop_metric', 'ssim')
    _es_better = ((lambda a, b: a > b) if _es_metric in ('ssim', 'combo', 'ssim_lpips')
                  else (lambda a, b: a < b))
    _es_higher_better = _es_metric in ('ssim', 'combo', 'ssim_lpips')
    # min_delta：只有超过 best ± min_delta 才算"真改善"。
    # 默认阈值按各指标的经验噪声量级选取（ssim/skel_iou 都是 0~1 量级，
    # mse 的量级随数据分布变化，默认不设阈值，需要时再显式配置）。
    _es_delta = {
        'ssim': float(getattr(args, 'early_stop_min_delta', 0.002)),
        'skel_iou': float(getattr(args, 'early_stop_min_delta_iou', 0.005)),
        'lpips': float(getattr(args, 'early_stop_min_delta_lpips', 0.003)),
        'mse': float(getattr(args, 'early_stop_min_delta_mse', 0.0)),
    }
    logger.info(f"[early-stop] metric={_es_metric}, patience="
                f"{getattr(args, 'early_stop_patience', 5)}, min_delta={_es_delta}")
    # 多指标组合（双保险）: 各自追踪 best/stale, **全部** stale 才停。
    # 用 (key, higher_better) 描述，因此天然支持方向混合
    #   - 'combo'      : ssim↑ + skel_iou↑   (skel_iou 已被证实不敏感，不推荐)
    #   - 'ssim_lpips' : ssim↑ + lpips↓      (推荐：像素结构 + 感知距离互补)
    _ES_SPECS = {
        'combo': (('ssim', True), ('skel_iou', True)),
        'ssim_lpips': (('ssim', True), ('lpips', False)),
    }
    _es_spec = _ES_SPECS.get(_es_metric)
    _es_combo_best = {k: None for k, _ in (_es_spec or ())}
    _es_combo_stale = {k: 0 for k, _ in (_es_spec or ())}
    _es_check_every = int(getattr(args, 'early_stop_check_every', 0))
    if _es_check_every <= 0:
        _es_check_every = max(int(getattr(args, 'ckpt_every', 5000)) // 2, 1000)

    def _early_stop_check(force=False):
        """读 ckpt 目录最新 eval_auto json, 更新 best/stale; 达到 patience 返回 True 表示停。
        combo 模式: ssim 和 skel_iou 都要连续 stale >= patience 才停 (双保险)。"""
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
            k = d.get("skel_iou")
            lp = d.get("lpips")
            if _es_metric == 'ssim_lpips':
                # lpips 是"越低越好"，这里取负号统一成"越大越好"，
                # 从而复用下面 (key, higher_better=True) 的通用比较逻辑。
                val = (float(s), -float(lp)) if s is not None and lp is not None else None
            elif _es_metric == 'combo':
                val = (float(s), float(k)) if s is not None and k is not None else None
            elif _es_metric == 'ssim':
                val = float(s) if s is not None else None
            else:
                val = float(m) if m is not None else None
        except Exception:
            return False
        if val is None:
            return False

        if _es_spec is not None:
            # 双保险: 每个指标各自追踪新鲜度 (任一刚创新高则整体 stale 清零)
            # min_delta: 只有超过 best + min_delta 才算"真改善"，否则指标噪声
            # （ssim 在 256² 二值字形上对笔画粗细/亚像素位移极敏感）会不停
            # 重置 stale 计数器，让早停实际上由噪声驱动。
            #
            # 注意 val 里的 lpips 已取负号，故下面对 y_v 的显示要还原。
            improved = False
            for (key, _hi), v in zip(_es_spec, val):
                b = _es_combo_best[key]
                dlt = _es_delta.get(key, 0.0)
                if b is None or v > b + dlt:
                    _es_combo_best[key] = v
                    _es_combo_stale[key] = 0
                    improved = True
                else:
                    _es_combo_stale[key] += 1
            # 显示：把取过负号的还原成原值
            shown = ", ".join(
                f"{k}={(-v if k == 'lpips' else v):.4f}" for (k, _), v in zip(_es_spec, val))
            best_shown = ", ".join(
                f"best_{k}={(-_es_combo_best[k] if k == 'lpips' else _es_combo_best[k]):.4f}"
                for k, _ in _es_spec)
            if improved:
                early_stop_stale = 0
                logger.info(f"[early-stop] eval step {ev_step}: {_es_metric} {shown} "
                            f"-> NEW BEST ({best_shown})")
            else:
                early_stop_stale += 1
                logger.info(f"[early-stop] eval step {ev_step}: {_es_metric} {shown} "
                            f"({best_shown}, stale {early_stop_stale}/"
                            f"{args.early_stop_patience})")
                if early_stop_stale >= int(getattr(args, 'early_stop_patience', 5)):
                    logger.info(f"[early-stop] {_es_metric} no improvement for "
                                f"{early_stop_stale} evals; early stopping.")
                    return True
            return False

        _es_d = _es_delta.get(_es_metric, 0.0)
        if early_stop_best is None or (
                (val > early_stop_best + _es_d) if _es_higher_better else
                (val < early_stop_best - _es_d)):
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

    # ── In-process GPU eval setup ──────────────────────────────────────────────
    _eval_cache = None
    _eval_show5_cache = None
    _eval_seen5_cache = None
    if _HAS_IN_PROCESS_EVAL and getattr(args, 'auto_eval', False) and rank == 0:
        _eval_csv = getattr(args, 'eval_csv', None)
        if _eval_csv and os.path.exists(_eval_csv):
            _eval_n = int(getattr(args, 'eval_n', 100))
            _vae_ds = int(getattr(args, 'vae_downscale', 4))
            _vae_lc = int(getattr(args, 'latent_channels', 4))
            _vae_sf = float(getattr(args, 'vae_scaling_factor', 0.18215))
            _img_root = getattr(args, 'img_root', '') or getattr(args, 'data_dir', '') or ''
            # 训练侧若启用标准字形条件，eval 必须同步喂 g，否则条件域不匹配：
            # 模型训练时依赖 g，eval 却拿到全零 → 表现为「该条件无效」的假象。
            _use_glyph = bool(getattr(args, 'use_glyph_cond', False))
            if bool(getattr(args, 'w_glyph_cond', False)) and not _use_glyph:
                _use_glyph = True       # w_glyph_cond 是同一个条件的别名
            if _use_glyph:
                logger.info("[glyph-cond] eval 侧同步启用标准字形条件 g")
            _eval_cache = prepare_eval_cache(
                _eval_csv, _img_root, args.image_size, _eval_n,
                _vae_ds, _vae_lc, _vae_sf, use_glyph_cond=_use_glyph)
            _show5_csv = getattr(args, 'show5_csv', None)
            if _show5_csv and os.path.exists(_show5_csv):
                _eval_show5_cache = prepare_small_cache(
                    _show5_csv, _img_root, args.image_size, _vae_ds, _vae_lc)
            _seen5_csv = getattr(args, 'seen5_csv', None)
            if _seen5_csv and os.path.exists(_seen5_csv):
                _eval_seen5_cache = prepare_small_cache(
                    _seen5_csv, _img_root, args.image_size, _vae_ds, _vae_lc)
            logger.info(f"[auto-eval] cache ready: eval_n={_eval_n}, "
                        f"show5={'yes' if _eval_show5_cache else 'no'}, "
                        f"seen5={'yes' if _eval_seen5_cache else 'no'}")
        else:
            logger.warning(f"[auto-eval] eval_csv not found ({_eval_csv!r}); auto-eval disabled")

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
                    # Latent-cached training: latent pre-encoded (scaled by vae_scaling_factor).
                    x_latent = batch['latent'].to(device)
                    x = batch.get('image', None)
                    x = x.to(device) if x is not None else None
                    canny_gt = batch['canny'].to(device) if need_canny_map else None
                    skel_gt = batch['skeleton'].to(device) if need_skel_map else None
                else:
                    x = batch['image'].to(device)
                    canny_gt = batch['canny'].to(device)
                    skel_gt = batch['skeleton'].to(device)
                    # VAE encode stays in fp32 for numerical stability (VAE is sensitive to low precision).
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float32):
                        x_latent = vae.encode(x).latent_dist.sample().mul_(_vae_sf)
                        x_latent = x_latent.float()

                # 统一时间步采样: FlowMatching.sample_t -> t∈[0,1); GaussianDiffusion.sample_t -> t∈{0..T-1}。
                # 调用方绝不自己分支 (否则会重蹈 flow/randint 错配覆辙)。
                t = diffusion.sample_t(x_latent.shape[0], device)
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
                #
                # return_pred_xstart: flow 分支默认**不返回** pred_xstart（避免
                # autograd 图膨胀），但下面 w_std_mid / latent_skel / latent_canny /
                # latent_struct 都依赖它。若这里不请求，flow 模式下这些机制会
                # 因 `loss_dict.get("pred_xstart", None)` 恒为 None 而**静默失效**
                # —— 不报错、loss 正常下降、但从未生效。w_std_mid 正是「把去噪中
                # 段预测的 x0 拉向标准字形 latent」的预训练改进项，失效代价很大。
                # gaussian_diffusion 不接受该参数，故用 try/except 兼容。
                _need_x0 = (latent_struct_loss_fn is not None
                            or getattr(args, 'w_latent_skel', 0) > 0
                            or getattr(args, 'w_latent_canny', 0) > 0
                            or getattr(args, 'w_std_mid', 0.0) > 0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if _need_x0:
                        try:
                            loss_dict = diffusion.training_losses(
                                model, x_latent, t, model_kwargs,
                                return_pred_xstart=True)
                        except TypeError:
                            # gaussian_diffusion 不支持该参数，本身就会返回 pred_xstart
                            loss_dict = diffusion.training_losses(
                                model, x_latent, t, model_kwargs)
                    else:
                        loss_dict = diffusion.training_losses(
                            model, x_latent, t, model_kwargs)
                    loss_diff = loss_dict["loss"].mean()

                loss_canny = torch.tensor(0.0, device=device)
                loss_skel = torch.tensor(0.0, device=device)
                loss_repa = torch.tensor(0.0, device=device)
                loss_latent_canny = torch.tensor(0.0, device=device)
                loss_latent_skel = torch.tensor(0.0, device=device)
                loss_x0lat = torch.tensor(0.0, device=device)

                # === INFRA FIX: pred_xstart from training_losses carries the ENTIRE DiT
                # forward graph (every block's activations are kept alive because the
                # graph flows back through _predict_xstart_from_eps → model_output).
                # This is THE memory leak: the 256-token × 384-dim × 12-block activation
                # graph stays pinned until backward() runs, and struct decoders build
                # ADDITIONAL 256×256 activations on top of it. When t<=500 (~50% of steps)
                # the decoder runs → graph doubles; when t>500 the graph lingers but
                # decoders don't run → the cycling 20G→22G→14G pattern.
                #
                # FIX: extract pred_xstart WITH the graph only when we actually need it
                # for a differentiable struct loss (t<=max_t). For t>500 steps, detach
                # immediately so the full graph is freed at the next zero_grad. We also
                # break the reference in loss_dict so no stale graph survives the loop.
                _need_x0_grad = (latent_struct_loss_fn is not None
                                 or getattr(args, 'w_latent_skel', 0) > 0
                                 or getattr(args, 'w_latent_canny', 0) > 0
                                 or getattr(args, 'w_std_mid', 0.0) > 0)
                pred_xstart_latent = loss_dict.get("pred_xstart", None)
                if pred_xstart_latent is not None and not _need_x0_grad:
                    # No struct loss this run at all — drop the graph immediately.
                    pred_xstart_latent = pred_xstart_latent.detach()
                # Clear the heavy reference inside loss_dict so the dict can't keep
                # the graph alive after we exit this step.
                loss_dict.pop("pred_xstart", None)

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
                        x_sub = x if args.use_canny else None
                    elif _use_this_step:
                        pred_xstart_sub = pred_xstart_latent[_idx]
                        x0_pred = None
                        canny_gt_sub = canny_gt[_idx] if args.use_canny else None
                        skel_gt_sub = skel_gt[_idx] if args.use_skel else None
                        x_sub = x[_idx] if args.use_canny else None
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
                        _decode_in = pred_xstart_sub.float() / _vae_sf
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
                            if x_sub is not None:
                                x_sub = F.interpolate(x_sub, scale_factor=_dscale, mode="area")
                        # Structural losses computed in fp32 (outside autocast) for stability.
                        if args.use_canny:
                            # EdgeGradientLoss: 预测图 vs GT 图的梯度幅值场 L1 匹配
                            loss_canny = canny_loss_fn(x0_pred, x_sub)
                        if args.use_skel:
                            loss_skel = skel_loss_fn(x0_pred, skel_gt_sub)

                if latent_structure_loss_fn is not None and pred_xstart_latent is not None:
                    latent_structure_losses = latent_structure_loss_fn(
                        pred_xstart_latent, x_latent, t,
                        canny=canny_gt if need_canny_map else None,
                        skeleton=skel_gt if need_skel_map else None)
                    loss_latent_canny = latent_structure_losses["canny"]
                    loss_latent_skel = latent_structure_losses["skeleton"]

                # ---- LatentStructLoss: 冻结 StructDecoder (latent→skel/canny) + BCE ----
                loss_latent_struct_skel = torch.tensor(0.0, device=device)
                loss_latent_struct_canny = torch.tensor(0.0, device=device)
                if latent_struct_loss_fn is not None and pred_xstart_latent is not None:
                    _lst_max_t = int(getattr(args, 'latent_struct_max_t', 500))
                    _lst_mask = t <= _lst_max_t
                    if bool(_lst_mask.any()):
                        _p = pred_xstart_latent[_lst_mask].float()
                        if latent_struct_skel_fn is not None and skel_gt is not None:
                            _sk = skel_gt[_lst_mask].float()
                            loss_latent_struct_skel = latent_struct_skel_fn(_p, _sk)
                        if latent_struct_canny_fn is not None and canny_gt is not None:
                            _ca = canny_gt[_lst_mask].float()
                            loss_latent_struct_canny = latent_struct_canny_fn(_p, _ca)
                        # Free the 256×256 decoder intermediate tensors NOW (they're
                        # captured in the graph of the loss tensors, but the inputs
                        # _p/_sk/_ca and the mask are no longer needed). This lets the
                        # allocator compact before backward instead of holding two
                        # decoder graphs + the DiT graph simultaneously.
                        del _p, _sk, _ca, _lst_mask

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
                        + getattr(args, 'w_latent_struct_skel', 0) * _struct_scale * loss_latent_struct_skel
                        + getattr(args, 'w_latent_struct_canny', 0) * _struct_scale * loss_latent_struct_canny
                        + args.w_repa * loss_repa
                        + getattr(args, 'w_skel_head', 0) * loss_skel_head
                        + getattr(args, 'w_std_mid', 0.0) * loss_std_mid)

                opt.zero_grad(set_to_none=True)  # INFRA: set_to_none 释放梯度tensor, 比 zero_() 快且省内存
                # Capture scalar values into plain Python floats BEFORE we del the
                # tensors. This lets the autograd graph be freed immediately while the
                # running accumulators (pure floats) survive for logging.
                _v_loss = loss.item() if torch.isfinite(loss) else 0.0
                _v_diff = loss_diff.item()
                _v_canny = loss_canny.item()
                _v_skel = loss_skel.item()
                _v_lc = loss_latent_canny.item()
                _v_ls = loss_latent_skel.item()
                _v_lss = loss_latent_struct_skel.item() if isinstance(loss_latent_struct_skel, torch.Tensor) else 0.0
                _v_lsc = loss_latent_struct_canny.item() if isinstance(loss_latent_struct_canny, torch.Tensor) else 0.0
                _v_repa = loss_repa.item()
                _v_skelh = loss_skel_head.item()
                _v_stdmid = loss_std_mid.item()

                # NaN guard: skip the step if loss is not finite (e.g. a bad sample).
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params_list, max_norm=1.0)
                    opt.step()
                    if scheduler is not None:
                        scheduler.step()
                    if ema_model is not None:
                        if args.ema_warmup:
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
                            f"(diff={_v_diff:.4f}, canny={_v_canny:.4f}, "
                            f"skel={_v_skel:.4f}). Accumulated skips: {nan_steps}"
                        )
                # === INFRA: release the autograd graph every step, finite or not.
                # The graph built by training_losses (DiT forward + pred_xstart) and the
                # struct decoder graphs must be freed BEFORE the next forward, otherwise
                # peak = diff_graph + struct_graph simultaneously → the 22G cycling.
                del loss, loss_dict, loss_diff, pred_xstart_latent
                del loss_canny, loss_skel, loss_repa, loss_latent_canny, loss_latent_skel
                del loss_latent_struct_skel, loss_latent_struct_canny, loss_skel_head, loss_std_mid, loss_x0lat

                if _v_loss:
                    running_loss += _v_loss
                    running_diff += _v_diff
                    running_canny += _v_canny
                    running_skel += _v_skel
                    running_latent_canny += _v_lc
                    running_latent_skel += _v_ls
                    running_latent_struct_skel += _v_lss
                    running_latent_struct_canny += _v_lsc
                    running_repa += _v_repa
                    running_skel_head += _v_skelh
                    running_std_mid += _v_stdmid
                    running_x0lat += 0
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
                    avg_lss = torch.tensor(running_latent_struct_skel / divisor, device=device)
                    avg_lsc = torch.tensor(running_latent_struct_canny / divisor, device=device)
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
                        dist.all_reduce(avg_lss, op=dist.ReduceOp.SUM)
                        dist.all_reduce(avg_lsc, op=dist.ReduceOp.SUM)
                        dist.all_reduce(avg_r, op=dist.ReduceOp.SUM)
                        dist.all_reduce(avg_x0, op=dist.ReduceOp.SUM)
                        dist.all_reduce(avg_skel_h, op=dist.ReduceOp.SUM)
                        dist.all_reduce(avg_std_mid, op=dist.ReduceOp.SUM)
                        avg_l, avg_d = avg_l.item()/world_size, avg_d.item()/world_size
                        avg_c, avg_s = avg_c.item()/world_size, avg_s.item()/world_size
                        avg_lc, avg_ls = avg_lc.item()/world_size, avg_ls.item()/world_size
                        avg_lss = avg_lss.item()/world_size
                        avg_lsc = avg_lsc.item()/world_size
                        avg_r = avg_r.item()/world_size
                        avg_x0 = avg_x0.item()/world_size
                        avg_skel_h = avg_skel_h.item()/world_size
                        avg_std_mid = avg_std_mid.item()/world_size
                    else:
                        avg_l, avg_d = avg_l.item(), avg_d.item()
                        avg_c, avg_s = avg_c.item(), avg_s.item()
                        avg_lc, avg_ls = avg_lc.item(), avg_ls.item()
                        avg_lss = avg_lss.item()
                        avg_lsc = avg_lsc.item()
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
                            f"LStrS: raw {avg_lss:.4f} x {getattr(args,'w_latent_struct_skel',0):.1f} = {getattr(args,'w_latent_struct_skel',0)*avg_lss:.4f} | "
                            f"LStrC: raw {avg_lsc:.4f} x {getattr(args,'w_latent_struct_canny',0):.1f} = {getattr(args,'w_latent_struct_canny',0)*avg_lsc:.4f} | "
                            f"REPA: raw {avg_r:.4f} x {wr:.2f} = {r_contrib:.4f} | "
                            f"SkelH: raw {avg_skel_h:.4f} | "
                            f"StdMid: raw {avg_std_mid:.4f} | "
                            f"LR: {opt.param_groups[0]['lr']:.2e} | {ema_log}"
                            f"Steps/Sec: {steps_per_sec:.2f} | "
                            f"Mem: {torch.cuda.memory_reserved() / 1024 ** 3:.2f}G/"
                            f"{torch.cuda.max_memory_reserved() / 1024 ** 3:.2f}G"
                        )
                    
                    running_loss = running_diff = running_canny = running_skel = 0
                    running_latent_canny = running_latent_skel = running_repa = running_x0lat = running_skel_head = 0
                    running_std_mid = 0
                    running_latent_struct_skel = 0
                    running_latent_struct_canny = 0
                    log_steps = 0
                    start_time = time()

                _save_ckpt = args.ckpt_every > 0 and train_steps % args.ckpt_every == 0
                if train_steps <= 5000:
                    _save_ckpt = args.ckpt_every > 0 and train_steps % 1000 == 0
                elif train_steps > 5000:
                    _save_ckpt = args.ckpt_every > 0 and (train_steps - 5000) % args.ckpt_every == 0

                if _save_ckpt and train_steps > 0:
                    if rank == 0:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        # 始终保存完整 state_dict（LoRA 的 delta-only 保存已随
                        # src/model/lora.py 于 2026-08-31 删除）。
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

                        # ── In-process GPU eval: bf16 DDIM → VAE decode → save PNGs ──
                        # GPU-only (~40s for 455 imgs at batch=48). CPU metrics
                        # computed by eval_metrics_daemon.py (separate process).
                        if (_HAS_IN_PROCESS_EVAL and _eval_cache is not None
                                and ema_model is not None):
                            try:
                                _eval_bs = int(getattr(args, 'eval_batch', 240))
                                _eval_vae_bs = int(getattr(args, 'eval_vae_batch', 32))
                                _eval_steps = int(getattr(args, 'eval_steps', 50))
                                _eval_cfg = float(getattr(args, 'eval_cfg', 4.0))
                                _eval_t0 = time()
                                # Swap to EMA weights for eval.
                                # 注意：必须用 try/finally 保证任何异常路径都能把训练权重
                                # 还回去。旧实现只在 except 里调了 model.train()，
                                # 一旦 eval 抛异常，训练会从 EMA 权重继续跑，而 Adam 的
                                # 一/二阶矩仍对应旧权重 —— 静默的 training corruption。
                                _orig_sd = {k: v.clone() for k, v in model.state_dict().items()}
                                try:
                                    _m = model.module if hasattr(model, 'module') else model
                                    _m.load_state_dict(ema_model.state_dict(), strict=False)
                                    model.eval()

                                    run_gpu_eval(
                                        _m, args, _eval_cache, train_steps,
                                        checkpoint_dir, device,
                                        dit_batch=_eval_bs,
                                        vae_batch=_eval_vae_bs,
                                        ddim_steps=_eval_steps,
                                        cfg_scale=_eval_cfg)

                                    if _eval_show5_cache is not None:
                                        run_show5(_m, args, _eval_show5_cache, train_steps,
                                                 checkpoint_dir, device,
                                                 ddim_steps=_eval_steps, cfg_scale=_eval_cfg,
                                                 tag="show5")
                                    if _eval_seen5_cache is not None:
                                        run_show5(_m, args, _eval_seen5_cache, train_steps,
                                                 checkpoint_dir, device,
                                                 ddim_steps=_eval_steps, cfg_scale=_eval_cfg,
                                                 tag="seen5")

                                    logger.info(
                                        f"[auto-eval] step {train_steps} eval done in "
                                        f"{time()-_eval_t0:.1f}s (GPU inference + PNG save; "
                                        f"metrics by CPU daemon)")
                                finally:
                                    # 无论成功/失败/异常，都无条件恢复训练权重并释放备份，
                                    # 否则 eval 失败会静默污染后续训练，且显存持续泄漏。
                                    _m2 = model.module if hasattr(model, 'module') else model
                                    _m2.load_state_dict(_orig_sd, strict=False)
                                    model.train()
                                    del _orig_sd
                                    torch.cuda.empty_cache()
                            except Exception as _ee:
                                logger.warning(f"[auto-eval] step {train_steps} FAILED: {_ee}",
                                               exc_info=True)

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

def main_from_cli(argv=None):
    """CLI entry: build the argparse parser (config-file defaults + CLI overrides),
    parse, and run main(args). Used by `python train.py` (root launcher) and
    `python -m src.train.train`.
    """
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
    # 注：DiT_3Cond_models 已随 DiT_3Cond 于 2026-08-31 删除，choices 只剩 2Cond。
    parser.add_argument("--model", type=str, choices=list(DiT_2Cond_models.keys()), default="DiT-2Cond-S/2")
    parser.add_argument("--cond-mode", type=str, choices=["2cond", "3cond"], default="2cond",
                        help="Conditioning mode: 2cond (callig+char) or 3cond (callig+script+char).")
    parser.add_argument("--condition-fusion", type=str,
                        choices=["legacy", "factorized_add", "xl_highdim"], default="legacy",
                        help="Cond fusion: legacy joint MLP | factorized_add (low-dim additive) | "
                             "xl_highdim (high-dim, XL-aligned, preserves pretrained adaLN).")
    parser.add_argument("--callig-embed-dim", type=int, default=None)
    parser.add_argument("--script-embed-dim", type=int, default=None)
    parser.add_argument("--char-embed-dim", type=int, default=None)
    parser.add_argument("--char-dino-embeddings", type=str, default=None,
                        help="Path to glyph-level DINO embeddings npy (N, dim) used to init "
                             "y_char_embedder rows via glyph_id = script_id*7026+character_id. "
                             "Glyphs missing from the vocab keep their random init.")
    parser.add_argument("--char-dino-index", type=str, default=None,
                        help="Path to glyph index json ({\"glyphs\": [[script_id, char_id], ...]} "
                             "aligned row-wise with char-dino-embeddings).")
    parser.add_argument("--char-proj-mode", type=str, choices=["full", "ln_only", "mlp"],
                        default="full",
                        help="char_proj: 'full'=LayerNorm+Linear (default) | "
                             "'ln_only'=LayerNorm only, requires char_embed_dim==hidden_size "
                             "(DINO 384 direct, drops redundant 384->384 Linear — 但只给字符分支 "
                             "留下 768 个可学习参数, 实测不足以利用有效秩仅 3.1 的 DINO 向量) | "
                             "'mlp'=LayerNorm+Linear+SiLU+Linear (推荐, 给字符分支真正的容量)。")
    parser.add_argument("--dino-per-script-center", type=int, default=0, choices=[0, 1],
                        help="注入前先按 script 去均值再 L2 归一化。实测: 有效秩 34.1->57.0, "
                             "跨书体字符检索 top1 1.9%%->2.6%%, top5 4.2%%->6.8%%, "
                             "书体泄漏 83.0%%->77.9%%。书体信息本该由 y_callig_embedder 提供。")
    parser.add_argument("--dino-fill-unknown", type=int, default=1, choices=[0, 1],
                        help="DINO 未覆盖的 char 行用 DINO 均值填充 (默认开)。关闭则保留 "
                             "nn.Embedding 默认的 N(0,0.02) 冻结噪声 —— 在 char_proj='ln_only' "
                             "下 LayerNorm 会把范数线索也抹掉, 模型无法区分已知/未知字符。"
                             "CFG null token 永远不会被覆盖。")
    parser.add_argument("--freeze-char-table", type=_str_to_bool, default=False,
                        help="Freeze y_char_embedder table after DINO init (keep CFG uncond row "
                             "trainable). Saves ~13.5M trainable params; conditions become pure "
                             "DINO 384 vectors.")
    # ---- IDS 组件码本字嵌入 ----
    parser.add_argument("--use-ids-char-embedder", type=_str_to_bool, default=False,
                        help="Use IDS component-based char embedder instead of LabelEmbedder. "
                             "Reduces char table from 35130×384 to ~1571×384 (95.5% fewer params), "
                             "enables zero-shot generalization to unseen chars.")
    parser.add_argument("--ids-file", type=str, default=None,
                        help="Path to IDS dictionary file (cjkvi ids.txt format).")
    parser.add_argument("--ids-char-map-csv", type=str, default=None,
                        help="Path to csv with character_id,character columns for char_id->char mapping. "
                             "If None, assumes char_id == Unicode codepoint.")
    # ---- 标准字形 DINO 字嵌入 (冻结查表, 零可训练参数) ----
    parser.add_argument("--use-std-dino-char-embedder", type=_str_to_bool, default=False,
                        help="Use standard-glyph DINO frozen lookup table as char embedder "
                             "(0 trainable params, shape-consistency AUC>0.92). "
                             "Requires char_embed_dim == DINO dim (768).")
    parser.add_argument("--std-dino-table-path", type=str, default=None,
                        help="Path to std DINO char table npy (default _sync_work/std_dino_char_table_768.npy).")
    parser.add_argument("--cond-drop-all-prob", type=float, default=0.05,
                        help="Probability of dropping all factors for CFG.")
    parser.add_argument("--cond-drop-one-prob", type=float, default=0.0,
                        help="Probability of dropping exactly one uniformly selected factor.")
    parser.add_argument("--cond-drop-which-glyph-prob", type=float, default=0.5,
                        help="drop-one 时选择 drop callig (→glyph-only, 学字符内容分) 的概率; "
                             "书家维度样本充足, 字符维度才是难点, 建议 >0.5. 0.5=均匀.")
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
    parser.add_argument("--early-stop-metric", type=str,
                        choices=["ssim", "mse", "combo"], default="ssim",
                        help="Metric to monitor for early stopping (ssim higher better, mse lower "
                             "better). 'combo' = dual-gate: require BOTH ssim and skel_iou to be "
                             "stale >= patience before stopping (skel_iou higher better).")
    parser.add_argument("--diffusion-type", type=str,
                        choices=["ddpm", "flow"], default="ddpm",
                        help="Diffusion formulation: 'ddpm' = standard GaussianDiffusion "
                             "(epsilon prediction, DDIM sampling); 'flow' = linear-interpolant "
                             "Flow Matching (velocity prediction, ODE sampling).")

    # ---- Flow Matching: t 分布 / 求解器 / schedule（flow free-lunch）----
    parser.add_argument("--t-sampler", type=str, default="logit_normal",
                        choices=["uniform", "logit_normal", "cosmap"],
                        dest="t_sampler",
                        help="Training-time t distribution. 'logit_normal' (SD3) concentrates "
                             "gradient budget on mid-t instead of wasting it on the uninformative "
                             "endpoints. 'uniform' = legacy behaviour.")
    parser.add_argument("--t-mean", type=float, default=0.0, dest="t_mean",
                        help="logit_normal mean (SD3 uses 0.0). >0 biases towards t=1 (noise).")
    parser.add_argument("--t-std", type=float, default=1.0, dest="t_std",
                        help="logit_normal std (SD3 uses 1.0). Smaller = more concentrated at t=0.5.")
    # NOTE: dest 用 flow_sampler 而不是 sampler —— 后者已被数据采样器
    # (--sampler: random|factor_balanced) 占用，同名 dest 会互相覆盖。
    parser.add_argument("--flow-sampler", type=str, default="heun",
                        choices=["euler", "heun"], dest="flow_sampler",
                        help="ODE solver. 'heun' = 2nd-order RK2 (trapezoidal), 2 NFE/step. "
                             "At equal NFE Heun@25 beats Euler@50 because truncation error drops "
                             "from O(dt) to O(dt^2).")
    parser.add_argument("--flow-heun-batch", type=int, default=1, dest="heun_batch",
                        help="1 = evaluate Heun's two stages as one batched forward (much better "
                             "GPU utilisation); 0 = two separate forwards.")
    parser.add_argument("--flow-shift", type=float, default=1.0, dest="shift",
                        help="Sampling-side timestep shift (SD3). 1.0 = no shift (default, correct "
                             "for detail-dominated 32x32 glyph latents). >1 concentrates steps near "
                             "t=1 (layout), <1 near t=0 (detail).")
    parser.add_argument("--use-ot", type=_str_to_bool, default=False, dest="use_ot",
                        help="Minibatch Optimal Transport (OT-CFM, Tong et al. 2024): per-batch "
                             "Hungarian reassignment of noise/data pairs so trajectories don't cross "
                             "and the velocity field is smoother. Cheap (O(B^3) scipy), usually "
                             "helps convergence. No downside for training.")
    parser.add_argument("--ot-chunks", type=int, default=1,
                        help="OT 分块数: 1 = 整 batch 全局匈牙利 (原版); k>1 把 batch 均分 k 块各自 "
                             "做匈牙利, 大 batch 下 O(B^3)->O(k*(B/k)^3) 显著降 CPU 开销, 质量近似. "
                             "例: batch 384 + ot_chunks=4 ~= 4x96 https://bit.ly/OT-chunks.")
    parser.add_argument("--learn-sigma", type=int, default=None, choices=[0, 1],
                        help="Force DiT learn_sigma on/off. Default: auto = False for flow "
                             "(flow has no variance head; leaving it True creates C permanently "
                             "dead zero-initialized output channels), True for ddpm.")

    # ---- 骨干现代化（v2 arch）----
    parser.add_argument("--norm-type", type=str, default="rms", choices=["rms", "layer"],
                        dest="norm_type", help="Normalization inside DiT blocks / final layer.")
    parser.add_argument("--mlp-type", type=str, default="swiglu", choices=["swiglu", "gelu"],
                        dest="mlp_type",
                        help="Feed-forward. 'swiglu' is parameter-matched to 'gelu' "
                             "(hidden = 2/3 * 4D, rounded to multiple of 64).")
    parser.add_argument("--qk-norm", type=int, default=1, choices=[0, 1], dest="qk_norm",
                        help="QK-Normalization on attention q/k (stabilises logits, allows higher LR).")
    parser.add_argument("--rope", type=int, default=1, choices=[0, 1], dest="rope",
                        help="2D axial RoPE on q/k. 0 = legacy fixed 2D sin-cos added to the "
                             "residual stream.")
    parser.add_argument("--rope-theta", type=float, default=100.0, dest="rope_theta",
                        help="RoPE base frequency (SD3/Lumina use 100 for 2D image RoPE).")
    parser.add_argument("--attn-impl", type=str, default="sdpa", choices=["sdpa", "eager"],
                        dest="attn_impl", help="Attention kernel. 'sdpa' = Flash/mem-efficient.")
    parser.add_argument("--compile", type=_str_to_bool, default=False,
                        help="Wrap the whole model with torch.compile before DDP (needs torch>=2.0). "
                             "Speeds up PyTorch 2.x inductor kernels on cu121 env; first step is slow "
                             "(compilation), then per-step cost drops.")
    parser.add_argument("--compile-mode", type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode: default / reduce-overhead (CUDA-graph, faster but "
                             "higher mem) / max-autotune (slowest first compile, best kernels).")
    parser.add_argument("--early-stop-patience", type=int, default=5,
                        help="Stop after this many consecutive evals without improvement.")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.002,
                        help="Minimum change in the monitored metric to qualify as an improvement. "
                             "Without it, sub-noise-level jitter (+0.0001) resets the stale counter "
                             "and early-stop is effectively driven by eval noise.")
    parser.add_argument("--early-stop-min-delta-iou", type=float, default=0.005,
                        help="min_delta for the skel_iou gate when --early-stop-metric=combo.")
    parser.add_argument("--early-stop-min-delta-mse", type=float, default=0.0,
                        help="min_delta when --early-stop-metric=mse (scale-dependent, off by default).")
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
    parser.add_argument("--vae-downscale", type=int, default=8, help="VAE spatial downsample factor (8=f8 sd-vae, 4=f4 kl-f4)")
    parser.add_argument("--latent-channels", type=int, default=4, help="VAE latent channel count (4=sd-vae, 3=kl-f4)")
    parser.add_argument("--vae-in-channels", type=int, default=3, help="VAE input image channels (3=RGB, 1=grayscale)")
    parser.add_argument("--vae-out-channels", type=int, default=3, help="VAE output image channels (3=RGB, 1=grayscale)")
    parser.add_argument("--vae-scaling-factor", type=float, default=0.18215, help="VAE latent scaling factor")
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
    parser.add_argument("--train-only-char-embed", type=_str_to_bool, default=False,
                        help="DIAGNOSTIC: freeze the whole backbone and train ONLY the character "
                             "conditioning (y_char_embedder.embedding_table + char_proj + CFG null token). "
                             "Any metric change is then directly attributable to the char condition, "
                             "which is how we test whether the frozen DINO glyph table is the bottleneck. "
                             "Pair with freeze_char_table=false and --resume-full from a trained ckpt.")
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
    parser.add_argument("--eval-cfg", type=float, default=1.7,
                        help="CFG scale for free-sampling auto-eval (flow 最佳 ~1.7).")
    parser.add_argument("--eval-seed", type=int, default=0,
                        help="Seed for free-sampling auto-eval noise.")
    parser.add_argument("--eval-batch", type=int, default=16,
                        help="DiT sampling batch for auto-eval (before CFG doubling).")
    parser.add_argument("--eval-vae-batch", type=int, default=32,
                        help="VAE decode batch for auto-eval (fp32, force_upcast=True).")
    parser.add_argument("--show5-csv", type=str, default=None,
                        help="固定跨书体展示样本 CSV(如 eval5)。设后每次都采样这 N 个(不算指标, "
                             "仅生成 eval_latest.png/eval_samples 展示, 与海报 GT 行同一批保证对照)。")
    parser.add_argument("--seen5-csv", type=str, default=None,
                        help="固定训练集内展示样本 CSV。")
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
    parser.add_argument("--w-latent-struct-skel", type=float, default=0.0,
                        help="Weight for frozen StructDecoder skel BCE loss (latent→skel decoder). "
                             "Gradient ~10^-5 of latent norm, so typical values 5000-20000.")
    parser.add_argument("--w-latent-struct-canny", type=float, default=0.0,
                        help="Weight for frozen StructDecoder canny BCE loss (latent→canny decoder).")
    parser.add_argument("--latent-struct-decoder", type=str, default="",
                        help="Path to struct decoder checkpoint (skel_best.pt or canny_best.pt). "
                             "Both skel and canny decoders are loaded from same dir, filename "
                             "skel→canny substitution applied automatically.")
    parser.add_argument("--latent-struct-pos-weight", type=float, default=15.0,
                        help="BCE pos_weight for latent struct loss (15 for 3px skel, 8 for 3px canny).")
    parser.add_argument("--w-repa", type=float, default=0.0, help="Weight for Representation Alignment (REPA) Loss (0 = disabled, default)")
    parser.add_argument("--repa-teacher-ckpt", type=str, default="",
                        help="Local path to DINOv2 teacher weights (ModelScope safetensors). "
                             "Empty = auto-detect pretrained_models/dinov2_vits14_pretrain.safetensors or $DINO_WEIGHTS.")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon"],
                        help="Optimizer: adamw (default) or muon (matrix NS-orth + adamw for vec/embed).")
    parser.add_argument("--muon-lr", type=float, default=0.02,
                        help="Muon matrix-group LR (independent scale, ~0.01-0.1; adamw uses --lr).")
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

    args = parser.parse_args(argv)
    main(args)
    return args


if __name__ == "__main__":
    main_from_cli()
