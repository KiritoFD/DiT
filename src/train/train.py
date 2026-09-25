import os
os.environ["XFORMERS_DISABLED"] = "1"

# ── 分配器策略（2026-09-17）────────────────────────────────────────────────
# ⚠ 必须在 `import torch` **之前**设置（PyTorch 在首次 CUDA 分配时读这个变量）。
#
# **默认不设**，原因见下。
#
# 曾经的尝试：`expandable_segments:True` —— 让 allocator 把空闲段**归还驱动**，
#   以压低 `torch.compile` warmup 顶起来的高水位（实测 batch240 高水位 20.51G
#   而活跃只需 14.42G）。
# **为什么又关掉**：它让 `memory_reserved()` **每步都在变**（段被反复归还/重申请），
#   日志里的 `Mem:` 一直在波动，非常难读，也干扰"哪一步真的变慢了"这类判断。
#   而 v12 的同类配置（4ch/adaLN4/batch360）**不带它也能跑**（峰值 22.76G < 24.5G），
#   说明它不是必需的。
#
# 需要时显式打开即可（不改代码、不影响默认行为）：
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train/train.py ...
#
# 注意：**这里故意不设置该变量**。设成空字符串 PyTorch 会解析报错；
# 设成别的值又等于强加默认。保持"不碰"才是真正的默认行为。
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# [OPTIM-FUSED 2026-09-16] 让 cuDNN 为 (固定的) glyph_embedder conv
# 选最快算法; 形状固定, 一次性 autotune 成本可忽略。
torch.backends.cudnn.benchmark = True
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
from src.loss import REPALoss
from torch.utils.checkpoint import checkpoint as grad_ckpt
from src.utils import (DistributedFactorBalancedSampler,
                       LongEpochDistributedSampler)
from src.train.early_stop import EarlyStopper
from src.train.cli import parse_args
# ★ 模块级导入，**不要**写成函数内的局部 import：
#   Python 里函数内任何位置的 import 都会把该名字变成**局部变量**，
#   于是"只在 --expand-from-4ch 分支里 import"会让 --resume-full 分支
#   引用到未绑定的名字 -> UnboundLocalError（实测踩过：
#   `local variable 'materialize_lazy_params' referenced before assignment`）。
from src.utils.channel_expand import (drop_shape_mismatched,
                                      expand_ckpt_4ch_to_12ch,
                                      materialize_lazy_params)
from src.model.dit import MultiStyleEmbedder
from src.train.ckpt import save_checkpoint, prune_checkpoints, drain_ckpt
from src.eval.in_mem_eval import maybe_run_in_training, run_in_mem_eval

# ── in-process GPU eval 路径已删除（2026-09-17）────────────────────────────
# 它由 `--auto-eval` 门控，而**全仓 0 个配置把它设为 true**（70 个显式 false）。
# 实现归档到 src/eval/legacy/in_process_eval.py。
# 当前在训评测统一走 `--in-mem-eval`（src/eval/in_mem_eval.py），
# 事后批量补跑走 `--eval-mode deferred` + `python -m src.eval.batch_eval`。



def aux_dirs_of(args):
    """解析 ``--aux-latent-shards-dirs``（逗号分隔）—— **唯一入口**。

    原来这段表达式在 train.py 里重复了 6 次（247/293/900/903/1238/1452），
    每处都自己 split + strip + 过滤空串。任何一处写法漂移都会让
    "aux 通道数" 与 "aux 权重分组" 对不上 -> 静默错位（见 docs/system/70 §3.3）。
    """
    return [x for x in str(getattr(args, "aux_latent_shards_dirs", "") or "").split(",")
            if x.strip()]


def resolve_aux_channel_weights(args):
    """解析 aux 通道权重 —— **循环不变量，只在训练开始前调一次**。

    返回 ``(aux_dirs, weights)``；无 aux 时返回 ``([], [])``。

    ⚠ 2026-09-17: 这段（字符串解析 + 3 处校验）原来放在**每步**的数据分支里，
    每步重做一次；``_ch_w`` 更是在每步在 GPU 上重新分配 + 切片赋值
    （每次切片赋值 = 一次 kernel launch）。全是与 step 无关的工作。

    校验规则（保持原样，别放宽）：
      * 配了 aux 就**必须显式给权重** —— 静默回退到"等权"是已知有害配置
        （等权时 aux 吃掉 ~51% final-layer 梯度，patch_embed 上 aux 梯度是 img 的 2.6x，
         见 doc54/60）。"忘了写一行"就会掉进这个坑且不报错。
      * 权重个数必须与目录个数一致，否则通道分组错位。
    """
    aux_dirs = aux_dirs_of(args)
    if not aux_dirs:
        return [], []
    weights = [float(s) for s in
               str(getattr(args, "aux_loss_weights", "") or "").split(",") if s.strip()]
    if not weights:
        w_aux = float(getattr(args, "aux_loss_weight", 1.0))
        if w_aux == 1.0:
            raise ValueError(
                f"启用了 {len(aux_dirs)} 个 aux latent 目录, 但既没给 "
                f"--aux-loss-weights 也没把 --aux-loss-weight 设成非 1.0 "
                f"-> 会退化成**等权**, 而等权已被证明有害"
                f"(aux 吃 ~51% final-layer 梯度, doc54/60)。\n"
                f"请显式指定, 例如 --aux-loss-weights "
                f"{','.join(['0.3'] * len(aux_dirs))}。")
        weights = [w_aux] * len(aux_dirs)
    if len(weights) != len(aux_dirs):
        raise ValueError(
            f"--aux-loss-weights 给了 {len(weights)} 个值 "
            f"({weights}), 但 --aux-latent-shards-dirs 有 "
            f"{len(aux_dirs)} 个目录 -> 通道分组会错位。")
    return aux_dirs, weights


def build_aux_channel_weights(aux_dirs, weights, latent_channels, device):
    """构造逐通道 loss 权重向量 (C+4K,)，img 通道恒为 1。**循环外调一次。**

    形状与 ``torch.cat([latent(4), aux(4*K)], dim=1)`` 的目标一致。
    """
    n_ch = int(latent_channels) + 4 * len(aux_dirs)
    ch_w = torch.ones(n_ch, device=device)
    for gi, w in enumerate(weights):
        ch_w[4 + 4 * gi: 8 + 4 * gi] = w
    return ch_w


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def _load_new_callig_row(path, dim, n_new):
    """few-shot 新行初始化向量来自外部文件 (--new-callig-pt)。

    为什么需要: 表里每一行**不是**任意向量 —— v15 每行是该 (书家×书体) 全部样本
    DINOv2 CLS 特征 K-Means 的 K 个质心 (tools/build_multistyle_k4.py)，主干学到的
    是"读这种质心"。所以问"风格条件能不能装下全新书家"时，**应当先按同一套构造
    方式把新书家的质心直接写进表**（零梯度），而不是先验地猜"均值向量"(mean_scaled)。
    支持 dict{embedding|centroids|row} 或裸张量 (n_new,dim)/(K,D)/(dim,)。
    """
    path = str(path or "")
    if not path:
        raise SystemExit("[callig-emb] --init-new-callig row_pt 需要 --new-callig-pt <path>")
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join("/root/Workspace/xy/DiT", path)
    d = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(d, dict):
        for k in ("embedding", "centroids", "row"):
            if k in d:
                d = d[k]
                break
    t = torch.as_tensor(d, dtype=torch.float32)
    if t.ndim == 3:                       # (n_new, K, D)
        t = t.reshape(t.shape[0], -1)
    elif t.ndim == 1:
        t = t.unsqueeze(0)
    if t.ndim != 2 or t.shape[-1] != int(dim):
        raise SystemExit(f"[callig-emb] --new-callig-pt {tuple(t.shape)} 与表最后一维 "
                         f"{dim} 不符 (多 token 请先 flatten 成 (n_new, K*D))")
    if t.shape[0] != int(n_new):
        raise SystemExit(f"[callig-emb] --new-callig-pt 有 {t.shape[0]} 行, "
                         f"但表只新增 {n_new} 行")
    return t


@torch.no_grad()
def update_ema(ema_model, model, decay):
    """Update a full-precision model EMA, including floating-point buffers.

    [INFRA 2026-09-18] 原版对每个参数单独 `mul_().add_()`，并在**每次调用**重建
    named_parameters / named_buffers 两个 dict —— `ema_interval` 默认 1 时这是每步
    几百个小 kernel + Python 遍历/建表开销（其他地方都在抠 launch 与同步，这处是漏网）。
    现在：
      * 首次调用把 (ema_tensor, src_tensor) 配对**缓存**到 ema_model 上 —— 参数张量对象
        由优化器就地更新、跨步稳定（detach 与原参数共享 storage，始终读到最新值），故可安全缓存，
        之后零建表；
      * 归一后用 `torch._foreach_mul_/add_` 批处理成极少的 multi-tensor kernel。
    数值口径与原实现一致（同样的 decay / 1-decay、浮点 buffer 做 decay、整型 buffer 做 copy）。
    """
    pairs = getattr(ema_model, "_ema_pairs", None)
    if pairs is None:
        source = model.module if hasattr(model, "module") else model
        source_params = dict(source.named_parameters())
        source_buffers = dict(source.named_buffers())
        ema_p, src_p = [], []
        for name, ep in ema_model.named_parameters():
            sp = source_params.get(name)
            if sp is not None:
                ema_p.append(ep); src_p.append(sp.detach())
        ema_fb, src_fb, ema_cb, src_cb = [], [], [], []
        for name, eb in ema_model.named_buffers():
            sb = source_buffers.get(name)
            if sb is None:
                continue
            if torch.is_floating_point(eb):
                ema_fb.append(eb); src_fb.append(sb.detach())
            else:
                ema_cb.append(eb); src_cb.append(sb)
        pairs = (ema_p, src_p, ema_fb, src_fb, ema_cb, src_cb)
        ema_model._ema_pairs = pairs
    ema_p, src_p, ema_fb, src_fb, ema_cb, src_cb = pairs
    if ema_p:
        torch._foreach_mul_(ema_p, decay)
        torch._foreach_add_(ema_p, src_p, alpha=1.0 - decay)
    if ema_fb:
        torch._foreach_mul_(ema_fb, decay)
        torch._foreach_add_(ema_fb, src_fb, alpha=1.0 - decay)
    for eb, sb in zip(ema_cb, src_cb):
        eb.copy_(sb)

# ckpt 相关实现已移到 src/train/ckpt.py（state_to_cpu / AsyncCkptWriter /
# save_checkpoint / prune_checkpoints / drain_ckpt）

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

    # ── 书家词表收紧: 稀疏 raw id -> 连续 0..N-1 (见 src/utils/callig_map.py) ──
    # 设置 --callig-id-map 后, num_calligraphers 自动设为词表长度, 数据层查表映射;
    # 未设置时行为与旧版完全一致 (稀疏 id 直通)。
    # 注意: 此时 logger 尚未创建 (create_logger 在 setup_log_directory 之后),
    # 用 print + 前缀, 与后续 [callig-emb] 日志区分。
    args._callig_map = None
    if getattr(args, "callig_id_map", ""):
        from src.utils.callig_map import load_callig_id_map
        _cm_path = args.callig_id_map
        if not os.path.isabs(_cm_path) and not os.path.exists(_cm_path):
            _cm_path = os.path.join("/root/Workspace/xy/DiT", _cm_path)
        _cmap, _n = load_callig_id_map(_cm_path)
        args._callig_map = _cmap
        args.callig_id_map = _cm_path     # 回写解析后的绝对路径 (ckpt args/eval 端复用)
        print(f"[callig-map] 加载词表 {_cm_path}: "
              f"num_calligraphers {args.num_calligraphers} -> {_n}", flush=True)
        args.num_calligraphers = _n

    # ── (书家×书体) 联合风格词表 (2026-09-18): 设了就把"书家类"细化为 pair ──
    # 动机见 src/utils/callig_script_map.py 与 docs/system/73: 单向量装不下多书体风格
    # (ratio_style=1.18, r=-0.62)。num_calligraphers 改为 pair 数(87), 模型/预训练表
    # 加载/冻结逻辑全部复用(只是看到 87 类) -> 干净的标签级单变量改动。
    args._callig_script_map = None
    if getattr(args, "callig_script_map", ""):
        from src.utils.callig_script_map import load_callig_script_map
        _csm_path = args.callig_script_map
        if not os.path.isabs(_csm_path) and not os.path.exists(_csm_path):
            _csm_path = os.path.join("/root/Workspace/xy/DiT", _csm_path)
        _csmap = load_callig_script_map(_csm_path)
        args._callig_script_map = _csmap
        args.callig_script_map = _csm_path
        args.num_calligraphers = int(_csmap["num_pairs"])
        print(f"[callig-script-map] 加载 {_csm_path}: num_calligraphers -> "
              f"{args.num_calligraphers} (书家×书体 pair; 原书家 "
              f"{_csmap['num_calligraphers']}, sparse {len(_csmap.get('sparse_pairs', []))} 对)",
              flush=True)

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
        # 供外部 CPU 评测进程定位当前活动实验的 ckpt 目录。
        with open(f"{args.results_dir}/_active_ckpt_dir.txt", "w", encoding="utf-8") as _m:
            _m.write(checkpoint_dir + "\n")
        # log.txt lives inside this experiment dir (created first), never overwritten.
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        with open(f"{experiment_dir}/resolved_config.json", "w", encoding="utf-8") as _cf:
            json.dump(vars(args), _cf, ensure_ascii=False, indent=2)
        _sources = {}
        for _path in ("models.py", "train.py", "losses.py", "latent_dataset.py",
                      "samplers.py", "eval_auto.py"):
            if os.path.isfile(_path):
                with open(_path, "rb") as _sf:
                    _sources[_path] = hashlib.sha256(_sf.read()).hexdigest()
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

        # ── 通道一致性断言 (2026-09-17) ────────────────────────────────────
        # 12ch 下三个数必须自洽, 否则要么在 x_embedder 处炸, 要么**静默错位**
        # (aux 通道与权重分组对不上)。此前没有任何校验。
        _aux_dirs_chk = aux_dirs_of(args)
        _n_aux_chk = len(_aux_dirs_chk)
        _lat_ch = int(getattr(args, 'latent_channels', 4))
        _img_ch = (args.image_channels
                   if getattr(args, 'image_channels', None) is not None else _lat_ch)
        _in_ch = _lat_ch + 4 * _n_aux_chk
        assert _img_ch == _lat_ch, (
            f"image_channels({_img_ch}) 必须等于 latent_channels({_lat_ch}) —— "
            f"它是 **CFG 作用域**, 不能跟着 aux 一起放大, 否则 aux 通道被异分布的"
            f"同一 cfg_scale 放大 -> 采样轨迹跑飞 (doc59 §1.3 '墨团' 根因)。")
        logger.info(f"[channels] latent={_lat_ch}  aux 组={_n_aux_chk}  "
                    f"in_channels={_in_ch}  image_channels(CFG 作用域)={_img_ch}")

        model = DiT_2Cond_models[args.model](
            input_size=latent_size,
            num_calligraphers=args.num_calligraphers,
            num_characters=args.num_characters,
            use_checkpoint=args.use_checkpoint,
            learn_sigma=_learn_sigma,
            condition_fusion=args.condition_fusion,
            callig_embed_dim=args.callig_embed_dim,
            char_embed_dim=args.char_embed_dim,
            # g 的全局内容向量因子 (v12): 池化 g_tok -> 向量, 与 callig 一起做 concat/add。
            # 让 adaLN 调制分支第一次能看到"在写哪个字"。
            glyph_vec_cond=getattr(args, 'glyph_vec_cond', False),
            glyph_vec_dim=int(getattr(args, 'glyph_vec_dim', 128)),
            glyph_vec_pool=getattr(args, 'glyph_vec_pool', 'mean'),
            cond_drop_all_prob=args.cond_drop_all_prob,
            cond_drop_one_prob=args.cond_drop_one_prob,
            cond_drop_which_glyph_prob=getattr(args, 'cond_drop_which_glyph_prob', 0.5),
            use_glyph_cond=(getattr(args, 'w_glyph_cond', 0) > 0
                            or getattr(args, 'skel_as_glyph_cond', False)),
            use_char_cond=not getattr(args, 'no_char_cond', False),
            glyph_scale_init=getattr(args, 'glyph_scale_init', 0.4),
            glyph_concat_input=bool(getattr(args, 'glyph_concat_input', False)),
            glyph_gate_t=float(getattr(args, 'glyph_gate_t', 0.0)),
            glyph_gate_floor=float(getattr(args, 'glyph_gate_floor', 0.35)),
            glyph_drop_prob=getattr(args, 'glyph_drop_prob', 0.0),
            glyph_inject_layers=getattr(args, 'glyph_inject_layers', 0),
            glyph_inject_mode=getattr(args, 'glyph_inject_mode', 'adaln'),
            xattn_q_pos=getattr(args, 'xattn_q_pos', False),
            glyph_embedder_depth=getattr(args, 'glyph_embedder_depth', 0),
            # [v12+] glyph_embedder 的 3x3 conv 用 depthwise-separable (省约 8.6% 总 FLOPs)
            glyph_embedder_sep=getattr(args, 'glyph_embedder_sep', False),
            style_token_n=getattr(args, 'style_token_n', 0),
            style_role_init=getattr(args, 'style_role_init', 0.02),
            glyph_in_channels=4,
            in_channels=(getattr(args, 'latent_channels', 4)
                         + 4 * len(aux_dirs_of(args))),
            char_proj_mode=getattr(args, 'char_proj_mode', 'full'),
            callig_proj_mode=getattr(args, 'callig_proj_mode', 'linear'),
            callig_scale_init=float(getattr(args, 'callig_scale_init', 1.0)),
            # ---- 922/80 改动 1: 风格分支独立 LN + 独立增益 ----
            # 默认 (False, 1.0) = 不建任何参数 -> 与旧 ckpt 逐位等价。
            style_ln=bool(getattr(args, 'style_ln', False)),
            style_gain_init=float(getattr(args, 'style_gain_init', 1.0)),
            # gain 自动标定目标 y/t；配合 --style-gain-init -1 使用
            style_y_over_t_init=float(getattr(args, 'style_y_over_t_init', 1.0)),
            # ---- 922/80 改动 2: adaLN 风格专用 low-rank 支路 ----
            # 0 = 关闭（逐位兼容旧 ckpt）; 推荐 64（+2.1M, +6.4%）
            style_ada_rank=int(getattr(args, 'style_ada_rank', 0)),
            # v15 多模态风格: >0 时 y_callig_embedder 换 MultiStyleEmbedder(K token 查表)
            callig_multi_style_k=int(getattr(args, 'callig_multi_style_k', 0)),
            callig_style_ca=bool(getattr(args, 'callig_style_ca', False)),
            style_ctx_every_layer=bool(getattr(args, 'style_ctx_every_layer', False)),
            callig_spatial=getattr(args, 'callig_spatial', False),
            callig_spatial_rank=int(getattr(args, 'callig_spatial_rank', 64)),
            # ---- S2 (2026-09-22): 三层语义分解 + 局部风格-骨架引导 ----
            hier_style=int(getattr(args, 'hier_style', 0)),
            num_pairs=int(getattr(args, 'num_pairs', 0)),
            num_scripts=int(getattr(args, 'num_scripts', 12) or 12),
            # ⚠ --script-embed-dim 的历史默认是 None（3cond 时由别处兜底），
            #   S2 需要具体维度 -> None 时取 64。直接 int(None) 会 TypeError。
            script_embed_dim=int(getattr(args, 'script_embed_dim', None) or 64),
            pair_init=str(getattr(args, 'pair_init', 'zero')),
            pair_residual=int(getattr(args, 'pair_residual', 1)),
            script_film=bool(getattr(args, 'script_film', False)),
            spatial_film_rank=int(getattr(args, 'spatial_film_rank', 0)),
            lowrank_spatial_rank=int(getattr(args, 'lowrank_spatial_rank', 0)),
            local_ca_layers=int(getattr(args, 'local_ca_layers', 0)),
            local_ca_heads=int(getattr(args, 'local_ca_heads', 4)),
            local_ca_rank=int(getattr(args, 'local_ca_rank', 64)),
            local_ca_q=str(getattr(args, 'local_ca_q', 'g')),
            local_ca_window=int(getattr(args, 'local_ca_window', 0)),
            local_ca_at=getattr(args, 'local_ca_at', None),
            # 实现代次: glyph_query(默认) / legacy。评测端不用管(按 state_dict 自动判定)。
            local_ca_impl=str(getattr(args, 'local_ca_impl', 'glyph_query')),
            # ★ 必须透传: 否则模型建出来没有 deform_skel -> 形变与中间监督静默失效。
            #   (冒烟测试的 [deform] 告警就是这么抓到的; 与 local_ca_impl 漏透传同源。)
            deform_skel=int(getattr(args, 'deform_skel', 0)),
            deform_width=int(getattr(args, 'deform_width', 64)),
            deform_grid=int(getattr(args, 'deform_grid', 32)),
            deform_max_off=float(getattr(args, 'deform_max_off', 3.0)),
            deform_coarse=int(getattr(args, 'deform_coarse', 8)),
            residual=int(getattr(args, 'residual', 0)),
            res_cap=float(getattr(args, 'res_cap', 1.0)),
            deform_ckpt=str(getattr(args, 'deform_ckpt', '') or ''),
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
            # image_channels = **CFG 作用域**(只对图像 latent 通道做引导, 不引导 aux
            #   结构通道)。与 latent_channels(图像 latent 通道数) 在 12ch 下不同:
            #   latent_channels=4, in_channels=4+4*len(aux)=12, 而 image_channels 必须
            #   保持 4 —— 否则 CFG 会放大分布不同的 aux 通道, 采样轨迹跑飞
            #   (doc59 §1.3 "墨团"根因)。显式 --image-channels 优先。
            image_channels=(args.image_channels
                            if getattr(args, 'image_channels', None) is not None
                            else getattr(args, 'latent_channels', 4)),
        )
        # ── 双轴 CFG 的推理期开关 (2026-09-17) ─────────────────────────────
        # 挂在**模型属性**上, 让 src/eval/inference.sample_latents 不必改签名即可透传。
        #   cfg_glyph_scale = None -> 经典 2 路 CFG (向后兼容)
        #   非 None                -> forward_with_cfg 委托给 forward_with_2axis_cfg
        # ⚠ 只在训练时 glyph_drop_prob > 0 的 ckpt 上有意义 —— g=0 必须是训练见过的
        #   条件, 否则"内容轴"同样未训练, 会给出误导性结果(与 callig null 行同理)。
        model.cfg_glyph_scale = (None if getattr(args, 'cfg_glyph_scale', None) is None
                                 else float(args.cfg_glyph_scale))
        model.cfg_w_inter = float(getattr(args, 'cfg_w_inter', 0.0) or 0.0)

        # ── S2 启动自检 (2026-09-22) ────────────────────────────────────────
        # 为什么必须打这行: 本项目反复栽在"配置没生效但一声不吭"上
        #   (drop-guard 漏 factorized_cat / batch_eval 把 args 转 dict /
        #    parse_known_args 漏传 argv ...)。S2 有 8 个开关, 更该显式自报家门。
        _s2_on = (int(getattr(model, 'hier_style', 0)) > 0
                  or getattr(model, 'spatial_film', None) is not None
                  or getattr(model, 'lowrank_spatial', None) is not None
                  or getattr(model, 'local_ca', None) is not None)
        if _s2_on:
            _n_new = sum(p.numel() for n, p in model.named_parameters()
                         if n.startswith(("style_hier.", "script_film.",
                                          "spatial_film.", "lowrank_spatial.",
                                          "local_ca.")))
            _n_all = sum(p.numel() for p in model.parameters())
            logger.info(
                f"[hier-style] 三层语义分解已启用: num_pairs={getattr(model.style_hier, 'num_pairs', 0)} "
                f"num_scripts={getattr(model.style_hier, 'num_scripts', 0)} "
                f"script_dim={getattr(model, 'script_embed_dim', 0)} "
                f"pair_residual={getattr(getattr(model, 'style_hier', None), 'pair_residual', '-')} "
                f"| 通路: script_film={model.script_film is not None} "
                f"lowrank_spatial={'rank' + str(getattr(args, 'lowrank_spatial_rank', 0)) if model.lowrank_spatial is not None else False} "
                f"glyph_gate={getattr(args, 'glyph_gate_t', 0)}->{getattr(args, 'glyph_gate_floor', 0.35)} "
                f"spatial_film={'rank' + str(getattr(args, 'spatial_film_rank', 0)) if model.spatial_film is not None else False} "
                f"local_ca={0 if model.local_ca is None else len(model.local_ca)}"
                f"@{getattr(model, 'local_ca_at', [])}(q={getattr(model, 'local_ca_q', '-')},"
                f"win={getattr(args, 'local_ca_window', 0)}) "
                f"| 新增参数={_n_new/1e3:.1f}K / 总 {_n_all/1e6:.3f}M ({_n_new/_n_all*100:.2f}%)")
            if model.style_hier is not None and getattr(model.style_hier, 'pair_residual', 1) == 0:
                logger.info("[hier-style] pair_residual=0 -> α 冻结为 0（'只加书体层'的同参数消融）")
            if getattr(model, 'local_ca', None) is not None and int(getattr(args, 'local_ca_layers', 0)) > 0:
                logger.info("[hier-style] 注意: 所有 S2 新模块都是 zero-init, "
                            "step0 与旧路径**逐位相等** -> 可直接 resume 做单变量 A/B。")
        else:
            logger.info("[hier-style] 未启用（hier_style=0 且无局部通路）—— 走 v13/v15 旧路径")
        if model.cfg_glyph_scale is not None:
            if float(getattr(args, 'glyph_drop_prob', 0.0) or 0.0) <= 0.0:
                logger.warning(
                    "[cfg-2axis] cfg_glyph_scale 已设置, 但 glyph_drop_prob=0 -> "
                    "**内容轴 (g=0) 在训练中从未出现过**, 双轴 CFG 的 content 轴无效。"
                    "要么把 glyph_drop_prob 设为 >0 重训, 要么别用双轴。")
            else:
                logger.info(f"[cfg-2axis] 双轴 CFG 已启用: cfg_callig=eval_cfg, "
                            f"cfg_glyph={model.cfg_glyph_scale}, w_inter={model.cfg_w_inter} "
                            f"(glyph_drop_prob={getattr(args,'glyph_drop_prob',0.0)})")

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
                    f"one:{args.cond_drop_one_prob}, "
                    f"glyph_cond={(getattr(args, 'w_glyph_cond', 0) > 0) or getattr(args, 'skel_as_glyph_cond', False)}, glyph_scale_init={getattr(args, 'glyph_scale_init', 0.4)}, "
                    f"char_proj_mode={getattr(args, 'char_proj_mode', 'full')}, "
                    f"freeze_char_table={getattr(args, 'freeze_char_table', False)})")

    # ── 书家词表: 对比预训练 embedding 加载 + 可选冻结 (2026-09-08) ──────────
    # 词表收紧 (--callig-id-map) 后表只有 N+1 行; 预训练表 (SupCon w/ DINO 风格
    # 特征, tools/pretrain_callig_emb.py) 覆盖前 N 行, CFG null 行保持随机。
    # freeze_callig_table 时 null token 拆成独立 Parameter 保持可训练
    # (LabelEmbedder.freeze_table(), 同 char 表的成熟机制)。
    if getattr(args, "callig_emb_pretrained", ""):
        _cep = args.callig_emb_pretrained
        if not os.path.isabs(_cep) and not os.path.exists(_cep):
            _cep = os.path.join("/root/Workspace/xy/DiT", _cep)
        _d = torch.load(_cep, map_location="cpu", weights_only=False)
        _emb = _d["embedding"] if isinstance(_d, dict) else _d
        _w = model.y_callig_embedder.embedding_table.weight
        if isinstance(model.y_callig_embedder, MultiStyleEmbedder):
            # v15: 表 (N, K*D) 不含 null 行, 全表覆盖; null_embed 独立参数保持随机
            _pn, _pd = _emb.shape
            if _pn > _w.shape[0] or _pd != _w.shape[1]:
                raise SystemExit(
                    f"[callig-emb] 预训练多模态风格表 {_emb.shape} 与表 "
                    f"{tuple(_w.shape)} 不兼容")
            with torch.no_grad():
                _w[:_pn].copy_(_emb.float())
                # ★ [2026-09-20] few-shot 新增 (书家x书体) 对: 表扩了但预训练表没那么大。
                #   新行用已有各行的均值(保留方向+拉到典型范数) —— v15 每行是
                #   K*D=1536 维(4 个子风格 x 384)，比 v13 的 128 维容量大得多，
                #   正好用来验证"更多风格容量能否捕获新书家"。
                if _w.shape[0] > _pn:
                    _init = str(getattr(args, "init_new_callig", "mean_scaled"))
                    _n_new = _w.shape[0] - _pn
                    if _init == "row_pt":
                        _rows = _load_new_callig_row(
                            getattr(args, "new_callig_pt", ""), _w.shape[1], _n_new)
                    else:
                        _base = _emb.float().mean(0)
                        if _init == "mean_scaled":
                            _n0 = _emb.float().norm(dim=1)
                            _tgt = float(_n0.median()); _cur = float(_base.norm())
                            if _cur > 1e-8 and _tgt > 0:
                                _base = _base * (_tgt / _cur)
                            logger.info(f"[callig-emb] v15 mean_scaled: {_cur:.3f} -> {_tgt:.3f}")
                        _rows = _base.expand(_n_new, -1)
                    _w[_pn:].copy_(_rows)
                    logger.info(f"[callig-emb] 新增 {_n_new} 个(书家x书体)行, "
                                f"初始化={_init}, 范数="
                                f"{[round(float(x), 3) for x in _w[_pn:].norm(dim=1)]}")
            logger.info(f"[callig-emb] 加载预训练多模态风格表 {_cep}: {_emb.shape} "
                        f"(K={model.y_callig_embedder.k_clusters}), null_embed 保持随机")
            _pretrained_n_callig = _pn
        else:
            _n_pre, _dim = _emb.shape
            _n_emb = _w.shape[0] - 1          # 表行数(不含 null 行)
            if _n_pre > _n_emb or _dim != _w.shape[1]:
                raise SystemExit(
                    f"[callig-emb] 预训练书家表形状 {_emb.shape} 与表 "
                    f"{tuple(_w.shape)}(-null) 不兼容")
            with torch.no_grad():
                _w[:_n_pre].copy_(_emb.float())
                # ★ [2026-09-20] few-shot 新增书家: 表扩了但预训练表没那么大。
                #   新行用**已有书家的均值**初始化 —— 从"平均书家"出发，
                #   比随机初始化(分布外)收敛快得多。这是 few-shot 成败的关键之一。
                if _n_emb > _n_pre:
                    _init = str(getattr(args, "init_new_callig", "mean_scaled"))
                    if _init == "row_pt":
                        _w[_n_pre:_n_emb].copy_(_load_new_callig_row(
                            getattr(args, "new_callig_pt", ""), _dim, _n_emb - _n_pre))
                        logger.info(f"[callig-emb] 新增 {_n_emb - _n_pre} 个书家行, "
                                    f"初始化=row_pt(--new-callig-pt)")
                        _base = None
                    elif _init in ("mean", "mean_scaled"):
                        _base = _emb.float().mean(0)
                        # ⚠⚠ 关键修正: 各行近似正交时, **均值向量的范数会远小于单个行**
                        #    (实测: 各行范数 med=11.0, 而均值向量范数只有 2.06)。
                        #    直接用均值初始化 -> 新行比典型书家小 5 倍 -> 起点就在分布外,
                        #    主干(冻结)根本用不了这么小的条件向量。
                        #    所以保留均值**方向**, 但把范数拉回典型书家的量级。
                        if _init == "mean_scaled":
                            _nrm = _emb.float().norm(dim=1)
                            _tgt = float(_nrm.median())
                            _cur = float(_base.norm())
                            if _cur > 1e-8 and _tgt > 0:
                                _base = _base * (_tgt / _cur)
                            logger.info(f"[callig-emb] mean_scaled: 均值范数 {_cur:.3f} "
                                        f"-> 目标(各行中位数) {_tgt:.3f}")
                    elif _init == "zeros":
                        _base = torch.zeros(_dim)
                    else:                       # random: 保持默认初始化
                        _base = None
                    if _base is not None:
                        _w[_n_pre:_n_emb].copy_(_base.expand(_n_emb - _n_pre, -1))
                    logger.info(f"[callig-emb] 新增 {_n_emb - _n_pre} 个书家行, "
                                f"初始化={_init}, 范数={float(_base.norm()):.3f}"
                                if _base is not None else
                                f"[callig-emb] 新增 {_n_emb - _n_pre} 个书家行, "
                                f"初始化={_init}")
            logger.info(f"[callig-emb] 加载预训练书家表 {_cep}: {_emb.shape}, "
                        f"null 行保持随机")
            _pretrained_n_callig = _n_pre      # 供 --train-only-new-callig 用
    if getattr(args, "freeze_callig_table", False):
        assert getattr(args, "callig_emb_pretrained", ""), \
            "freeze_callig_table 需要先 --callig-emb-pretrained (否则冻结随机表)"
        model.y_callig_embedder.freeze_table()
        logger.info("[callig-emb] 书家表已冻结 [0,N), CFG null token 保持可训练")

    # ── 注入门控统计 (debug): forward hook 记录 xattn 注入输出的平均 L2 ──
    # 零初始化起点, 该值增长 = 注入正在学会写入残差流 (惰性注册, 首个 debug 步挂)
    _inj_stats = {}

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

    # 加载顺序(固定): 预训练主干 -> 重置条件调制层 -> 加载 delta。
    # ckpt 里的 `delta` 是**完整** state_dict（LoRA 的 delta-only 保存已于 2026-08-31 删除）。
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


    # 2.5) 4ch -> 12ch 通道扩展（12ch 后训练）。
    #      ⚠ 必须在 --resume-full **之前**跑，且这里载入的权重优先 ——
    #      否则 strict=False 会静默跳过形状不匹配的 3 个张量。
    if getattr(args, 'expand_from_4ch', None):
        import torch as _torch
        _ec = _torch.load(args.expand_from_4ch, map_location="cpu", weights_only=False)
        _n_aux = len(aux_dirs_of(args))
        if _n_aux == 0:
            raise SystemExit(
                "[expand-4ch] 需要 --aux-latent-shards-dirs（否则目标仍是 4ch，无需扩展）")
        _ec = expand_ckpt_4ch_to_12ch(_ec, _n_aux)
        _esd = _ec.get("delta", _ec.get("model", _ec))
        # 先把 ckpt 里的懒参数（null_embed）补出来，否则会被静默丢弃
        _lazy = materialize_lazy_params(model, _esd)
        if _lazy:
            logger.info(f"[expand-4ch] 补出懒参数: {_lazy}")
        _emiss, _eunexp = model.load_state_dict(_esd, strict=False)
        logger.info(
            f"[expand-4ch] {args.expand_from_4ch} -> in_channels="
            f"{4 + 4 * _n_aux} (aux 组={_n_aux})；"
            f"扩展张量 {_ec.get('_expand_report', {})}；"
            f"missing={len(_emiss)} unexpected={len(_eunexp)}")
        if _emiss or _eunexp:
            # 扩展后仍有缺失/多余 = 说明 4ch 与 12ch 的差异不止那 3 个张量，
            # 或前缀剥离不对 —— 这种必须报出来，不能静默继续。
            logger.warning(f"[expand-4ch] 扩展后仍有 missing={_emiss[:6]} "
                           f"unexpected={_eunexp[:6]} —— 请核对")
        # 优化器状态与参数形状绑定，不复用（expand_ckpt 已 pop 掉 opt）
        _resume_full_ckpt = None
        if getattr(args, 'resume_full', None):
            logger.warning("[expand-4ch] 已忽略 --resume-full 的模型权重（通道扩展优先）")

    # 3) 完整 resume（从零训练与续跑共用同一条路径）。
    if (getattr(args, 'resume_full', None) is not None
            and not getattr(args, 'expand_from_4ch', None)):
        import torch as _torch
        _rf = _torch.load(args.resume_full, map_location="cpu", weights_only=False)
        _resume_full_ckpt = _rf
        _sd = _rf.get("delta", _rf.get("model", _rf))
        # torch.compile 存盘键带 _orig_mod. 前缀 —— 不剥离则全部 missing, 模型静默随机重启
        _sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
               for k, v in _sd.items()}
        # 先把 ckpt 里的懒参数（null_embed）补出来，否则会被静默丢弃
        _lazy = materialize_lazy_params(model, _sd)
        if _lazy:
            logger.info(f"[resume-full] 补出懒参数: {_lazy}")
        # 架构演进式 resume（如 v15 换条件头形状）: 形状对不上的键剔除并视为新模块
        # （strict=False 遇形状不匹配会 RuntimeError; 剔除后由冻结策略保持可训）
        # ★ [2026-09-21] 例外: **书家风格表扩行**。few-shot 把表从 87 行扩到 88 行,
        #   若走下面的 shape guard, 整张**已训好**的风格表会被当成"形状不匹配的新模块"
        #   剔除 -> 模型只剩 --callig-emb-pretrained 那份 K-Means **step 0 初始质心**,
        #   而主干是 150k 的。150k 的主干读 0 步的表 = 首尾不匹配,
        #   之前所有 v15 few-shot 的 baseline 与训练数字全部因此作废。
        #   正确语义: ckpt 的前 n 行逐行搬进模型表, 新增行保留 --init-new-callig 初值。
        _tbl_key = "y_callig_embedder.embedding_table.weight"
        if _sd.get(_tbl_key, None) is not None and _sd[_tbl_key].ndim == 2:
            _mt = model.y_callig_embedder.embedding_table.weight
            _ct = _sd[_tbl_key]
            if _ct.shape[1] != _mt.shape[1]:
                raise SystemExit(
                    f"[resume-full] 风格表列数不符: ckpt {tuple(_ct.shape)} vs "
                    f"模型 {tuple(_mt.shape)} —— 不是扩行, 是换维度, 请确认基模")
            if _ct.shape[0] > _mt.shape[0]:
                raise SystemExit(
                    f"[resume-full] ckpt 表有 {_ct.shape[0]} 行 > 模型 {_mt.shape[0]} 行, "
                    f"装不下 —— 检查 --num-calligraphers")
            if _ct.shape[0] < _mt.shape[0]:
                with torch.no_grad():
                    _mt[:_ct.shape[0]].copy_(_ct.to(_mt.dtype))
                logger.info(
                    f"[resume-full] 风格表扩行: ckpt {tuple(_ct.shape)} -> 模型 "
                    f"{tuple(_mt.shape)}，前 {_ct.shape[0]} 行=已训好的表，"
                    f"新增行保留 init 值 (范数="
                    f"{[round(float(x), 3) for x in _mt[_ct.shape[0]:].norm(dim=1)]})")
                del _sd[_tbl_key]         # 已手工搬入, 不再交给 shape guard
        _shape_dropped = drop_shape_mismatched(model, _sd)
        if _shape_dropped:
            logger.warning(
                f"[resume-full] {len(_shape_dropped)} 个键形状不匹配, 已剔除"
                f"(重新初始化, 须保持可训): {sorted(_shape_dropped)[:8]} ...")
        missing, unexpected = model.load_state_dict(_sd, strict=False)
        logger.info(f"[resume-full] Loaded weights from {args.resume_full} "
                    f"(missing={len(missing)}, unexpected={len(unexpected)}, "
                    f"shape_dropped={len(_shape_dropped)}).")
        # 供后面的冻结策略用：ckpt 里**没有**的权重 = 新模块，必须可训
        # （含形状失配被剔除的键 —— 它们同样从随机初始化开始）
        _resume_missing = set(missing) | set(_shape_dropped)
        # ★ 护栏: 若仍有 null_embed 落在 unexpected 里，说明加载顺序错了 ——
        #   不报错的话 CFG 的 null 向量会被静默重新随机化（loss 照降，但 CFG 变味）。
        _lost_null = [k for k in unexpected if k.endswith(".null_embed")]
        if _lost_null:
            logger.error(
                f"[resume-full] ✗ {_lost_null} 未被加载 —— 这些是懒参数，"
                f"必须在 load_state_dict **之前**创建（见 materialize_lazy_params）。"
                f"CFG 的 null 向量会因此被重新随机化。")

    # ---- freeze / trainable policy --------------------------------------------
    # Two regimes:
    #   1) pretrained body: freeze the pretrained transformer body,
    #      train only the *new* condition head + adaLN/final_layer.
    #      adaLN is reset to std=0.02 by `reset_cond_head`, so it MUST be trainable
    #      (`train_cond_head=true`), otherwise the model is stuck on random modulation.
    #   2) from-scratch (pretrained=None): keep all params trainable.
    #   3) [2026-09-18] --train-only-style: 冻结**全部**主干，只训风格模块。
    #      风格模块 CalligStyleCrossAttn 的 out_proj 是 zero-init，
    #      所以 step0 输出**恒等于**已训好的 ckpt -> 任何 metric 变化都可归因于新增风格容量。
    #   4) [2026-09-20] --train-only-new-callig: few-shot 新书家。
    #      冻结全部，**只让书家表的【新行】**更新（旧行梯度用 hook 清零），
    #      保证已训好的 45 个书家表示不漂移。新增行数 = 表行数-1 - 预训练行数。
    _train_only_new_callig = bool(getattr(args, 'train_only_new_callig', False))
    if _train_only_new_callig:
        _tbl = model.y_callig_embedder.embedding_table.weight
        _n_all = _tbl.shape[0]
        _n_old = int(locals().get('_pretrained_n_callig', _n_all - 1))
        # 表布局有两种（[2026-09-20 bugfix]）:
        #   LabelEmbedder(v13)      表 (N+1, D)   末行 = CFG null 行 -> 新行 [_n_old, _n_all-1)
        #   MultiStyleEmbedder(v15) 表 (N, K*D)   **无 null 行**(null_embed 是独立参数)
        #                           -> 新行 [_n_old, _n_all)
        #   ⚠⚠ 旧代码写死 [_n_old, _n_all-1)，对 v15 恰好把**唯一的新行(=末行)**排除掉
        #      -> 0 参数可训、训练完全空转（v15_fs 首轮 4 书家全部实测踩中:
        #         "[87:87) -> 0 参数"，150100/150200 的 eval 与 baseline 完全同值）。
        if isinstance(model.y_callig_embedder, MultiStyleEmbedder):
            _n_new_end = _n_all
        else:
            _n_new_end = _n_all - 1
        if _n_old >= _n_new_end:
            raise SystemExit(
                f"[train-only-new-callig] 表没有新增行（预训练 {_n_old} / 表 {_n_all}, "
                f"新行范围 [{_n_old}:{_n_new_end})）—— "
                f"请设 --num-calligraphers > 预训练书家数")
        requires_grad(model, False)
        _tbl.requires_grad_(True)
        # ⚠ 三个坑（都实测踩过）:
        #   1) requires_grad 是**逐张量**的，不能只放开某几行 -> 整表放开 +
        #      用 grad hook 把旧行梯度清零。
        #   2) 表末行: v13 是 CFG null 行（要排除）；v15 没有 null 行（见上面的分支）。
        #   3) hook 里的 mask 必须在**同一 device**（grad 在 cuda，mask 默认 cpu
        #      -> RuntimeError: Expected all tensors to be on the same device）
        _mask = torch.zeros_like(_tbl)
        _mask[_n_old:_n_new_end] = 1.0
        # ⚠⚠ 必须在 hook **内部**把 mask 搬到 grad 的 device。
        #   建 mask 时模型还在 CPU（冻结策略跑在 .to(device) 之前），
        #   写死 device 会导致 backward 时报
        #   "Expected all tensors to be on the same device, cuda:0 and cpu!"
        #   （这个坑连踩两次：先建在默认 cpu，再改成 _tbl.device，都不行）
        _tbl.register_hook(lambda g: g * _mask.to(g.device))
        # ⚠⚠ [2026-09-21] AdamW 的 weight decay 是**解耦**的 (p -= lr·wd·p)，
        #   不经过梯度 -> 上面的 grad hook **拦不住它**：被冻结的 87 行会每步缩小，
        #   整张风格表连同主干一起退化，few-shot 表现为"越训越差"。
        #   (lr=3e-3, wd=0.1 -> 每步 ×0.9997, 2000 步后只剩 0.55)
        #   few-shot 只有 1536 个可训参数，WD 在这里没有任何收益 -> 归零。
        if float(getattr(args, "weight_decay", 0.0) or 0.0) > 0:
            logger.warning(f"[train-only-new-callig] weight_decay "
                           f"{args.weight_decay} -> 0.0（解耦 WD 会绕过 grad hook，"
                           f"把已训好的 {_n_old} 行风格向量一起衰减掉）")
            args.weight_decay = 0.0
        _is_mse = isinstance(model.y_callig_embedder, MultiStyleEmbedder)
        logger.info(f"[train-only-new-callig] 冻结主干，只训书家表新增行 "
                    f"[{_n_old}:{_n_new_end}) -> "
                    f"{(_n_new_end - _n_old) * _tbl.shape[1]} 参数"
                    + ("" if _is_mse else
                       f"（末行 {_n_all - 1} 是 CFG null 行，已排除）"))

    _train_only_style = bool(getattr(args, 'train_only_style', False))
    if _train_only_style:
        requires_grad(model, False)
        # ⚠ 只训 callig_style_ca 是**不够的**（实测踩过）：
        #   把 glyph_inject_mode 从 adaln 改成 xattn 会换掉整条注入路径，
        #   ckpt 里这 53 个权重是**全新随机初始化**的。若只放开 callig_style_ca，
        #   这些新模块就永远训不了 -> 字形条件彻底坏掉 -> Diff 从 0.20 炸到 1.73。
        #   正确语义：**只训练 ckpt 里没有的新模块**（主干仍然全冻）。
        _miss = set(locals().get('_resume_missing', set()) or set())
        _n_tr, _tot, _n_new = 0, 0, 0
        # v15 多模态风格: 风格链 = 查表(+null) + 书家化骨架 CA + 条件融合层,
        # 连同 ckpt 缺失的新张量全部放开; 其余主干(x_embedder/blocks/t_embedder/
        # glyph_embedder/glyph_vec_proj/glyph_scale)保持冻结。
        _multi_style = isinstance(getattr(model, 'y_callig_embedder', None),
                                  MultiStyleEmbedder)
        for _nm, _p in model.named_parameters():
            _tot += _p.numel()
            if ('callig_style_ca' in _nm or 'style_role' in _nm
                    or (_multi_style and ('y_callig_embedder' in _nm
                                          or 'cond_fusion' in _nm
                                          or 'callig_scale' in _nm))):
                _p.requires_grad = True
                _n_tr += _p.numel()
            elif _nm in _miss:
                _p.requires_grad = True
                _n_tr += _p.numel()
                _n_new += 1
        if _n_tr == 0:
            raise SystemExit(
                "[train-only-style] 匹配不到任何可训参数 —— "
                "检查是否设了 --style-token-n > 0 或 --callig-multi-style-k > 0"
                "（否则模型里没有风格模块）")
        logger.info(f"[train-only-style] 冻结主干，只训【新】模块: "
                    f"{_n_tr:,} / {_tot:,} 参数可训 ({_n_tr/_tot*100:.2f}%)，"
                    f"其中 ckpt 缺失的新张量 {_n_new} 个")
        if _n_new > 0:
            logger.warning(
                f"[train-only-style] ⚠ 有 {_n_new} 个张量在 ckpt 里不存在（换了架构），"
                f"它们从随机初始化开始训 —— 因此 **step0 不等于原 ckpt**，"
                f"需要足够的步数让新路径收敛后再比较指标")

    # ── Stage 3: 冻主干、只微调书家风格表 + 锚定正则 (2026-09-18) ──────────────
    # 动机: 端到端训表会塌缩(cos 0.323); styletok 实测"冻主干只训风格参数"若无护栏会
    # 漂移/记忆 -> strict 掉头。这里 (a) 只放开书家表本体(+null), (b) 加"锚到预训练表"的
    # L2 正则, 让它能朝渲染需要特化但不许漂回塌缩。判据用 ratio_style, 不用 strict ssim。
    _train_only_callig_table = bool(getattr(args, 'train_only_callig_table', False))
    if _train_only_callig_table:
        requires_grad(model, False)
        _n_tr = 0
        for _nm, _p in model.named_parameters():
            _is_table = ('y_callig_embedder.embedding_table' in _nm
                         or 'y_callig_embedder.null_embed' in _nm)
            _is_proj = (getattr(args, 'style_tune_proj', False)
                        and ('callig_proj' in _nm or 'cond_fusion' in _nm
                             or 'callig_scale' in _nm))
            if _is_table or _is_proj:
                _p.requires_grad = True
                _n_tr += _p.numel()
        if _n_tr == 0:
            raise SystemExit("[train-only-callig-table] 没匹配到可训参数 "
                             "(模型是否有 y_callig_embedder?)")
        logger.info(f"[train-only-callig-table] 冻结主干, 只训书家风格表: {_n_tr:,} 参数可训")

    # 锚定正则的目标表(Stage 3 / v15 用): 当前风格参数 vs 预训练的 L2, 防塌缩/防漂。
    # 参数张量对象跨步稳定(优化器就地更新), 故循环前取一次引用即可。
    # mode="row" : 锚每行表向量本身 (v14 stage3 口径, target = 预训练表 embedding)
    # mode="mean": 锚 K token 的 mean pooling (v15 口径, target = pair 级 DINO 质心
    #              'pair_mean') —— 允许 K 个 token 各自分化, 只约束其均值不漂离
    #              该 pair 的真实风格中心, 防止 K=4 聚类在训练中重新塌缩。
    _style_anchor_w = float(getattr(args, 'style_anchor_weight', 0.0) or 0.0)
    _style_anchor_mode = str(getattr(args, 'style_anchor_mode', 'row') or 'row')
    _style_table_param = None
    _style_anchor_target = None
    _style_anchor_k = 1
    if _style_anchor_w > 0:
        _raw = model.module if hasattr(model, "module") else model
        _raw = getattr(_raw, "_orig_mod", _raw)
        _style_table_param = _raw.y_callig_embedder.embedding_table.weight
        _style_anchor_k = int(getattr(_raw.y_callig_embedder, 'k_clusters', 1))
        _cep = getattr(args, "callig_emb_pretrained", "") or ""
        if _cep:
            if not os.path.isabs(_cep) and not os.path.exists(_cep):
                _cep = os.path.join("/root/Workspace/xy/DiT", _cep)
            _d = torch.load(_cep, map_location="cpu", weights_only=False)
            if _style_anchor_mode == "mean":
                _tgt = _d.get("pair_mean")
                if _tgt is None:
                    raise SystemExit(
                        "[style-anchor] mode=mean 需要 pt 文件里有 'pair_mean' "
                        "(87, D) —— 用 tools/build_multistyle_k4.py 重新生成")
                _tgt = _tgt.float()
            else:
                _tgt = (_d["embedding"] if isinstance(_d, dict) else _d).float()
            _style_anchor_target = _tgt.to(device)
            del _d
            if rank == 0:
                logger.info(f"[style-anchor] λ={_style_anchor_w} mode={_style_anchor_mode} "
                            f"锚到预训练目标 {tuple(_tgt.shape)} ({_cep})")
        elif rank == 0:
            logger.warning("[style-anchor] style_anchor_weight>0 但无 callig_emb_pretrained, 锚定不生效")
    # ★ [2026-09-21] 原先这里有一段"提前把步数计数器归零"的代码（2026-09-19 加的），
    #   但它引用的 resume_start_step 要到 optim 之后才定义 —— 一跑就是
    #   UnboundLocalError，所以 --fresh-scheduler 这个开关**从来没被真正跑通过**。
    #   归零逻辑已统一挪到下面恢复块里（那里在 scheduler 构建之前，语义相同）。

    _has_pretrained = args.pretrained is not None
    if _has_pretrained:
        requires_grad(model, False)
        train_cond_head = getattr(args, 'train_cond_head', True)
        for name, param in model.named_parameters():
            if ('y_callig_embedder' in name or 'y_char_embedder' in name
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
            # [INFRA 2026-09-16] pattern_matcher guard
            # torch 2.1.2 的 inductor pattern_matcher 在整图 replacement 路径上有一个
            # 无 visited 集保护的指数级递归 (torch/_inductor/pattern_matcher.py:652-655
            # percolate_tags 自递归), 整图编译时把编译卡死: 实测 >30min 无任何输出,
            # faulthandler 栈为上万层同一函数。关闭后同一张图 ~65s 编完, 实测 step
            # 时间无损失 (181.1ms vs 181.2ms)。如需恢复 pattern 融合: DIT_PATTERN_MATCHER=1
            if os.environ.get("DIT_PATTERN_MATCHER", "0") != "1":
                from torch._inductor import config as _ind_cfg
                _ind_cfg.pattern_matcher = False
                logger.info("[compile] pattern_matcher=False (规避 torch 2.1.2 percolate_tags 递归爆炸)")
            logger.info(f"[compile] torch.compile(mode={_compile_mode}) 注入（DDP 之前）...")
            model = torch.compile(model, mode=_compile_mode)
    ema_model = None
    if getattr(args, "use_ema", False):
        ema_model = copy.deepcopy(model).eval()
        requires_grad(ema_model, False)
        if _resume_full_ckpt is not None and _resume_full_ckpt.get("ema") is not None:
            # strict=False: 模型新增模块 (如 callig_spatial) 在旧 ckpt EMA 里没有 keys;
            # 这些 keys 会走 _LRScheduler 式的缺省初始化路径, 由 copy.deepcopy(model) 保持零初始化.
            # ★ 2026-09-19: 形状失配的键先剔除 (strict=False 只容忍 key 差异,
            #   容忍不了形状差异 —— v15 换条件头形状在这里崩过一次)。
            _ema_sd = _resume_full_ckpt["ema"]
            _ema_dropped = drop_shape_mismatched(ema_model, _ema_sd)
            if _ema_dropped:
                logger.info(f"[EMA] {len(_ema_dropped)} 个键形状失配, 剔除"
                            f"(EMA 侧保持模型当前初始化): {sorted(_ema_dropped)[:6]} ...")
            _ema_miss, _ema_unexp = ema_model.load_state_dict(_ema_sd, strict=False)
            if _ema_miss:
                logger.info(f"[EMA] missing (new modules, kept init): {sorted(set(k.split('.')[0] for k in _ema_miss))[:8]}")
            logger.info("[EMA] restored EMA weights from checkpoint")
        logger.info(f"[EMA] enabled with decay={args.ema_decay}, interval={getattr(args, 'ema_interval', 1)} "
                    f"(effective per-step decay={args.ema_decay ** max(1, int(getattr(args, 'ema_interval', 1)))})")
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
        logger.info("[flow] Flow-Matching enabled (velocity target, Euler ODE sampling, t in [0,1])")
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

    # 2026-09-05: REPA 不强制 VAE —— 配了 latent_shards_dir 时用预编码 latent,
    # REPA 只需 GT 像素图(喂 DINO teacher), 不需要 vae.encode。
    _need_vae = (not bool(getattr(args, "latent_shards_dir", None)))
    if _need_vae:
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
    else:
        vae = MockVAE(device)
        logger.info("[infra] VAE skipped (latent-only mode) -> saves ~500MB VRAM")

    # ---- 统一 REPA (公共 infra src.loss.repa) ----
    # 支持多层 (repa_layers="8,11"/"8") + warmup 渐进, 与后训练(train_controlnet)完全一致。
    repa_loss_fn = None
    _repa_layers = getattr(args, "repa_layers", "") or ""
    if args.w_repa > 0:
        try:
            layers = tuple(int(x) for x in str(_repa_layers).split(",") if x.strip()) or (8,)
        except Exception:
            layers = (8,)
        try:
            _proj = model.x_embedder.proj
            # PatchEmbed.proj 是 Conv2d (out_channels), 非 Linear (out_features)
            student_hidden_size = int(getattr(_proj, "out_channels",
                                              getattr(_proj, "out_features", 0)))
            if student_hidden_size <= 0:
                raise ValueError("cannot infer student dim from x_embedder.proj")
        except Exception:
            student_hidden_size = 384
        from src.loss.repa import build_repa_module
        # REPA teacher 特征离线缓存: 命中免 DINO 前向, miss 兜底 (lazy teacher)
        _feature_cache = None
        _repa_cache_dir = str(getattr(args, "repa_cache_dir", "") or "")
        if _repa_cache_dir:
            try:
                from src.utils.dino_cache import DinoFeatureCache
                _feature_cache = DinoFeatureCache(_repa_cache_dir)
                logger.info(f"[repa-cache] {_feature_cache.stats()}")
            except Exception as _e:
                logger.warning(f"[repa-cache] failed to load ({_e!r}) -> teacher forward every step")
        repa_loss_fn = build_repa_module(
            student_dim=student_hidden_size, layers=layers,
            teacher_ckpt=getattr(args, "repa_teacher_ckpt", "") or None,
            w_repa=float(args.w_repa),
            warmup_steps=int(getattr(args, "repa_warmup", 0) or 0),
            device=device, feature_cache=_feature_cache)
        logger.info(f"Initializing unified REPA (Teacher: dinov2_vits14, "
                    f"Student Dim: {student_hidden_size}, layers={layers}, "
                    f"w={args.w_repa}, warmup={getattr(args, 'repa_warmup', 0)}, "
                    f"cache={'yes' if _feature_cache is not None else 'no'})")

    # ---- 冻结风格排序头 loss (2026-09-25 冒烟): 全冻结, added=0 ----
    # loss = 1 - cos(f(pred_xstart), centroid[y_pair]); 只在 t∈[t_min,t_max] 生效。
    _style_rank_loss_fn = None
    if getattr(args, 'w_style_rank', 0.0) > 0:
        import importlib
        _srm = importlib.import_module('src.loss.style_rank_module')
        _style_rank_loss_fn = _srm.StyleRankLoss(
            ckpt=getattr(args, 'style_rank_ckpt', 'assets/style_enc_latent.pt'),
            cent_npy=getattr(args, 'style_rank_cent', 'assets/rank_cent87.npy'),
            t_min=float(getattr(args, 'style_rank_t_min', 0.05)),
            t_max=float(getattr(args, 'style_rank_t_max', 0.25)))
        _style_rank_loss_fn.to(device)
        logger.info("[style-rank] enabled: w=%.4f t=[%.2f,%.2f] cent=%s added=0"
                    % (args.w_style_rank, _style_rank_loss_fn.t_min,
                       _style_rank_loss_fn.t_max, tuple(_style_rank_loss_fn.cent.shape)))

    # ---- 实例骨架结构 loss: 冻结 probe (新增可训练参数 0) ----
    # 形态对比见 src/train/latent_structure.py:LatentSkelStructureLoss 的 docstring。
    # target = batch['skel_latent'], **必须指向实例骨架**; 若指到标准字形就是纯重复 g, 必败。
    _skel_struct_loss_fn = None
    _SKEL_STRUCT_WARNED = False
    if getattr(args, 'w_latent_skel', 0.0) > 0:
        # ⚠ 硬拦截: target 绝不能是标准字形骨架。
        # 当前数据集只暴露 `skel_latent`(来自 skel_latent_shards_dir)。v12 该目录是
        # **shards_std** 且 skel_as_glyph_cond=True -> batch['skel_latent'] 就是 g 本身。
        # 拿它当结构 target = 纯重复条件信号 = doc60 §6 那条纪律的翻版, 必败且静默
        # (loss 会正常下降, 因为预测 g 很容易)。故显式拒绝启动, 不给人跑废一整轮的机会。
        if getattr(args, 'w_latent_skel', 0.0) > 0 and not (
                getattr(args, 'inst_skel_shards_dir', '') or '').strip():
            # ⚠ 硬拦截升级 (2026-09-16): v12 的 skel_latent_shards_dir 是 **shards_std**
            # 且 skel_as_glyph_cond=True -> batch['skel_latent'] 就是 g 本身。拿它当结构
            # target = 纯重复条件信号 = 必败且静默 (loss 会正常下降, 因为预测 g 很容易)。
            # 现在 target 必须来自独立的 batch['inst_skel'](实例骨架, GT 图派生, 与 g 解耦),
            # 故要求显式配置 --inst-skel-shards-dir。不给人跑废一整轮的机会。
            raise ValueError(
                "--w-latent-skel>0 需要 --inst-skel-shards-dir 指向**实例骨架** latent shards "
                "(GT 图 -> skeletonize -> 3px 膨胀 -> VAE encode, 由 "
                "tools/build_skel_latents.py 生成; 与条件 g 完全解耦)。\n"
                "注意: skel_latent_shards_dir=shards_std + skel_as_glyph_cond=True 时 "
                "batch['skel_latent'] 就是 g 本身, 不能当结构 target。\n"
                "先跑 _sync_work/inst_skel_pipeline.sh (stage1 建 shards + stage2 训 probe)。")

        _pcfg = getattr(args, 'latent_skel_probe', '') or ''
        if not _pcfg or not os.path.exists(_pcfg):
            raise ValueError(
                f"--w-latent-skel > 0 需要 --latent-skel-probe 指向训好的 probe ckpt, "
                f"当前={_pcfg!r}。用 tools/train_latent_skel_probe.py 训练。")
        from src.train.latent_structure import LatentSkelProbe, LatentSkelStructureLoss
        _pck = torch.load(_pcfg, map_location="cpu", weights_only=False)
        _pargs = _pck.get("args", {}) if isinstance(_pck, dict) else {}
        _probe = LatentSkelProbe(
            in_channels=int(_pargs.get("in_channels", 4)),
            out_channels=int(_pargs.get("out_channels", 4)),
            width=int(_pargs.get("width", 64)),
            depth=int(_pargs.get("depth", 3)))
        _psd = _pck.get("model", _pck)
        _missing, _unexp = _probe.load_state_dict(_psd, strict=False)
        if _missing or _unexp:
            raise ValueError(f"probe ckpt 不匹配: missing={_missing} unexpected={_unexp}")
        _probe.to(device).eval().requires_grad_(False)
        _skel_struct_loss_fn = LatentSkelStructureLoss(
            probe=_probe, max_t=float(getattr(args, 'latent_skel_max_t', 0.3)))
        logger.info(f"[skel-struct] enabled: w={args.w_latent_skel}, probe={_pcfg}, "
                    f"max_t={_skel_struct_loss_fn.max_t}, metrics={_pck.get('metrics', {})}, "
                    f"trainable_params_added=0 (probe frozen)")

    # ---- 中程结构 aux loss (独立模块 src/loss/structure_mid.py, 替代旧内联 std_mid) ----
    # 旧内联: MSE(x0_pred, 细骨架 g) —— 与中程"软墨迹"形态冲突。新模块: 载体粗化
    # (blur GT / dilate skel, σ(t)/k(t) 由 calibrate_mid_structure.py 校准) + 低通子空间
    # + 通道归一 -> k/σ 不敏感。默认 w_std_mid=0 关闭, 零破坏。
    _mid_struct_loss = None
    if getattr(args, 'w_std_mid', 0.0) > 0:
        from src.loss.structure_mid import MidStructureLoss, TSchedule
        _sched = TSchedule.from_json(
            getattr(args, 'std_mid_calib_json', '') or '',
            sigma_slope=float(getattr(args, 'std_mid_sigma_slope', 4.0)),
            k_slope=float(getattr(args, 'std_mid_k_slope', 2.0)))
        _mid_struct_loss = MidStructureLoss(
            carrier=getattr(args, 'std_mid_carrier', 'blur_gt'),
            alo=float(getattr(args, 'std_mid_alo', 0.35)),
            ahi=float(getattr(args, 'std_mid_ahi', 0.75)),
            lp_factor=int(getattr(args, 'std_mid_lp', 2)),
            sigma_sched=_sched,
            latent_channels=int(getattr(args, 'latent_channels', 4)))
        if rank == 0:
            logger.info(f"[std_mid] MidStructureLoss carrier={_mid_struct_loss.carrier} "
                        f"window a_t∈[{_mid_struct_loss.alo},{_mid_struct_loss.ahi}] "
                        f"lp={_mid_struct_loss.lp_factor} "
                        f"calib={getattr(args, 'std_mid_calib_json', '') or '(默认线性σ/k)'}")

    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    if repa_loss_fn is not None:
        trainable_params_list.extend(repa_loss_fn.trainable_params())

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
        # [OPTIM-FUSED 2026-09-16] AdamW fused (单内核多张量更新, 省 elementwise 带宽)。
        # 守卫: 仅 CUDA 且 torch 支持 fused 时启用, 否则退回默认实现。
        _fused = bool(torch.cuda.is_available()) and float(torch.__version__[:3]) >= 2.0
        try:
            opt = torch.optim.AdamW(trainable_params_list, lr=args.lr,
                                    weight_decay=args.weight_decay, fused=_fused)
        except TypeError:
            opt = torch.optim.AdamW(trainable_params_list, lr=args.lr, weight_decay=args.weight_decay)
            _fused = False
        logger.info(f"[optim] AdamW lr={args.lr} wd={args.weight_decay}")

    # ★ 形变头单独一组 lr（v18-skelnet）。
    #   为什么单独一组: 它是在**另一个目标**(aux_skel3)上预训练好的, 已经在一个好状态;
    #   扩散 loss 对它是"扰动信号"而非有用信号 -> 用主 lr 会在头几百步破坏它。
    #   LambdaLR 按各组的**初始 lr** 等比缩放 -> 单独设初值即可自动跟随 cosine。
    if getattr(model, "deform_skel", None) is not None:
        _dtr = bool(int(getattr(args, "deform_trainable", 1)))
        for _p in model.deform_skel.parameters():
            _p.requires_grad_(_dtr)
        if _dtr:
            _scale = float(getattr(args, "deform_lr_scale", 0.1))
            _did = {id(p) for p in model.deform_skel.parameters()}
            _pg0 = opt.param_groups[0]
            _rest = [p for p in _pg0["params"] if id(p) not in _did]
            _dm = [p for p in _pg0["params"] if id(p) in _did]
            if _dm:
                _pg0["params"] = _rest
                _new = {k: v for k, v in _pg0.items() if k != "params"}
                _new["params"] = _dm
                _new["lr"] = args.lr * _scale
                opt.add_param_group(_new)
                logger.info(f"[optim] deform_skel 单独一组: lr={args.lr * _scale:.2e} "
                            f"({_scale}x 主 lr), {sum(p.numel() for p in _dm):,} 参数")
        else:
            logger.info("[optim] deform_skel **冻结**（--deform-trainable 0）")

    # Restore optimizer state + step counter for full resume. If --resume-lr is given,
    # override the LR so we can test whether a smaller LR avoids the NaN.
    # ★ [2026-09-21] --fresh-scheduler: 只训**与预训练参数不重合**的新模块（典型: few-shot
    #   往书家表里加的第 88 行）。此时旧优化器状态对新参数毫无意义，而旧步数计数器会把
    #   新 LR 调度推到旧 cosine 的尾部（甚至 max_steps 比较直接判"已完成"退出）。
    #   所以这条路径下**不恢复优化器状态、步数计数器保持 0**，LR 按 max_steps 从 step 0
    #   走完整条调度（few-shot 的"和预训练一样有衰减"就该这么来）。
    resume_start_step = 0
    if _resume_full_ckpt is not None and getattr(args, 'fresh_scheduler', False):
        logger.info("[resume-full] fresh-scheduler: 不恢复优化器状态、步数计数器保持 0 "
                    f"(LR 按 max_steps={args.max_steps} 从 step 0 重新调度; "
                    f"ckpt 里原步数 {_resume_full_ckpt.get('train_steps', '?')} 已忽略)")
    elif _resume_full_ckpt is not None:
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
        # Recover the step counter, 优先级:
        #   1) ckpt **顶层** "train_steps"  —— 唯一权威: 存盘时由真实步数写入(见下方
        #      checkpoint["train_steps"]), 与文件名无关, 任何文件名的 ckpt 都对。
        #      [2026-09-16] 此前**从未被读取**, 只靠文件名数字推测 -> 跨阶段备份用的
        #      无时间戳名字会被解析错, 例: "v12_S2_cat_last.pt" 的 digits=['12','2']
        #      -> resume_start_step=2 -> LR 退回 warmup 起点(≈6.7e-8, 看着像卡死),
        #      且 ckpt 从 0001000.pt 重新编号会**覆盖历史 ckpt**。
        #   2) 文件名末尾数字 (0055000.pt -> 55000), 兼容旧 ckpt (无顶层字段)。
        #   3) 保存的 args.train_steps —— 兜底 (历史 comment 认为该字段不存在)。
        import re as _re
        _top_steps = _resume_full_ckpt.get("train_steps", None)
        _fname = os.path.basename(str(args.resume_full))
        _digits = _re.findall(r"\d+", _fname)
        _ckpt_args = _resume_full_ckpt.get("args", None)
        _args_steps = (getattr(_ckpt_args, "train_steps", None)
                       if _ckpt_args is not None else None)
        _src = None
        if _top_steps is not None:
            resume_start_step, _src = int(_top_steps), "ckpt 顶层 train_steps"
        elif _digits:
            resume_start_step, _src = int(_digits[-1]), f"文件名 {_fname}"
        elif _args_steps is not None:
            resume_start_step, _src = int(_args_steps), "保存的 args.train_steps"
        if _src is not None:
            logger.info(f"[resume-full] start step={resume_start_step} (来源: {_src})")
            if _digits and int(_digits[-1]) != resume_start_step:
                logger.warning(
                    f"[resume-full] 文件名数字 {_digits[-1]} != 真实步数 "
                    f"{resume_start_step} —— 已按真实步数走 (文件名仅供人看, 不可信)")

    # bf16 training: run the model in bf16 autocast (no loss scaling needed — bf16 has
    # the same exponent range as fp32, so it does not overflow like fp16 AMP). VAE and
    # structural losses stay fp32 for numerical stability.
    use_latent = bool(getattr(args, "latent_shards_dir", None))

    if use_latent:
        dataset = MCCDLatentDataset(csv_file=args.data_csv,
                                    latent_shards_dir=args.latent_shards_dir,
                                    img_root=args.img_root,
                                    image_size=args.image_size,
                                    preload=bool(getattr(args, 'preload', False)),
                                    load_image=(args.w_repa > 0),
                                    num_preload_workers=int(getattr(args, 'preload_workers', 16)),
                                    use_glyph_cond=getattr(args, 'w_glyph_cond', False),
                                    skel_latent_shards_dir=(args.skel_latent_shards_dir
                                                            if getattr(args, 'skel_as_glyph_cond', False)
                                                            else None),
                                    # ★ 条件增强: 多个骨架几何变体目录（逗号分隔）。
                                    #   第一个必须是未扰动的原始条件。训练时每样本每步随机选一个。
                                    skel_latent_shards_dirs=(
                                        [d.strip() for d in str(
                                            getattr(args, 'skel_latent_shards_dirs', '') or ''
                                        ).split(',') if d.strip()]
                                        if getattr(args, 'skel_as_glyph_cond', False) else None),
                                    # ⚠ 这里必须把 w_deform_skel 也算进去: 否则开了形变监督
                                    #   但 w_latent_skel/w_std_mid 都为 0 时, batch['inst_skel']
                                    #   根本不会被加载 -> 中间监督静默失效(本项目反复踩的接线缺口)。
                                    inst_skel_shards_dir=(
                                        (getattr(args, 'inst_skel_shards_dir', '') or None)
                                        if (getattr(args, 'w_latent_skel', 0.0) > 0
                                            or getattr(args, 'w_std_mid', 0.0) > 0
                                            or getattr(args, 'w_deform_skel', 0.0) > 0)
                                        else None),
                                    callig_id_map=getattr(args, '_callig_map', None),
                                    callig_script_map=getattr(args, '_callig_script_map', None),
                                    aux_latent_shards_dirs=aux_dirs_of(args))
        logger.info("Using latent-cached dataset (skip on-the-fly VAE encode)."
                    + (" preload=ON" if getattr(args, 'preload', False) else ""))
        _aux_dirs = aux_dirs_of(args)
        if _aux_dirs:
            _aw = getattr(args, 'aux_loss_weights', '') or getattr(args, 'aux_loss_weight', 1.0)
            logger.info(f"[aux] target=12ch+ dirs={_aux_dirs} per-group weights={_aw}")
    else:
        dataset = MCCDDataset(csv_file=args.data_csv, root_dir=args.data_dir, image_size=args.image_size)
    # ── 单一刻度: ckpt_every 同时定义 epoch 长度 / 存盘点 / 评测点 ──────────
    # [2026-09-16] epoch 长度 = **固定步数**（而不是"数据集一遍"）。
    #   起因: 原先 epoch = len(dataset)//batch = 119 步(~21s)，每 119 步 DataLoader
    #   就要重建一次迭代器；而 ckpt + eval 落在 train_steps % ckpt_every 上，
    #   两者互不对齐 -> iterator reset 成为一份**独立**代价，Steps/Sec 出现规律锯齿。
    #
    # ★ 2026-09-17 合并: 原先 `epoch_steps` 与 `ckpt_every` 是两个独立参数，
    #   训练循环里还要用 `min()` 拼出"实际存盘周期"，deferred eval 还得再断言
    #   一次"ckpt 周期整除 eval 周期"。既然**语义上它们本就该相等**（都是"tick"），
    #   干脆合并成一个 —— 于是：
    #     · 存盘条件退化成 `train_steps % _epoch_steps == 0`（一行）
    #     · eval 点 == 存盘点 == epoch 边界，**恒成立**，无需任何断言
    #     · deferred eval 的前置条件自动满足
    #   实测: 121 个配置用 ckpt_every、6 个两者都配且**全部相等**、0 个只配 epoch_steps
    #   -> 保留 ckpt_every 作为唯一参数，**零配置迁移**。
    #   `epoch_steps` 降级为**兼容别名**: 配了就必须与 ckpt_every 相等，否则拒绝启动。
    _epoch_steps = int(getattr(args, 'ckpt_every', 0) or 0)
    _es_alias = int(getattr(args, 'epoch_steps', 0) or 0)
    if _es_alias > 0:
        if _epoch_steps == 0:
            _epoch_steps = _es_alias          # 只配了 epoch_steps 的旧配置
        elif _es_alias != _epoch_steps:
            raise SystemExit(
                f"[epoch] epoch_steps={_es_alias} 与 ckpt_every={_epoch_steps} 不一致。\n"
                f"  两者已合并为**同一个刻度**（epoch 长度 = 存盘周期 = 评测周期），\n"
                f"  不再支持取不同值。请只保留 ckpt_every，或让两者相等。")
    if _epoch_steps < 0:
        _epoch_steps = 0                      # 显式退回旧行为(epoch = 数据集一遍)
    # ckpt_every = 0 -> 不存盘, 且 epoch 退回"数据集一遍"(旧行为)

    # deferred 模式下两处 eval 块整段跳过（见 --eval-mode 的 help）。
    # 注: 不再需要"ckpt 周期整除 eval 周期"的护栏 —— 合并刻度后该条件**恒成立**。
    _EVAL_INLINE = str(getattr(args, 'eval_mode', 'inline')) == "inline"
    if args.sampler == "factor_balanced":
        sampler = DistributedFactorBalancedSampler(
            dataset, num_replicas=dist.get_world_size(), rank=rank, seed=args.global_seed,
            char_alpha=args.balance_char_alpha,
            callig_alpha=args.balance_callig_alpha)
        logger.info(f"Using factor-balanced sampler: {sampler.summary()} "
                    f"(char_alpha={args.balance_char_alpha}, "
                    f"callig_alpha={args.balance_callig_alpha})")
    else:
        if _epoch_steps > 0:
            sampler = LongEpochDistributedSampler(
                dataset,
                steps_per_epoch=_epoch_steps,
                batch_size=int(args.global_batch_size // dist.get_world_size()),
                num_replicas=dist.get_world_size(),
                rank=rank,
                seed=args.global_seed,
            )
            _samples_per_epoch = (_epoch_steps
                                  * (args.global_batch_size // dist.get_world_size())
                                  * dist.get_world_size())
            logger.info(
                f"[sampler] LongEpoch: epoch = {_epoch_steps} 步 "
                f"(= {_samples_per_epoch:,} 样本 = 数据集的 "
                f"{_samples_per_epoch / max(len(dataset), 1):.1f} 倍); "
                f"迭代器每 {_epoch_steps} 步 reset 一次, 与 ckpt_every 对齐。")
            if resume_start_step > 0:
                _inner = sampler.inner_epoch_for_step(resume_start_step)
                sampler.set_start_inner_epoch(_inner)
                logger.info(f"[sampler] resume 数据流对齐: step {resume_start_step} "
                            f"-> 已消耗 {_inner} 个 randperm 份, 从第 {_inner} 份继续")
        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=rank,
                shuffle=True,
                seed=args.global_seed
            )
    # [INFRA 2026-09-18 实测校正] 曾以为 "preload 下数据已在内存, num_workers 只剩 IPC
    #   开销, 应强制 0" —— **错**。4090 上合成基准 (_review/bench_dataloader_preload.py,
    #   batch=360 / uint8 图像, 复现 preload 返回形态): preload 下 workers 反而更快 ——
    #   num_workers 0→68ms/批, 4→40ms, 8→24ms。原因: __getitem__+default_collate 要对每批
    #   360 样本各做一次 from_numpy 切片 + stack 拷贝 + pin(~71MB/批), workers 把这份
    #   collate **并行化**并与 pin 线程重叠, 收益远超 IPC。故**保留配置的 num_workers**,
    #   不再对 preload 强制 0。
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
    if (args.sampler == "random" and _epoch_steps > 0
            and len(loader) != _epoch_steps):
        logger.warning(f"[sampler] len(loader)={len(loader)} != epoch_steps={_epoch_steps}"
                       f" (整除性/drop_last?) —— epoch 与 ckpt 可能错位")
    _epochs_needed = args.epochs
    if _epoch_steps > 0 and args.max_steps > 0:
        # epoch 已退化为 ckpt 刻度: 只需跑到 max_steps 所需的份数
        _epochs_needed = min(args.epochs, -(-args.max_steps // _epoch_steps))

    if args.max_steps > 0:
        total_planned_steps = args.max_steps
    elif _epoch_steps > 0:
        total_planned_steps = _epochs_needed * len(loader)
    else:
        total_planned_steps = args.epochs * len(loader)
    if getattr(args, 'fresh_scheduler', False) and _resume_full_ckpt is not None and args.max_steps > 0:
        # 调度器按绝对步数(从 step 0)计算: resume 点落在 cosine 中段, 不是从头 warm restart
        total_planned_steps = args.max_steps
        for _pg in opt.param_groups:
            _pg.pop('initial_lr', None)  # 丢弃 ckpt 里旧 LambdaLR 的 base, 让 resume_lr 覆盖生效
        logger.info(f"[LR] fresh-scheduler: cosine computed from absolute step 0 "
                    f"(total {total_planned_steps}); resume at {resume_start_step} "
                    f"-> LR starts mid-decay at "
                    f"{args.min_lr_ratio + (1.0 - args.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * max((resume_start_step - min(args.warmup_steps, total_planned_steps - 1)) / max(total_planned_steps - min(args.warmup_steps, total_planned_steps - 1), 1), 0.0))):.2e} (base)")
    scheduler = None
    if args.lr_schedule == "cosine":
        warmup_steps = min(args.warmup_steps, max(total_planned_steps - 1, 0))
        _step_offset = resume_start_step if (getattr(args, 'fresh_scheduler', False)
                                             and _resume_full_ckpt is not None) else 0

        def _lr_scale(step):
            step = step + _step_offset
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

    # ★ 2026-09-19: --fresh-scheduler 时**同时把步数计数器归零**。
    #   否则 stage2 传过来的 train_steps (如 160000) 会:
    #   1) 与 max_steps=40000 比较 -> 160000 >= 40000 -> 立即退出
    #   2) 通过 _step_offset 把 LR 调度推到 cosine 末端 -> LR = min_ratio * base (≈3e-6)
    #   "fresh scheduler" 的正确语义 = **当新训练开始**:
    #   权重来自 ckpt, 但步数/LR/scheduler 全部归零。
    if (getattr(args, 'fresh_scheduler', False)
            and _resume_full_ckpt is not None):
        logger.info(f"[resume-full] fresh-scheduler: 步数计数器归零 "
                    f"(原 {resume_start_step} -> 0), max_steps={args.max_steps} 从 0 起算")
        resume_start_step = 0

    model.train()

    train_steps = resume_start_step
    log_steps = 0
    # ★ 2026-09-17: 累加器从 Python float 改成 **GPU 0-dim 张量**。
    #   原来每步 `.item()` 取标量再累加 = 每步多次全 GPU 同步（且都在 backward 之前，
    #   打断 CPU run-ahead）。现在每步只做一次张量加法（不出 GPU），
    #   标量化推迟到日志步（每 log_every 步一次）。
    #   dtype=float64: 原来累加的是 Python float（float64）。用 float32 累加 log_every
    #   个值会累积舍入（实测 5 项就差 2.2e-8），虽然显示到 4 位看不出来，但没必要改语义。
    #   0-dim 张量用 float64 无任何成本。
    running_loss = torch.zeros((), device=device, dtype=torch.float64)
    running_diff = torch.zeros((), device=device, dtype=torch.float64)
    running_repa = torch.zeros((), device=device, dtype=torch.float64)
    running_std_mid = torch.zeros((), device=device, dtype=torch.float64)
    running_skel = torch.zeros((), device=device, dtype=torch.float64)
    _acc_c12 = None          # (C,) 逐通道 MSE 累加，日志步再拆成 image/canny/skel
    # torch.compile warmup 结束后回收显存空洞（见 cli.py --empty-cache-after-warmup）
    _empty_cache_at = int(getattr(args, "empty_cache_after_warmup", 0) or 0)
    nan_steps = 0
    current_ema_decay = args.ema_decay
    start_time = time()

    # ---- 早停 (实现见 src/train/early_stop.py) ----
    # 判据来自 ckpt 目录下评测写出的 eval_auto_<step>.json。
    _stopper = EarlyStopper(args, checkpoint_dir, logger)
    early_stop_stopped = False
    _es_check_every = _stopper.check_every

    logger.info(f"Training for {args.epochs} epochs...")

    # in-process GPU eval 已停用（见文件顶部 `_HAS_IN_PROCESS_EVAL` 的说明）。
    # 当前在训评测走 `--in-mem-eval`，实现与缓存都在 src/eval/in_mem_eval.py。

    # ── aux 通道权重：循环不变量，只解析/构造一次 ──────────────────────────
    # 原来这段（字符串解析 + 3 处校验 + _ch_w 的 GPU 分配与切片赋值）在**每步**的
    # 数据分支里重做一次，纯属浪费（切片赋值还是 kernel launch）。
    # 校验在这里 fail-fast：配了 aux 却没给权重就直接拒绝启动。
    _aux_dirs, _aux_w_list = resolve_aux_channel_weights(args)
    _ch_w = None
    if _aux_dirs:
        _ch_w = build_aux_channel_weights(
            _aux_dirs, _aux_w_list, int(getattr(args, 'latent_channels', 4)), device)
        logger.info(f"[aux] {len(_aux_dirs)} 组 aux, 权重 {_aux_w_list} "
                    f"-> in_channels={_ch_w.numel()}, "
                    f"image_channels={getattr(args, 'image_channels', None) or getattr(args, 'latent_channels', 4)}")

    # ── 训练循环前的显存基线（2026-09-17）─────────────────────────────────
    # 用来回答"torch.compile 到底占了多少"。之后每步的 [alloc] 行可以与之对比，
    # 定位 6G 空洞是在哪一步、由什么累积起来的。
    # 已知: xattn@240 的 reserved 从 step50 起就是 20.51G，而 eval 的 empty_cache()
    #   能把它降到 14.42G -> 说明有 6.09G 是**缓存但未被使用**的。
    # 但"是谁在 warmup 期间分配了这 6G"**尚未查明**（`mode=default` 不做 Triton
    #   autotune，所以不是 autotune 缓冲 —— 我之前的说法是错的）。
    if rank == 0:
        logger.info(
            f"[alloc] 循环前基线: reserved {torch.cuda.memory_reserved() / 2 ** 30:.2f}G | "
            f"活跃 {torch.cuda.memory_allocated() / 2 ** 30:.2f}G | "
            f"高水位 {torch.cuda.max_memory_reserved() / 2 ** 30:.2f}G")

    # ── 纯评测模式：加载 ckpt -> 评一次 -> 退出，**不进训练循环** ──────────
    # 与"resume 再跑几千步等 epoch 落点"的区别：这里一步都不训，
    # 不污染 step 计数 / LR 调度 / optimizer / EMA。
    if getattr(args, 'eval_only', False):
        if rank == 0:
            _ev_step = int(train_steps)
            logger.info(f"[eval-only] 只评测不训练：step={_ev_step}, "
                        f"权重={'ema' if ema_model is not None else 'model'}, "
                        f"sets={getattr(args, 'in_mem_eval_sets', '')!r}")
            _m = ema_model if ema_model is not None else model
            _m.eval()
            try:
                # ⚠ run_in_mem_eval 的 logger 形参要的是**可调用对象**（默认 print），
                #   传 Logger 实例会 TypeError: 'Logger' object is not callable。
                _res = run_in_mem_eval(
                    _m, args, _ev_step, device,
                    os.path.join(checkpoint_dir, "..", "eval_only"),
                    logger=logger.info)
                logger.info(f"[eval-only] 完成: {_res}")
            except Exception as _e:
                logger.error(f"[eval-only] 失败: {type(_e).__name__}: {_e}")
                import traceback as _tb
                logger.error(_tb.format_exc())
                raise
        if dist.is_initialized():
            dist.barrier()
        logger.info("[eval-only] 退出")
        return

    for epoch in range(_epochs_needed):
        sampler.set_epoch(epoch)
        if rank == 0:
            # [2026-09-16] epoch 已退化为"ckpt/eval 刻度"(见 --epoch-steps), 且续训时
            # 外层 epoch 计数器从 0 重新开始 -> 直接乘 epoch 会算错区间号(盯着日志判断
            # 进度时会误判)。这里用**真实已训步数**算, 续训后依旧正确。
            _span = (f" (steps {resume_start_step + epoch * _epoch_steps + 1}"
                     f"-{resume_start_step + (epoch + 1) * _epoch_steps})"
                     if _epoch_steps > 0 else "")
            logger.info(f"Begin epoch {epoch}{_span} — ckpt+eval 落点...")
        
        try:
            for batch_idx, batch in enumerate(loader):
                # ★ non_blocking=True: DataLoader 已开 pin_memory=True，但**必须显式传**
                #   这个参数才会走异步拷贝；否则即使源在 pinned 内存里也是同步拷贝，
                #   主机线程会被阻塞（图像单批 ~283MB，实测代价可观）。见 docs/system/70 §8.6。
                y_callig = batch['y_callig'].to(device, non_blocking=True)
                y_char = batch['y_char'].to(device, non_blocking=True)
                if cond_mode == "3cond":
                    y_script = batch['y_script'].to(device, non_blocking=True)

                # 注意: `_ch_w` 与 aux 权重已在**循环外**解析好（见 resolve_aux_channel_weights），
                # 这里不再重算，也不要把它重置为 None。
                if 'latent' in batch:
                    # Latent-cached training: latent pre-encoded (scaled by vae_scaling_factor).
                    x_latent = batch['latent'].to(device, non_blocking=True)
                    # moyi 式辅助目标通道: aux latents (skel/canny) 与图像 latent 拼成扩散目标
                    _aux = batch.get('aux_latents', None)
                    if _aux is not None and _aux.numel() > 0:
                        x_latent = torch.cat(
                            [x_latent, _aux.to(device, non_blocking=True).float()], dim=1)
                    x = batch.get('image', None)
                    x = x.to(device, non_blocking=True) if x is not None else None
                else:
                    x = batch['image'].to(device, non_blocking=True)
                    # dataset 现在给的是 uint8（省 H2D 流量），VAE 需要 [-1,1] float
                    if x.dtype == torch.uint8:
                        x = x.float().div_(255.0).mul_(2.0).sub_(1.0)
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
                # ── S2: 三层语义分解的三个 id（数据集始终提供，模型按需消费）──
                # hier_style=0 时 forward 会忽略它们，零副作用。
                if int(getattr(args, 'hier_style', 0)) > 0:
                    model_kwargs['y_callig_raw'] = batch['y_callig_raw'].to(
                        device, non_blocking=True)
                    model_kwargs['y_pair'] = batch['y_pair'].to(device, non_blocking=True)
                    model_kwargs['y_script'] = batch['y_script'].to(device, non_blocking=True)
                # 标准字形条件 g(甲2 token-add): batch 由 dataset 提供, None=禁用对应项
                if getattr(args, 'w_glyph_cond', False) and 'g' in batch and batch['g'].numel() > 0:
                    model_kwargs['g'] = batch['g'].to(device, non_blocking=True)  # (N,4,32,32)
                elif getattr(args, 'skel_as_glyph_cond', False) and 'skel_latent' in batch \
                        and batch['skel_latent'].numel() > 0:
                    # v10a: 实例 skel latent 即字条件 (与 ControlNet 的 cond 同源不同路)
                    model_kwargs['g'] = batch['skel_latent'].to(
                        device, non_blocking=True).float()

                # ── 条件噪声增强 (Condition Noise Augmentation) ───────────
                # 标准骨架 g 是固定的印刷体 latent (cos=0.902 to GT)。加噪声迫使
                # 模型学会从「不完美的结构条件」中提取拓扑信息，避免对 g 精确数值过拟合。
                _gn_scale = float(getattr(args, 'glyph_noise_scale', 0.0))
                _gn_prob = float(getattr(args, 'glyph_noise_prob', 0.0))
                _gpd = float(getattr(args, 'glyph_patch_drop', 0.0))
                if 'g' in model_kwargs and (_gn_scale > 0 or _gpd > 0):
                    _g = model_kwargs['g']
                    if _gn_scale > 0 and _gn_prob > 0:
                        _mask = (torch.rand(_g.shape[0], 1, 1, 1, device=device) < _gn_prob).to(_g.dtype)
                        _scales = torch.rand(_g.shape[0], 1, 1, 1, device=device) * _gn_scale
                        _g = _g + _mask * torch.randn_like(_g) * _scales
                    if _gpd > 0:
                        _pmask = (torch.rand(_g.shape[0], 1, _g.shape[2], _g.shape[3],
                                             device=device) > _gpd).to(_g.dtype)
                        _g = _g * _pmask
                    model_kwargs['g'] = _g
                
                # REPA: 请求多层中间特征 (统一 infra, 多层 dict / 单层兼容)
                if args.w_repa > 0 and repa_loss_fn is not None:
                    if len(repa_loss_fn.layers) > 1:
                        model_kwargs['return_intermediate_layers'] = repa_loss_fn.layers
                    else:
                        model_kwargs['return_intermediate_layer'] = repa_loss_fn.layers[0]

                # Forward pass under bf16 autocast (same exponent range as fp32, no overflow).
                #
                # return_pred_xstart: flow 分支默认**不返回** pred_xstart（避免
                # autograd 图膨胀），但下面 w_std_mid 依赖它。若这里不请求，flow
                # 模式下该机制会因 `loss_dict.get("pred_xstart", None)` 恒为 None 而
                # **静默失效** —— 不报错、loss 正常下降、但从未生效。w_std_mid 正是
                # 「把去噪中段预测的 x0 拉向标准字形 latent」的预训练改进项。
                # gaussian_diffusion 不接受该参数，故用 try/except 兼容。
                _need_x0 = (getattr(args, 'w_std_mid', 0.0) > 0
                            or getattr(args, 'w_latent_skel', 0.0) > 0
                            or getattr(args, 'w_style_rank', 0.0) > 0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if _need_x0:
                        try:
                            loss_dict = diffusion.training_losses(
                                model, x_latent, t, model_kwargs,
                                return_pred_xstart=True, channel_weights=_ch_w)
                        except TypeError:
                            # gaussian_diffusion 不支持该参数，本身就会返回 pred_xstart
                            loss_dict = diffusion.training_losses(
                                model, x_latent, t, model_kwargs)
                    else:
                        loss_dict = diffusion.training_losses(
                            model, x_latent, t, model_kwargs, channel_weights=_ch_w)
                    loss_diff = loss_dict["loss"].mean()

                loss_repa = torch.tensor(0.0, device=device)
                loss_x0lat = torch.tensor(0.0, device=device)
                loss_skel_struct = torch.tensor(0.0, device=device)

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
                _need_x0_grad = (getattr(args, 'w_std_mid', 0.0) > 0
                                 or getattr(args, 'w_latent_skel', 0.0) > 0
                                 or getattr(args, 'w_style_rank', 0.0) > 0)
                pred_xstart_latent = loss_dict.get("pred_xstart", None)
                if pred_xstart_latent is not None and not _need_x0_grad:
                    # No struct loss this run at all — drop the graph immediately.
                    pred_xstart_latent = pred_xstart_latent.detach()
                # Clear the heavy reference inside loss_dict so the dict can't keep
                # the graph alive after we exit this step.
                loss_dict.pop("pred_xstart", None)

                # ---- MIDSTEP_STD (已抽到 src/loss/structure_mid.py: MidStructureLoss) ----
                # 旧内联是 MSE(x0_pred, 细骨架 g), 与中程软墨迹形态冲突 -> 换成
                # 载体粗化 + 低通子空间 + 通道归一。gt=GT图像latent(现成), skel=g。
                loss_std_mid = torch.tensor(0.0, device=device)
                if (_mid_struct_loss is not None
                        and pred_xstart_latent is not None):
                    _lc = int(getattr(args, 'latent_channels', 4))
                    _gt = x_latent[:, :_lc]
                    # 膨胀后的 GT 实例骨架 (20px)。不用标准骨架 g, 那是条件本身。
                    _gsk = batch.get('inst_skel', None)
                    if _gsk is None or _gsk.numel() == 0:
                        if not getattr(_mid_struct_loss, "_empty_warned", False):
                            _mid_struct_loss._empty_warned = True
                            logger.warning(
                                "[std_mid] batch['inst_skel'] 为空, 中程监督未生效。"
                                "需要 --inst-skel-shards-dir 指向膨胀后的 GT 骨架。")
                        loss_std_mid = torch.tensor(0.0, device=device)
                    else:
                        loss_std_mid = _mid_struct_loss(
                            pred_xstart_latent, _gt,
                            _gsk.to(device, non_blocking=True).float(),
                            t.to(device))

                intermediate_feats = loss_dict.get("intermediate_feats", None)
                if x is not None and intermediate_feats is not None and repa_loss_fn is not None and args.w_repa > 0:
                    # 统一 REPA (公共 infra): 多层 dict / 单层张量 + warmup 渐进
                    # img_ids: 配合 --repa-cache-dir 查表 (命中免 DINO 前向)
                    _img_ids = batch.get('img_id', None)
                    loss_repa = repa_loss_fn(intermediate_feats, x, step=train_steps,
                                             img_ids=_img_ids)

                # ---- 实例骨架结构 loss (辅助, 冻结 probe, 零可训练参数) ----
                # 见 src/train/latent_structure.py:LatentSkelStructureLoss 的 docstring:
                # 与 12ch 的区别是"不在扩散目标里" —— 4ch 主干 / CFG 作用域 / 推理成本
                # 全部不受影响, 新增可训练参数 0(当前主要矛盾是过拟合, 这是决定性优势)。
                # target = batch['skel_latent'] **必须是实例骨架**; 若指到标准字形就是纯重复 g。
                if _skel_struct_loss_fn is not None and pred_xstart_latent is not None:
                    # [inst-skel 2026-09-16] target 现在读 **inst_skel**(实例骨架, GT 图派生),
                    # 与条件 g (skel_latent=shards_std) 解耦。不再读 skel_latent。
                    _sk = batch.get('inst_skel', None)
                    if _sk is None or _sk.numel() == 0:
                        if not _SKEL_STRUCT_WARNED:
                            _SKEL_STRUCT_WARNED = True
                            logger.warning(
                                "[skel-struct] w_latent_skel>0 但 batch['inst_skel'] 为空 "
                                "(需配置 --inst-skel-shards-dir 指向实例骨架 shards) "
                                "-> 本项静默失效。这类静默失效已踩过多次, 故只在首次告警。")
                        loss_skel_struct = torch.tensor(0.0, device=device)
                    else:
                        loss_skel_struct = _skel_struct_loss_fn(
                            pred_xstart_latent,
                            _sk.to(device, non_blocking=True).float(),
                            t.to(device))

                loss = (loss_diff
                        + loss_repa  # 统一 REPA: w × (1 - cos) 已在 RepaModule.forward 内含 warmup
                        + getattr(args, 'w_std_mid', 0.0) * loss_std_mid
                        + getattr(args, 'w_latent_skel', 0.0) * loss_skel_struct)

                # ---- style-rank loss (t 门控在模块内) ----
                loss_style_rank = torch.tensor(0.0, device=device)
                if _style_rank_loss_fn is not None and getattr(args, 'w_style_rank', 0.0) > 0:
                    if pred_xstart_latent is not None:
                        _yp = batch['y_pair'].to(device, non_blocking=True)
                        loss_style_rank, _n_style_rank = _style_rank_loss_fn(
                            pred_xstart_latent, _yp, t.to(device))
                        loss = loss + getattr(args, 'w_style_rank', 0.0) * loss_style_rank

                # ---- DeformSkel 中间监督: 形变后的骨架应逼近**该书家的 GT 骨架** ----
                # 这是本项目第一个"自带稠密监督"的风格干预: 目标数据(实例骨架)现成,
                # 且 step0 时 g'=g_std != g_gt -> 立刻有梯度, 不像其它风格模块那样学不动。
                loss_deform = torch.tensor(0.0, device=device)
                _DEFORM_EMA = globals().setdefault('_DEFORM_EMA', {'loss': None, 'off': None})
                if getattr(args, 'w_deform_skel', 0.0) > 0:
                    _dm = getattr(model, 'deform_skel', None)
                    if _dm is None:
                        if not globals().get('_WARNED_DEFORM', False):
                            globals()['_WARNED_DEFORM'] = True
                            logger.warning("[deform] w_deform_skel>0 但模型没有 deform_skel "
                                           "-> 本项静默失效(已踩过多次, 故只首次告警)")
                    elif getattr(_dm, 'last_out', None) is None:
                        if not globals().get('_WARNED_DEFORM2', False):
                            globals()['_WARNED_DEFORM2'] = True
                            logger.warning("[deform] deform_skel.last_out 为空 -> 形变没被调用"
                                           "(compile 下属性写入可能被吞) -> 本项静默失效")
                    else:
                        _gsk = batch.get('inst_skel', None)
                        if _gsk is None:
                            if not globals().get('_WARNED_DEFORM3', False):
                                globals()['_WARNED_DEFORM3'] = True
                                logger.warning("[deform] batch['inst_skel'] 为空 "
                                               "(需 --inst-skel-shards-dir 指向 GT 骨架) "
                                               "-> 中间监督静默失效")
                        else:
                            _g2 = _dm.last_out
                            _tgt = _gsk.to(device, non_blocking=True).float()
                            # ★ 只在"真的做了形变"的样本上算: 条件被 drop 的样本用的是
                            #   null 风格向量, 形变头产出的是垃圾, 不该进监督。
                            #   (第一版只改前向没改 loss, 实测 loss 一模一样 0.3687)
                            _m = getattr(_dm, 'last_mask', None)
                            if _m is not None and not bool(_m.all()):
                                if bool(_m.any()):
                                    _g2 = _g2[_m]
                                    _tgt = _tgt[_m]
                                else:
                                    _g2 = None
                            if _g2 is not None and _g2.shape == _tgt.shape:
                                loss_deform = (_g2 - _tgt).pow(2).mean()
                                loss = loss + getattr(args, 'w_deform_skel', 0.0) * loss_deform
                                _v = float(loss_deform.detach())
                                _DEFORM_EMA['loss'] = (_v if _DEFORM_EMA['loss'] is None else 0.9 * _DEFORM_EMA['loss'] + 0.1 * _v)
                                _os = _dm.offset_stats()
                                if _os:
                                    _DEFORM_EMA['off'] = (_os['mean_abs'] if _DEFORM_EMA['off'] is None else 0.9 * _DEFORM_EMA['off'] + 0.1 * _os['mean_abs'])
                            else:
                                if not globals().get('_WARNED_DEFORM4', False):
                                    globals()['_WARNED_DEFORM4'] = True
                                    logger.warning(f"[deform] 形状不匹配 g'={tuple(_g2.shape)} "
                                                   f"target={tuple(_tgt.shape)} -> 跳过")

                # Stage 3 / v15 锚定正则: 把书家风格参数拉向预训练目标, 防"冻主干
                # 只训风格"时塌缩/漂移 (styletok 实测无护栏则 strict 掉头)。
                # λ 由 --style-anchor-weight 控制, 口径由 --style-anchor-mode 决定。
                if _style_anchor_w > 0 and _style_table_param is not None \
                        and _style_anchor_target is not None:
                    _Ecur = _style_table_param[:_style_anchor_target.shape[0]]
                    if _style_anchor_mode == "mean":
                        _Ecur = _Ecur.view(_style_anchor_target.shape[0],
                                           _style_anchor_k, -1).mean(dim=1)
                    loss = loss + _style_anchor_w * ((_Ecur - _style_anchor_target) ** 2).mean()

                opt.zero_grad(set_to_none=True)  # INFRA: set_to_none 释放梯度tensor, 比 zero_() 快且省内存

                # ★ 2026-09-17: 这里**不再**逐项 `.item()`（原来 7 次全 GPU 同步 -> 现在 1 次）。
                # ★ 2026-09-18: 把 `loss.backward()` 提前到 finiteness 同步**之前**。
                #   `bool(isfinite(loss))` 是一次全设备同步；原先放在 backward 前，会让 CPU
                #   在 forward 尾端停住 —— GPU 跑完 forward 后只能干等 CPU 重新入队 backward,
                #   形成 forward→backward 之间的流水线气泡。而 backward 本身**不需要** skip 决策
                #   （NaN loss 反传只产生 NaN 梯度, 不碰参数），真正必须同步的只有 `opt.step()`。
                #   故先无脑 backward, 再读 `_finite`, 让 forward+backward 在 GPU 上连跑。
                #   代价: 极少数 NaN 步白跑一次 backward —— 可忽略（且 NaN 梯度会在下一轮顶部
                #   `opt.zero_grad(set_to_none=True)` 被清掉, 不会污染 clip/step）。
                loss.backward()
                _finite = bool(torch.isfinite(loss))

                # NaN guard: skip clip/step/ema/accumulate if loss is not finite (e.g. a bad sample).
                if _finite:
                    torch.nn.utils.clip_grad_norm_(trainable_params_list, max_norm=1.0)
                    opt.step()
                    if scheduler is not None:
                        scheduler.step()
                    if ema_model is not None and (train_steps % max(1, int(getattr(args, 'ema_interval', 1))) == 0):
                        # 间隔更新: 每 N 步用 decay**N 更新, 数学上严格等价每步 decay
                        # (β 连乘 N 次 = β^N), 省 46M 参数 x N-1 次的显存带宽往返
                        if args.ema_warmup:
                            current_ema_decay = min(
                                args.ema_decay, (1.0 + train_steps) / (10.0 + train_steps))
                        else:
                            current_ema_decay = args.ema_decay
                        update_ema(ema_model, model, current_ema_decay ** max(1, int(getattr(args, 'ema_interval', 1))))
                else:
                    # 已 backward -> .grad 里留下 NaN，但**不** clip/step/ema -> 参数不被污染；
                    # 这些 NaN 梯度由下一轮顶部的 zero_grad(set_to_none=True) 释放。
                    nan_steps += 1
                    if rank == 0:
                        # 这里必须取标量（要打出来），但只在**出问题时**才发生 -> 不在热路径
                        logger.warning(
                            f"Step {train_steps} skipped: non-finite loss "
                            f"(diff={float(loss_diff):.4f}). Accumulated skips: {nan_steps}"
                        )
                # === INFRA: release the autograd graph every step, finite or not.
                # The graph built by training_losses (DiT forward + pred_xstart) must be
                # freed BEFORE the next forward, otherwise peak = diff_graph + aux_graph.
                # ★ 累加器只留 `.detach()`（仍是 GPU 张量，不触发同步），
                #   标量化推迟到日志步。见上面 `_finite` 处的说明。
                if _finite:
                    running_loss = running_loss + loss.detach()
                    running_diff = running_diff + loss_diff.detach()
                    running_repa = running_repa + loss_repa.detach()
                    running_std_mid = running_std_mid + loss_std_mid.detach()
                    running_skel = running_skel + loss_skel_struct.detach()
                    # c12 逐通道拆解也改成 GPU 侧累加 (C,) 向量，日志步再分组
                    _mse_ch = loss_dict.get("mse_ch", None) if isinstance(loss_dict, dict) else None
                    if _mse_ch is not None and _mse_ch.shape[1] > 4:
                        _cm = _mse_ch.mean(dim=0).detach()
                        _acc_c12 = _cm if _acc_c12 is None else (_acc_c12 + _cm)
                    log_steps += 1

                del loss, loss_dict, loss_diff, pred_xstart_latent
                del loss_repa, loss_std_mid, loss_x0lat, loss_skel_struct
                _mse_ch = None

                train_steps += 1

                # ★ 回收 torch.compile 的显存空洞（见 cli.py --empty-cache-after-warmup 的说明）
                #
                # ⚠ 必须在 warmup **期间每一步**都回收，不能只在结束后调一次：
                #   空洞是在前几十步里逐步累积的，而 batch 撑爆时 OOM 就发生在
                #   **warmup 期间**（实测 batch360 挂在 empty_strided_cuda((360,256,1024))）。
                #   结束后再回收，峰值已经过去了，救不了 OOM。
                #   每步调一次 empty_cache() 把高水位钉在当前步的活跃需求上。
                # 成本: 前 N 步每步多一次 device sync(~1-5ms)，一次性 ~0.2s，可忽略。
                if _empty_cache_at > 0 and train_steps <= _empty_cache_at:
                    _r0 = torch.cuda.memory_reserved() / 2 ** 30
                    _a0 = torch.cuda.memory_allocated() / 2 ** 30
                    torch.cuda.empty_cache()
                    if rank == 0 and (train_steps == _empty_cache_at
                                      or train_steps in (1, 10, 25)):
                        logger.info(
                            f"[alloc] step {train_steps} warmup empty_cache: reserved "
                            f"{_r0:.2f}G -> {torch.cuda.memory_reserved() / 2 ** 30:.2f}G | "
                            f"活跃 {_a0:.2f}G -> "
                            f"{torch.cuda.memory_allocated() / 2 ** 30:.2f}G | "
                            f"高水位 {torch.cuda.max_memory_reserved() / 2 ** 30:.2f}G | "
                            f"空洞 {_r0 - _a0:.2f}G")
                        # ★ 空洞 = reserved - allocated（allocator 缓存住、但没有任何
                        #   活张量在用的块）。要弄清"是谁把它顶上去的"，光看总量不够 ——
                        #   把按**尺寸分桶**的分配统计打出来，就能看出是"很多个小块"还是
                        #   "几个巨块"：前者是碎片/内核工作区，后者是某个大张量。
                        if getattr(args, "alloc_debug", False):
                            _st = torch.cuda.memory_stats()
                            _big = {k: v for k, v in _st.items()
                                    if "allocated_bytes" in k or "segment" in k
                                    or "num_" in k}
                            logger.info("[alloc] 尺寸分桶: " + " ".join(
                                f"{k}={v}" for k, v in sorted(_big.items())))
                            logger.info("[alloc] memory_summary:\n"
                                        + torch.cuda.memory_summary())

                if train_steps % args.log_every == 0:
                    torch.cuda.synchronize()
                    end_time = time()
                    # Guard against a logging window in which every step was skipped due to non-finite loss.
                    divisor = max(log_steps, 1)
                    steps_per_sec = log_steps / max(end_time - start_time, 1e-9)
                    
                    # 累加器已是 GPU 张量 -> 除法仍在 GPU；**只在这一次**取标量
                    avg_l = running_loss / divisor
                    avg_d = running_diff / divisor
                    avg_r = running_repa / divisor
                    avg_std_mid = running_std_mid / divisor
                    world_size = dist.get_world_size()
                    if world_size > 1:
                        dist.all_reduce(avg_l, op=dist.ReduceOp.SUM)
                        dist.all_reduce(avg_d, op=dist.ReduceOp.SUM)
                        dist.all_reduce(avg_r, op=dist.ReduceOp.SUM)
                        dist.all_reduce(avg_std_mid, op=dist.ReduceOp.SUM)
                        avg_l, avg_d = avg_l.item()/world_size, avg_d.item()/world_size
                        avg_r = avg_r.item()/world_size
                        avg_std_mid = avg_std_mid.item()/world_size
                    else:
                        avg_l, avg_d = avg_l.item(), avg_d.item()
                        avg_r = avg_r.item()
                        avg_std_mid = avg_std_mid.item()

                    # ref 12ch 分组 loss: 从累加的 (C,) 向量里拆出 image/canny/skel 三组。
                    # 等权目标下 Diff 即三组均值, 分组打印便于直接看到结构通道在被优化。
                    avg_skel = (running_skel / divisor).item()
                    avg_c12i = avg_c12c = avg_c12s = 0.0
                    if _acc_c12 is not None:
                        _cm = (_acc_c12 / divisor)
                        avg_c12i = float(_cm[:4].mean())
                        for _gi, _nm in enumerate(_aux_dirs):
                            _gm = float(_cm[4 + 4 * _gi: 8 + 4 * _gi].mean())
                            if 'canny' in _nm:
                                avg_c12c = _gm
                            elif 'skel' in _nm:
                                avg_c12s = _gm

                    if rank == 0:
                        wr = args.w_repa
                        ema_log = (f"EMA: {current_ema_decay:.6f} | "
                                   if ema_model is not None else "")
                        logger.info(
                            f"(step={train_steps:07d}) Diff: {avg_d:.4f} | "
                            f"REPA(w={wr:.2f}): {avg_r:.4f} | "
                            + (f"StdMid(w={args.w_std_mid:.3f}): {avg_std_mid:.4f} | "
                               if getattr(args, 'w_std_mid', 0.0) > 0 else "")
                            + (f"Deform(w={args.w_deform_skel:.2f}): "
                               f"{(globals().get('_DEFORM_EMA') or {}).get('loss') or 0.0:.4f} "
                               f"off={((globals().get('_DEFORM_EMA') or {}).get('off') or 0.0):.4f} | "
                               if getattr(args, 'w_deform_skel', 0.0) > 0 else "")
                            + (f"SkelStruct(w={args.w_latent_skel:.3f}): {avg_skel:.4f} | "
                               if getattr(args, 'w_latent_skel', 0.0) > 0 else "") +
                            f"LR: {opt.param_groups[0]['lr']:.2e} | {ema_log}"
                            f"Steps/Sec: {steps_per_sec:.2f} | "
                            f"Mem: {torch.cuda.memory_reserved() / 1024 ** 3:.2f}G/"
                            f"{torch.cuda.max_memory_reserved() / 1024 ** 3:.2f}G"
                        )
                        # ── 调试诊断 (每 1000 步): 梯度分组 + 注入门控强度 ──
                        # 门控 = xattn 注入输出在残差流上的平均 L2 (零初始化起,
                        # 增长 = 注入正在学会写入); 梯度分组看各通路是否在学习。
                        if train_steps % 1000 == 0:
                            if not _inj_stats:
                                _mm = model.module if hasattr(model, 'module') else model
                                for _i, _inj in enumerate(
                                        getattr(_mm, 'glyph_injections', []) or []):
                                    if not hasattr(_inj, 'out_proj'):
                                        continue

                                    def _mk(_i):
                                        def _h(_m, _inp, _out):
                                            _inj_stats[_i] = _out.detach().float() \
                                                .norm(dim=-1).mean().item()
                                        return _h
                                    _inj.register_forward_hook(_mk(_i))
                            _gn = {}
                            for _n, _p in model.named_parameters():
                                if _p.grad is None:
                                    continue
                                _g = _p.grad.norm().item()
                                if "glyph_injections" in _n:
                                    _k = "inj_out_proj"
                                elif "glyph_embedder" in _n:
                                    _k = "glyph_embedder"
                                elif "callig_proj" in _n or "callig_scale" in _n:
                                    _k = "callig_chain"
                                elif "blocks." in _n:
                                    _k = "blocks"
                                else:
                                    _k = "other"
                                _gn[_k] = _gn.get(_k, 0.0) + _g * _g
                            _gn = {k: v ** 0.5 for k, v in _gn.items()}
                            _gate = ""
                            if _inj_stats:
                                _gate = " | inj|out|: " + " ".join(
                                    f"L{i}={v:.3f}" for i, v in sorted(_inj_stats.items()))
                            logger.info(
                                f"(step={train_steps:07d}) [diag] grad-norm: "
                                + " ".join(f"{k}={v:.3f}" for k, v in sorted(_gn.items()))
                                + _gate)
                    
                    # 重置累加器（GPU 张量用 zero_ 原地清，避免重新分配）
                    running_loss.zero_()
                    running_diff.zero_()
                    running_repa.zero_()
                    running_std_mid.zero_()
                    running_skel.zero_()
                    _acc_c12 = None
                    log_steps = 0
                    start_time = time()

                # ★ 2026-09-17 简化: 原来三段 if/elif 拼出"前 5000 步每 1000 存一次、
                #   之后按 ckpt_every 存"的混合节奏（ckpt_every=5000 时实际落在
                #   1000..5000 + 7500/12500/... —— 与 eval 的 5000 边界**错开**）。
                #   现在 epoch 长度 / 存盘点 / 评测点已合并为同一个刻度
                #   （见上面的 `_epoch_steps`），存盘条件就是一行。
                #   顺带省掉开头 1000/2000/3000/4000 四次存盘（每次 592MB）。
                _save_ckpt = _epoch_steps > 0 and train_steps % _epoch_steps == 0

                if _save_ckpt and train_steps > 0:
                    if rank == 0:
                        # 组装(同步搬 CPU) + 异步入队 + 轮转，实现见 src/train/ckpt.py
                        save_checkpoint(model, opt, ema_model, scheduler, args,
                                        train_steps, checkpoint_dir, logger)
                        prune_checkpoints(checkpoint_dir,
                                          int(getattr(args, 'ckpt_keep', 0)), logger)

                        # 在训评测：是否该跑 / 用哪份权重 / 计时 / 报错 全在 eval 模块里。
                        # 失败只 warning 不中断训练（评测不该把训练带崩）。
                        # 另有已停用的 in-process GPU eval 路径，见 src/eval/legacy/。
                        maybe_run_in_training(
                            args, ema_model, model, train_steps, device,
                            checkpoint_dir, logger, is_eval_step=_save_ckpt,
                            inline=_EVAL_INLINE)

                if args.max_steps > 0 and train_steps >= args.max_steps:
                    logger.info(f"Reached max_steps={args.max_steps}; stopping cleanly.")
                    break

                if (train_steps >= int(getattr(args, 'early_stop_min_steps', 0))
                        and train_steps % _es_check_every == 0
                        and rank == 0
                        and _stopper.check()):      # 内部已判 early_stop 开关
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
    # 等后台写盘收尾 —— 否则最后一个 ckpt 可能还没落盘进程就退了。
    if dist.get_rank() == 0:
        drain_ckpt(logger, timeout=600)
    logger.info("Done!")
    cleanup()

def main_from_cli(argv=None):
    """CLI entry: build the argparse parser (config-file defaults + CLI overrides),
    parse, and run main(args). Used by `python train.py` (root launcher) and
    `python -m src.train.train`.
    """
def main_from_cli(argv=None):
    """CLI 入口：解析参数后跑 main(args)。

    argparse 定义（162 个参数）与 config 合并逻辑都在 `src/train/cli.py`。
    被 `python train.py`（根启动器）与 `python -m src.train.train` 使用。
    """
    args = parse_args(argv)
    main(args)
    return args


if __name__ == "__main__":
    main_from_cli()
