from src.model.build import build_dit_from_args   # noqa: E402
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


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# [OPTIM-FUSED 2026-09-16] 让 cuDNN 为 (固定的) glyph_embedder conv
# 选最快算法; 形状固定, 一次性 autotune 成本可忽略。
torch.backends.cudnn.benchmark = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

from time import time

import json
import copy
import math

# ═══════════════════════════════════════════════════════════════════════════
# ★★ 铁律 (2026-10-03, 用户裁定): **绝不启用 expandable_segments** ★★
#   PyTorch 的 OOM 报错会主动建议 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,
#   **看到就忽略** —— 那是报错文案, 不是解决方案; 本项目已判定不允许。
#   这道闸门在**任何 CUDA 分配之前**强制剔除该环境变量: torch 的 caching
#   allocator 是在**首次 CUDA 分配时**才读它, 所以在此删除有效 (import torch
#   之后再删就晚了)。
#   显存不足时的正确应对: 调小 batch / 换省显存实现, **不许动分配器**。
# ═══════════════════════════════════════════════════════════════════════════
import os as _os_alloc_gate

for _k in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
    _v = _os_alloc_gate.environ.get(_k)
    if _v and "expandable" in _v.lower():
        _os_alloc_gate.environ.pop(_k, None)
        print(f"[alloc-gate] ✗ 检测到 {_k}={_v!r} -> **已强制剔除** "
              f"(铁律: 不允许 expandable_segments)", flush=True)
del _k, _v
from src.model import DiT_2Cond_models
from src.loss import create_diffusion_or_flow, flow_kwargs_from
from diffusers.models import AutoencoderKL
from src.utils import find_model


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
from src.train import backward as _bw          # ★ PR4: backward 策略模块 ✓
from src.train import ema as _ema              # ★ PR4: EMA 组件 ✓
from src.train.checkpoint import CheckpointManager as _CkptManager   # ★ PR4: 持久化 ✓
from src.train import optimizer as _opt        # ★ PR4: 优化器装配（纯决策 + 薄工厂）✓
from src.train.trainer import TrainState as _TrainState, training_loop as _training_loop
from src.train.step import (NonFiniteGuard as _NonFiniteGuard,   # ★ PR4: 单步组件 ✓
                            )
from src.train.ckpt import (restore_rng)
from src.train.data_setup import build_data   # ★ PR4: 数据装配 ✓
from src.train.setup import setup_runtime   # ★ PR4: 运行时装配 ✓
from src.train.writer_table_tuning import configure_writer_table, split_writer_optimizer_group

from src.eval.in_mem_eval import run_in_mem_eval

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


# ★ 2026-10-14 [重构 PR4]: `update_ema` 已**逐字搬移**到 src/train/ema.py ✓
#   这里保留同名**再导出** ✓ ⇒ 既有 `from src.train.train import update_ema` 的
#   调用点（含测试 ✓）无需改动 ✓。新代码请 `from src.train import ema` ✓。
from src.train.ema import update_ema            # noqa: E402,F401

# ckpt 相关实现已移到 src/train/ckpt.py（state_to_cpu / AsyncCkptWriter /
# save_checkpoint / prune_checkpoints / drain_ckpt）

def cleanup():
    dist.destroy_process_group()

# ★ 2026-10-13 [审查 PR4]: create_logger 已迁到 src/train/setup.py ✓（下方 import ✓）

def _resolve_and_validate_args(args):
    """args 校验与词表解析（原 main 内联 89 行 ✓, 2026-10-13 抽出）。

    内容与原文逐字相同, 只整体缩进不变 ⇒ 行为不变 ✓（由 main() 的 **setup 行为指纹**
    裁决: setup 打印的解析后配置值必须一字不差 ✓）。
    args 为**原地修改**（写回 content_cond / _callig_map / num_calligraphers 等）
    ⇒ 只返回后续 main 需要用的 _std_soft_cond ✓。
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # std-soft is a third, explicit g representation.  Fail before allocating a
    # process group/model if a config accidentally mixes it with either legacy
    # VAE glyph source; those tensors have different channel/spatial contracts.
    _std_soft_cond = bool(getattr(args, "std_soft_cond", False))
    if _std_soft_cond:
        if bool(getattr(args, "w_glyph_cond", False)) or bool(
                getattr(args, "skel_as_glyph_cond", False)):
            raise ValueError("std_soft_cond is mutually exclusive with w_glyph_cond and "
                             "skel_as_glyph_cond")
        if not str(getattr(args, "std_soft_root", "") or "").strip():
            raise ValueError("std_soft_cond requires --std-soft-root")
        if int(getattr(args, "std_soft_size", 64)) != 64:
            raise ValueError("current std-soft encoder requires --std-soft-size 64")
        if bool(getattr(args, "glyph_concat_input", False)):
            raise ValueError("std_soft_cond is incompatible with glyph_concat_input")
        if int(getattr(args, "deform_skel", 0) or 0) > 0:
            raise ValueError("std_soft_cond is incompatible with latent-space DeformSkel")

    # ── ★ 2026-10-13 [审查 PR2]: 内容条件校验**收敛到唯一真源** ✓ ──────────────────
    #   此处原先**复刻**了一份校验, 且只允许 table/std ✗ —— 于是 CLI / ContentConfig /
    #   模型三处都合法的 content_cond="hybrid" 会被**训练入口拒掉** ✗✗(v83A/B/C 因此
    #   起不来, 已实测确认)。现在只调 src/model/conditioning.py, 三处定义合一 ✓。
    #   （历史 ckpt 的旧格式推导放在同一转换器里, 不再混在新训练路径 ✓）
    from src.model.conditioning import content_config_from_args as _cc_from_args
    _cc = _cc_from_args(args, log=print)
    args.content_cond = _cc.mode
    args.no_char_cond = not _cc.use_char_cond
    args.std_soft_cond = _cc.std_soft_cond
    _std_soft_cond = _cc.std_soft_cond
    print(f"[content-cond] 模式 = {_cc.mode} ✓ "
          f"(table=仅字表 / std=仅标准字形 / hybrid=两者并存, 归因不可分离 ✗；"
          f"书家/书体表三种模式都保留 ✓)")
    # 只保留**与真实数据资源有关**的额外校验（模式合法性已由 ContentConfig 负责 ✓）
    if _cc.mode == "std" and not str(getattr(args, "std_soft_root", "") or "").strip():
        raise ValueError("content_cond='std' 需要 --std-soft-root（标准字形库路径）")

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
    return _std_soft_cond



def _build_aux_losses(args, device, logger, rank):
    """辅助 loss 的构建（原 main 内联 87 行 ✓, 2026-10-13 抽出）。

    三块: style-rank 风格秩 loss · 实例骨架结构 loss(冻结 probe, 新增可训练参数 0)
    · 中程结构 aux loss。**返回 4 个供训练循环使用的对象** ✓
    （耦合分析在这里给出"输出为空"是**假阴性** ✗ —— 它只扫了 setup 区,
      而这 4 个名字在 [Begin epoch] 之后的训练循环里被使用 ✓ 已人工核对 L2287-2353 ✓）。
    """
    _style_rank_loss_fn = None
    # ---- v85B/C: 冻结双头 critic（Writer+Brush ✓ 方案 §六 ✓）----
    _critic_loss_fn = None
    if getattr(args, 'w_critic_writer', 0.0) > 0 or getattr(args, 'w_critic_away', 0.0) > 0:
        import importlib
        _clm = importlib.import_module('src.loss.critic_loss')
        _critic_loss_fn = _clm.CriticLoss(
            ckpt=getattr(args, 'critic_ckpt', 'assets/writer_brush_critic.pt'),
            device=device)
        _critic_loss_fn.attach_std_bank(
            getattr(args, 'critic_std_bank', 'scratch/disc_lat_full.npz'))
        _lut = _critic_loss_fn.build_id2slot(
            args.data_csv, getattr(args, 'callig_remap_json', ''),
            getattr(args, 'font_remap_json', ''))
        logger.info('[critic] id2slot lut: %d 对（callig_raw×script ✓）' % len(_lut))
        logger.info("[critic] frozen writer+brush critic loaded: %s (%d slots) | "
                    "w_writer=%.4g w_away=%.4g t=[%.2f,%.2f]"
                    % (getattr(args, 'critic_ckpt', ''), len(_critic_loss_fn.slots),
                       args.w_critic_writer, args.w_critic_away,
                       getattr(args, 'critic_t_min', 0.05),
                       getattr(args, 'critic_t_max', 0.30)))
    if getattr(args, 'w_style_rank', 0.0) > 0:
        import importlib
        _srm = importlib.import_module('src.loss.style_rank_module')
        _style_rank_loss_fn = _srm.StyleRankLoss(
            ckpt=getattr(args, 'style_rank_ckpt', 'assets/style_enc_latent.pt'),
            cent_npy=getattr(args, 'style_rank_cent', 'assets/rank_cent87.npy'),
            t_min=float(getattr(args, 'style_rank_t_min', 0.05)),
            t_max=float(getattr(args, 'style_rank_t_max', 0.25)),
            # ★ v85B: 默认 mode="cos" ⇒ **与历史逐位一致** ✓（不静默改变旧实验 ✓）
            mode=str(getattr(args, 'style_rank_mode', 'cos') or 'cos'),
            protos=str(getattr(args, 'style_rank_protos', '') or ''),
            margin=float(getattr(args, 'style_rank_margin', 0.0) or 0.0),
            tau=float(getattr(args, 'style_rank_tau', 0.07) or 0.07),
            verbose=True)
        _style_rank_loss_fn.to(device)
        logger.info("[style-rank] enabled: w=%.4f mode=%s t=[%.2f,%.2f] cent=%s "
                    "rival=有 ✓ added=0"
                    % (args.w_style_rank, _style_rank_loss_fn.mode,
                       _style_rank_loss_fn.t_min, _style_rank_loss_fn.t_max,
                       tuple(_style_rank_loss_fn.cent.shape)))

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
    return _style_rank_loss_fn, _skel_struct_loss_fn, _SKEL_STRUCT_WARNED, _mid_struct_loss, _critic_loss_fn



def main(args):
    # ★ 2026-10-13: 原 89 行内联的 args 校验+词表解析已抽成函数 ✓
    _std_soft_cond = _resolve_and_validate_args(args)

    # gloo is the only backend available on Windows; prefer nccl on Linux for speed.
    # ★ 2026-10-13 [审查 PR4]: 运行时装配已抽成 src/train/setup.setup_runtime ✓（逐行等价 ✓）
    _rt = setup_runtime(args)
    rank, device, seed, logger = _rt.rank, _rt.device, _rt.seed, _rt.logger
    experiment_dir, checkpoint_dir = _rt.experiment_dir, _rt.checkpoint_dir
    vae_downscale, latent_size = _rt.vae_downscale, _rt.latent_size
    # ★ 2026-10-14 [重构 PR4] **持久化的状态所有者** ✓（审查 §4.2: CheckpointManager ✓）
    #   把"目录 / 轮转 / 严格模式 / 上次存到第几步"收成对象 ✓
    #   实现仍在 src/train/ckpt.py ✓（异步写 + fsync + os.replace + SHA256 manifest 全保留 ✓）
    _ckpt = _CkptManager(checkpoint_dir,
                         keep=int(getattr(args, "ckpt_keep", 0) or 0),
                         logger=logger)
    logger.info(f"[ckpt] {_ckpt.describe()}")     # 让存盘策略成为**显式身份** ✓
    
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

        # ── ★ 2026-10-13 [审查 P0-A]: **有效配置落盘** ✓ ─────────────────────────────
        #   此前只有"输入的 config JSON", 没有"真正生效的配置" ✗ —— 一旦未注册键被静默
        #   忽略 / CLI 覆盖 / 代码默认参与, 事后无法知道实际跑了什么。现在把解析后的
        #   完整 vars(args) + sha256 落到实验目录, "记录配置 ≠ 实际配置"可直接对账 ✓。
        try:
            import hashlib as _hl
            import json as _json
            _eff = {k: (v if isinstance(v, (int, float, bool, str, type(None), list, dict)) else str(v))
                    for k, v in vars(args).items() if not k.startswith("__")}
            _eff_blob = _json.dumps(_eff, ensure_ascii=False, sort_keys=True, default=str)
            _eff_hash = _hl.sha256(_eff_blob.encode("utf-8")).hexdigest()
            _eff_dir = getattr(args, "results_dir", ".") or "."
            os.makedirs(_eff_dir, exist_ok=True)
            _eff_path = os.path.join(_eff_dir, "effective_config.json")
            with open(_eff_path, "w", encoding="utf-8") as _f:
                _json.dump({"_sha256": _eff_hash, "_config_path": getattr(args, "config", None),
                            "args": _eff}, _f, ensure_ascii=False, indent=2, sort_keys=True)
            print(f"[effective-config] 已落盘 {_eff_path} (sha256={_eff_hash[:16]}) ✓", flush=True)
        except Exception as _e:                                     # noqa: BLE001
            print(f"[effective-config] ⚠ 落盘失败(不致命): {type(_e).__name__}: {_e}", flush=True)

        model = build_dit_from_args(args)   # ★ 唯一构造器（src/model/build.py）✓ 不再手写 kwargs
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
                    f"glyph_cond={(getattr(args, 'w_glyph_cond', 0) > 0) or getattr(args, 'skel_as_glyph_cond', False) or _std_soft_cond}, "
                    f"std_soft={_std_soft_cond}, glyph_scale_init={getattr(args, 'glyph_scale_init', 0.4)}, "
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

    # ── v53: 三表重映射 json + SupCon 表初始化 ──────────────────────────────
    _char_remaps = None
    if getattr(args, 'char_remap_json', ''):
        import json as _json
        _char_remaps = {}
        for _nm, _p in (('char', args.char_remap_json),
                        ('callig', getattr(args, 'callig_remap_json', '')),
                        ('font', getattr(args, 'font_remap_json', ''))):
            if _p:
                _char_remaps[_nm] = {int(k): int(v) for k, v in
                                     _json.load(open(_p, encoding="utf-8")).items()}
        logger.info(f"[triple] remap json 已加载: {list(_char_remaps.keys())}")
    _ttp = getattr(args, 'triple_table_prefix', '') or ''
    if _ttp:
        import json as _json
        import numpy as _np
        for _nm, _emb in (('callig', model.y_callig_embedder),
                          ('font', model.y_script_embedder),
                          ('char', model.y_char_embedder)):
            if _emb is None:
                continue
            _w = _np.load(f"{_ttp}{_nm}_table.npy")
            _cls = _json.load(open(f"{_ttp}{_nm}_index.json",
                                   encoding="utf-8"))["classes"]
            _tab = _emb.embedding_table.weight
            assert _w.shape[0] == len(_cls), \
                f"{_nm}: 表行 {_w.shape[0]} != 类数 {len(_cls)}"
            assert _w.shape[1] == _tab.shape[1], \
                f"{_nm}: 维度 {_w.shape[1]} != 表 {_tab.shape[1]}"
            with torch.no_grad():
                for _i in range(len(_cls)):
                    # 表行号 == builder 索引 (dataset 侧 remap 负责把原始 id 映到行号)
                    _tab[_i].copy_(torch.from_numpy(_w[_i]).float())
        logger.info(f"[triple-init] 三表 SupCon 初始化完成 (prefix={_ttp})")

    # ── v54: 三表完全冻结 ────────────────────────────────────────────────────
    # 用 requires_grad_(False) 直接冻结, **不走** freeze_table()/freeze_char_table:
    #   那两条路会把 CFG null token 拆成独立 Parameter → 参数结构变化 → 评测侧
    #   (model_io) 必须复刻同样的拆解, 且 freeze_callig_table 还带
    #   "必须先 callig_emb_pretrained" 的 assert (与 triple_table_prefix 初始化冲突)。
    # 代价: 冻结后 CFG null 行也停在 init 值 (cond_drop_all 5% 时用到), 换取零结构差异。
    if getattr(args, "freeze_triple_tables", False):
        _fz_n = 0
        _mm = model.module if hasattr(model, "module") else model
        for _nm_f in ("y_callig_embedder", "y_script_embedder", "y_char_embedder"):
            _m = getattr(_mm, _nm_f, None)
            if _m is None:
                continue
            for _p in _m.parameters():
                _p.requires_grad_(False)
                _fz_n += _p.numel()
        logger.info(f"[triple-freeze] 三表已完全冻结: {_fz_n:,} 参数 "
                    f"(结构不变, 评测侧无需改动; CFG null 行同样冻结)")

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
                # [v52 2026-10-04] glyphs 兼容两种口径:
                #   历史: [script, char] 二元组 / "s_c" 字符串 -> gid = s*7026 + c
                #   SupCon v2: 一维 int = 全局 character_id (与 y_char label 同空间,
                #              latent_dataset.py:615 实证 csv glyph_id == character_id)
                for _gi, _g in enumerate(_glyphs):
                    if isinstance(_g, (list, tuple)) and len(_g) >= 2:
                        _gid = int(_g[0]) * _NUM_CH + int(_g[1])
                    elif isinstance(_g, str) and "_" in _g:
                        _a, _b = _g.split("_")[:2]
                        _gid = int(_a) * _NUM_CH + int(_b)
                    else:
                        _gid = int(_g)
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

    # 3) ★ v85 **受控 warm-start**（方案 §二.5）—— 放在重置**之后** ⇒ 我们的权重胜出 ✓
    #    与 `--resume-full` 的区别（方案明确要求 ✓）:
    #      · 只灌**模型权重**, **不动 optimizer**（新模型参数集不同 ⇒ 必须重建 optimizer ✓）
    #      · 允许缺失的键**仅限**新增的 style-ada 分支 ✓；原模块 missing/unexpected 一律报错 ✓
    #      · 不恢复 step 计数（这是一次**新训练** ✓）
    _warm = str(getattr(args, "warm_start_from", "") or "")
    if _warm:
        from src.eval.model_io import check_state_dict as _chk
        _ck = torch.load(_warm, map_location="cpu", weights_only=False)
        if isinstance(_ck, dict):
            _src = None
            for _cand in ("ema", "model", "delta"):
                if _ck.get(_cand) is not None:
                    _sd, _src = _ck[_cand], _cand
                    break
            if _sd is None:
                _sd, _src = _ck, "raw"
        else:
            _sd, _src = _ck, "raw"
        if any(k.startswith("_orig_mod.") for k in _sd):
            _sd = {k.replace("_orig_mod.", "", 1): v for k, v in _sd.items()}
        _miss, _unexp = model.load_state_dict(_sd, strict=False)
        # ★ 只允许**新增的**风格支路缺失 ✓（这是"受控"二字的全部含义 ✓）
        _m2, _u2 = _chk(model, _sd, tag="warm-start",
                        allow_missing=("style_ada", "style_gain"), raise_on_missing=True)
        if _u2:
            raise RuntimeError(
                "[warm-start] ckpt 里有 %d 个**本模型不认识**的键 ✗（原模块不该有 unexpected ✓）: %s"
                % (len(_u2), sorted(_u2)[:6]))
        _new = [k for k in model.state_dict()
                if ("style_ada" in k or "style_gain" in k)]
        logger.info("[warm-start] %s (%s): 新支路 %d 个键按**本模型初始化** ✓; "
                    "原有模块 0 缺失/0 意外 ✓", _warm, _src, len(_new))
        del _ck, _sd


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
    if (bool(getattr(args, 'resume_full', None))
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

    # ── [2026-09-29] --train-only-deform: 冻结**整个主干**, 只训形变头 SkelNet ─────
    # 两阶段方案的第 ② 步:
    #   ① 先用 GT 骨架当 g 训好一个模型 (v26) 并冻结;
    #   ② 把 SkelNet 接到前面 (g = SkelNet(std, style)), **只训 SkelNet**,
    #      让它学会输出"该冻结模型吃得下"的骨架 —— 即被拉向 GT 骨架空间。
    # 与 v24 联合训练 (deform_trainable=1 且主干也在训) 的区别: 主干完全不动,
    #   梯度只用于塑造 SkelNet, 不会把已训好的主干带偏 (v24 实测 off 1.40→2.37 膨胀即反例)。
    # ⚠ 必须放在优化器构建 (trainable_params_list = ... requires_grad) **之前** —— 本段
    #   位于 freeze policy 区 (~L940-1170), 早于 L1401 的过滤, 所以顺序是对的。
    _train_only_deform = bool(getattr(args, 'train_only_deform', False))
    if _train_only_deform:
        if getattr(model, 'deform_skel', None) is None:
            raise SystemExit(
                "[train-only-deform] 模型里没有 deform_skel —— 需要 --deform-skel 1, "
                "并建议 --deform-ckpt 指向已训的 SkelNet 做热启动")
        requires_grad(model, False)
        _n_tr, _tot = 0, 0
        for _p in model.deform_skel.parameters():
            _p.requires_grad = True
            _n_tr += _p.numel()
        for _p in model.parameters():
            _tot += _p.numel()
        if _n_tr == 0:
            raise SystemExit("[train-only-deform] deform_skel 没有参数可训")
        logger.info(f"[train-only-deform] 冻结主干, 只训形变头 SkelNet: "
                    f"{_n_tr:,} / {_tot:,} 参数可训 ({_n_tr / _tot * 100:.2f}%)")

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
    # ★ 2026-10-03: K 个风格 token 的正交正则 (v15b/c 的"伪多模态"对策)。
    #   实测 K-Means 初始化的 4 个 token 两两余弦 0.900 = 已塔缩;
    #   用 PCA 正交基初始化(cos≈0)后, 本 loss 防止训练中重新靠拢。
    _style_ortho_w = float(getattr(args, 'w_style_ortho', 0.0) or 0.0)
    _style_table_param = None
    _style_anchor_target = None
    _style_anchor_k = 1
    if _style_anchor_w > 0 or _style_ortho_w > 0:
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
    # ★ 2026-10-03 启动自检: K 个风格 token 的初始互余弦平方。
    #   期望 ≈0 (PCA 正交基); 若 ≈0.8 (=cos 0.900) 说明拿的还是共线的 K-Means 资产。
    if _style_ortho_w > 0 and _style_table_param is not None and _style_anchor_k > 1 \
            and rank == 0:
        with torch.no_grad():
            _t0 = _style_table_param.detach().view(
                _style_table_param.shape[0], _style_anchor_k, -1)
            _n0 = torch.nn.functional.normalize(_t0, dim=-1)
            _s0 = _n0 @ _n0.transpose(1, 2)
            _of0 = ~torch.eye(_style_anchor_k, dtype=torch.bool, device=_s0.device)
            _c0 = float((_s0[:, _of0] ** 2).mean())
        logger.info(f"[style-ortho] λ={_style_ortho_w} K={_style_anchor_k} "
                    f"表={tuple(_style_table_param.shape)} 初始 mean(cos²)={_c0:.6f} → mean cos={_c0 ** 0.5:.4f} "
                    + ("✓ 正交基 (已打破对称)" if _c0 < 0.05
                       else "✗✗ 仍共线! 检查 callig_emb_pretrained"))
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

    # This runs after all initialization/warm-start and freezing policies, and
    # before DDP, compile and optimizer setup; the checkpoint structure stays intact.
    _writer_table_tuning = configure_writer_table(model, args)
    if _writer_table_tuning is not None:
        logger.info('[writer-table] real rows=%d dim=%d; null fixed; lr=%.3g wd=0 anchor=%.3g',
                    _writer_table_tuning.n_writers, _writer_table_tuning.weight.shape[1],
                    args.callig_table_lr, args.callig_table_anchor_weight)

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
            # [infra 2026-10-04] dynamo 缓存上限防御: 默认 256, 打穿后 torch.compile
            # 静默回退 eager (batch192 下 eager ~22G, compiled 14.4G) -> 回退即 OOM,
            # 症状伪装成 batch 太大。提高上限让偶发重编译继续走编译路径。
            from torch import _dynamo as _td   # 勿用 import torch._dynamo: 会把 torch
            _td.config.cache_size_limit = max(    # 变成本函数局部名 -> UnboundLocalError
                64, _td.config.cache_size_limit)
            _td.config.accumulated_cache_size_limit = 4096
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
        # ★ 2026-10-14 [重构 PR4] 把**反向传播策略**写进启动日志 ✓
        #   审查 §4.2 要求"分别测量速度与显存" ✓ ⇒ 先要能按策略分组 ✓
        #   （joint = 图同时存活 ✓ 峰值显存高; sequential = 先释放 FM 图 ✓ 峰值低 ✓）
        logger.info(f"[backward] {_bw.describe(args)}")
        logger.info(f"[ema] {_ema.describe(args)}")     # ★ PR4: interval 属实验身份 ✓
        logger.info(f"[optim-identity] {_opt.describe(args)}")   # ★ PR4: 优化器身份 ✓
        logger.info(f"[EMA] enabled with decay={args.ema_decay}, interval={getattr(args, 'ema_interval', 1)} "
                    f"(effective per-step decay={args.ema_decay ** max(1, int(getattr(args, 'ema_interval', 1)))})")
    if dist.get_world_size() > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    # flow 的 t 分布 / 求解器 / shift 由 config 指定，训练与 eval 共用同一份
    # （通过 flow_kwargs_from 抽取），避免两侧静默不一致。
    _flow_kw = flow_kwargs_from(args)

    # ★ [2026-10-06] 层级敏感度探针（默认关：--probe-inject-every<=0 时只是空对象）。
    #   用途：跑 500~1000 步测出"主干到底想在哪些层看骨架"，再据此确定正式注入层。
    from src.utils.inject_probe import InjectLayerProbe
    _inj_probe = InjectLayerProbe(args, model, logger)
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
    # ★ 2026-10-10: 在线编码必须与 shards 的存盘约定一致（合同 I2）:
    #   shards = (z_raw - shift) * sf; 解码 = decode(z/sf + shift)。
    #   原来这两条在线路径只 mul_(sf) 漏了 shift -> 与预编码路径约定不一致(静默)。
    _vae_shift = float(getattr(args, 'vae_shift_factor', 0.0) or 0.0)
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
    # 2026-10-08 REPA-E: 解冻 VAE Encoder 联合微调时必须加载 VAE
    _train_vae_enc = bool(getattr(args, "train_vae_encoder", False))
    _need_vae = _train_vae_enc or (not bool(getattr(args, "latent_shards_dir", None)))
    if _need_vae:
        try:
            vae_target_path = getattr(args, 'calli_vae_path', None) or getattr(args, 'vae_path', None)
            if vae_target_path is not None and os.path.exists(vae_target_path):
                logger.info(f"Loading VAE from local path: {vae_target_path}")
                vae = AutoencoderKL.from_pretrained(vae_target_path).to(device)
            else:
                vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
            
            if _train_vae_enc:
                logger.info("[REPA-E] 激活端到端联合微调: 解冻 VAE Encoder, 彻底冻结 Decoder")
                # 1. 🔒 VAE Decoder 完全冻结 (不训练, 杜绝高频解码退化并省下显存)
                vae.decoder.eval()
                requires_grad(vae.decoder, False)
                if hasattr(vae, "post_quant_conv"):
                    requires_grad(vae.post_quant_conv, False)
                
                # 2. 🟢 VAE Encoder 开启梯度 (端到端自适应流形雕刻)
                vae.encoder.train()
                requires_grad(vae.encoder, True)
                if hasattr(vae, "quant_conv"):
                    vae.quant_conv.train()
                    requires_grad(vae.quant_conv, True)
                
                # 激活 Encoder 梯度检查点以压缩激活显存
                if hasattr(vae.encoder, "gradient_checkpointing"):
                    vae.encoder.gradient_checkpointing = True
                    logger.info("[REPA-E] 成功开启 VAE Encoder 梯度检查点 (Gradient Checkpointing)")
            else:
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
    # ★ 2026-10-13: 原 87 行内联的 aux-loss 构建已抽成函数 ✓（4 个输出供训练循环使用 ✓）
    _style_rank_loss_fn, _skel_struct_loss_fn, _SKEL_STRUCT_WARNED, _mid_struct_loss, _critic_loss_fn = _build_aux_losses(args, device, logger, rank)
    from src.train.strong_style import build_strong_style
    _strong_style = build_strong_style(args, device, logger)

    _train_vae_enc = bool(getattr(args, "train_vae_encoder", False))
    if _train_vae_enc:
        # ⭐️ REPA-E 双轨优化器: 把 VAE Encoder 与 DiT/REPA 拆进不同 Parameter Groups
        dit_trainable = [p for p in model.parameters() if p.requires_grad]
        repa_trainable = repa_loss_fn.trainable_params() if repa_loss_fn is not None else []
        vae_enc_trainable = [p for p in vae.encoder.parameters() if p.requires_grad]
        if hasattr(vae, "quant_conv"):
            vae_enc_trainable.extend([p for p in vae.quant_conv.parameters() if p.requires_grad])

        vae_lr = float(getattr(args, 'vae_lr', 2e-6))
        optim_groups = [
            # Group 1: DiT 主干与条件嵌入 (高强度学习)
            {"params": dit_trainable, "lr": args.lr, "weight_decay": args.weight_decay},
            # Group 2: REPA 投影层
            {"params": repa_trainable, "lr": args.lr, "weight_decay": 0.0},
            # Group 3: VAE Encoder (极微小学习率，做流形软着陆)
            {"params": vae_enc_trainable, "lr": vae_lr, "weight_decay": 0.0},
        ]
        # ★ 2026-10-14 [重构 PR4] fused 判定与 AdamW 构造统一走 src/train/optimizer.py ✓
        #   （旧代码把 fused 判定**写了两遍** ✗ ⇒ 现在只有一处 ✓ 不会漂移 ✓）
        #   ⚠ 组内**已自带** lr/wd ✓ ⇒ 工厂绝不得再传 lr/wd ✗
        #     （传了会把三组的 lr 统一覆盖 ✗✗ —— VAE 双轨就靠各组不同 lr 区分 ✓）
        opt, _fused = _opt.build_adamw_groups(optim_groups)
        logger.info(f"[optim] REPA-E Two-Speed AdamW: DiT lr={args.lr} wd={args.weight_decay} | "
                    f"REPA lr={args.lr} | VAE Encoder lr={vae_lr} (trainable={len(vae_enc_trainable)} tensors)")
    else:
        trainable_params_list = [p for p in model.parameters() if p.requires_grad]
        if repa_loss_fn is not None:
            trainable_params_list.extend(repa_loss_fn.trainable_params())

        _opt_name = getattr(args, "optimizer", "adamw")
        if _opt_name == "muon":
            # Muon: 矩阵权重 NS-正交化 (独立 muon_lr, 典型 0.02-0.05), 向量/embedding 走 AdamW (adamw_lr)
            # REPA proj 是矩阵 dim=2 -> 自动进 muon 组, 会用它; 向量(embedding标量)进 adamw
            # ★ 2026-10-14 [重构 PR4] 构造走 src/train/optimizer.py ✓（含导入失败报错文本 ✓）
            opt = _opt.build_muon(trainable_params_list, args, logger)
        else:
            # [OPTIM-FUSED 2026-09-16] AdamW fused (单内核多张量更新, 省 elementwise 带宽)。
            # 守卫: 仅 CUDA 且 torch 支持 fused 时启用, 否则退回默认实现。
            # ★ 2026-10-14 [重构 PR4] 同上（fused 判定与回退都在工厂里 ✓ 只有一处实现 ✓）
            opt, _fused = _opt.build_adamw(trainable_params_list, lr=args.lr,
                                           weight_decay=args.weight_decay,
                                           logger=logger, tag="AdamW")

    split_writer_optimizer_group(opt, _writer_table_tuning,
                                 float(getattr(args, 'callig_table_lr', 2e-6)))

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

    # ★ v85A: 新增的**风格支路**单独一组 lr（方案 §二.5: 主干 1e-5 / 新支路 5e-5 ✓）
    #   为什么单独一组: 主干是从 v84A@200k **warm-start** 来的成熟状态, 大 lr 会把它推坏 ✗;
    #   而风格支路是**全新初始化**的, 需要正常 lr 才能开始学 ✓。两个诉求相反 ⇒ 必须分组 ✓。
    #   实现完全照 `deform_skel` 的先例 ✓（LambdaLR 按各组**初始 lr** 等比缩放 ⇒
    #   设好初值即自动跟随同一条 cosine ✓）。**没有改优化器实现** ✓。
    _sal = float(getattr(args, "style_ada_lr", 0.0) or 0.0)
    if _sal > 0 and int(getattr(model, "style_ada_rank", 0) or 0) > 0:
        _sid = {id(p) for n, p in model.named_parameters()
                if ("style_ada" in n or "style_gain" in n)}
        _pg0 = opt.param_groups[0]
        _rest = [p for p in _pg0["params"] if id(p) not in _sid]
        _sm = [p for p in _pg0["params"] if id(p) in _sid]
        if _sm:
            _pg0["params"] = _rest
            _new = {k: v for k, v in _pg0.items() if k != "params"}
            _new["params"] = _sm
            _new["lr"] = _sal
            opt.add_param_group(_new)
            logger.info("[optim] style-ada 单独一组: lr=%.2e（主 lr %.2e）, %s 参数 / %d 个张量"
                        % (_sal, args.lr, format(sum(p.numel() for p in _sm), ","), len(_sm)))
    elif int(getattr(model, "style_ada_rank", 0) or 0) > 0:
        logger.info("[optim] style-ada 用**主 lr**（--style-ada-lr 未设 ✓）")

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
                logger.info("[resume-full] Restored optimizer state.")
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
    # ★ 2026-10-13 [审查 PR4]: 数据装配已抽成 src/train/data_setup.build_data ✓（逐行等价 ✓）
    #   ⚠ 边界复核（第一版漏了两个 **块内定义、块外使用** 的变量 ✗，已由 pyflakes 抓出 ✓）:
    #     _epoch_steps 由块内的"单一刻度"校验产生、块外仍用于 sampler 告警 ✓
    #     _EVAL_INLINE 在块内定义、块外用于 deferred eval 判定 ✓ ⇒ 二者一并返回 ✓
    dataset, sampler, loader, _epoch_steps, _EVAL_INLINE = build_data(
        args, rank=rank, world_size=dist.get_world_size(),
        resume_start_step=resume_start_step, std_soft_cond=_std_soft_cond,
        char_remaps=_char_remaps, aux_dirs=aux_dirs_of(args), logger=logger)
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

    # ── v53: 三张条件表用**独立小学习率** (lr = 主 lr × triple_table_lr_scale) ──
    # 位置很关键, 三条顺序约束:
    #   1) 必须在 `opt.load_state_dict` **之后** —— ckpt 里只有 1 个 param_group,
    #      先拆组会让组数不匹配 -> 恢复抛错 -> 整份优化器状态被丢弃 (主干动量一起丢)。
    #   2) 必须在 `scheduler.load_state_dict` **之后** —— 它的 base_lrs 长度也要匹配。
    #   3) 拆完组要**同步 append scheduler.base_lrs**, 否则 zip 截断 -> 新组永远
    #      不被调度 (lr 冻在初值, 不随 cosine 衰减)。
    _tt_scale = float(getattr(args, "triple_table_lr_scale", 0.0) or 0.0)
    if _tt_scale > 0:
        _tt_params, _tt_names = [], []
        _mm = model.module if hasattr(model, "module") else model   # DDP 已包装
        for _nm in ("y_callig_embedder", "y_script_embedder", "y_char_embedder"):
            _mod = getattr(_mm, _nm, None)
            if _mod is None:
                continue
            _ps = [p for p in _mod.parameters() if p.requires_grad]
            if _ps:
                _tt_params.extend(_ps)
                _tt_names.append(_nm)
        if not _tt_params:
            logger.warning("[optim] triple_table_lr_scale>0 但三表没有可训参数 (被冻结?)")
        else:
            _tt_ids = {id(p) for p in _tt_params}
            _pg0 = opt.param_groups[0]
            _taken = [p for p in _pg0["params"] if id(p) in _tt_ids]
            _pg0["params"] = [p for p in _pg0["params"] if id(p) not in _tt_ids]
            _new = {k: v for k, v in _pg0.items() if k != "params"}
            _new["params"] = _taken
            # ★ base 必须取 scheduler.base_lrs[0] (真实基准 lr = args.lr),
            #   不能读 _pg0["lr"]: 那是调度器刚初始化/步进后的**瞬时值**
            #   (实测读到 warmup 起点 1.67e-08, 再乘 0.1 = 1.67e-09 等于把表冻死)。
            _base0 = (float(scheduler.base_lrs[0])
                      if (scheduler is not None and getattr(scheduler, "base_lrs", None))
                      else float(_pg0["lr"]))
            _new["lr"] = _base0 * _tt_scale
            opt.add_param_group(_new)
            if scheduler is not None and getattr(scheduler, "base_lrs", None):
                scheduler.base_lrs.append(_base0 * _tt_scale)
            logger.info(
                f"[optim] 三表独立小 lr: base={_base0:.2e} x {_tt_scale} -> "
                f"{_new['lr']:.2e} (当前主 lr {float(_pg0['lr']):.2e}), "
                f"{sum(p.numel() for p in _taken):,} 参数, 模块={_tt_names}"
                + (" (已同步 scheduler.base_lrs)" if scheduler is not None else ""))

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
    # ★ 2026-10-14 [重构 PR4] 非有限 loss 的**跳过策略**收进组件 ✓
    #   （审查 §4.2 第 5 条 "finite 梯度检查" ✓；它是循环体里少见的**纯状态机** ✓）
    _guard = _NonFiniteGuard(logger, rank)
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
                    str(getattr(args, "eval_dir", "") or "").strip()
                    or os.path.join(checkpoint_dir, "..", "eval_only"),
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

    # ★ 2026-10-14 [重构 P1-2] **精确恢复 RNG** ✗
    #   审查: 旧 ckpt 只存 model/opt/ema/scheduler/args/step ✗ ⇒
    #     "恢复可以继续训练, 但**不保证逐步、逐样本可重放**" ✗
    #   而本项目经常比较 5k–10k 的**微小增益** ⇒ 这种差别很重要 ✓。
    #   位置: **训练循环之前** ✓ —— setup 期的随机消耗不再影响 ✓，
    #   且 DataLoader worker 的 seed 会从**恢复后**的基状态派生 ✓。
    if _resume_full_ckpt is not None:
        _rm = _resume_full_ckpt.get("rng_meta") or {}
        if rank == 0 and not _rm.get("epoch_aligned", True):
            logger.warning("[ckpt] ⚠ epoch_steps(%s) != ckpt_every(%s) ⇒ 恢复后"
                           "**数据相位可能错位** ✗（P1-2 提醒 ✓）",
                           _rm.get("epoch_steps"), _rm.get("ckpt_every"))
        _rng_ok = restore_rng(_resume_full_ckpt.get("rng") or {},
                              logger if rank == 0 else None)
        if rank == 0:
            logger.info("[ckpt] resume_kind=exact ✓ RNG %s ⇒ %s",
                        "完整恢复 ✓" if _rng_ok else "**未完整** ✗",
                        "**可逐样本重放** ✓" if _rng_ok
                        else "只能近似续训 ✗（旧 ckpt 无 rng 字段 ✓）")

    _train_fatal = None            # ★ 训练期致命异常（见 except 处）
    # ★ 2026-10-14 [重构 PR4 收官] 810 行训练循环搬进 src/train/trainer.py ✓
    #   （审查 §4.2 ✓）逐字搬移 ✓ 顶部重绑 44 个状态名 ✓
    #   ⇒ 等价性是**构造性**的 ✓（AST 已验证: 状态在循环内只读 ✓）
    _train_fatal = _training_loop(_TrainState(
    args=args,
        _SKEL_STRUCT_WARNED=_SKEL_STRUCT_WARNED,
        _acc_c12=_acc_c12,
        log_steps=log_steps,
        running_diff=running_diff,
        running_loss=running_loss,
        running_repa=running_repa,
        running_skel=running_skel,
        running_std_mid=running_std_mid,
        start_time=start_time,
        train_steps=train_steps,
        _EVAL_INLINE=_EVAL_INLINE,
        _aux_dirs=_aux_dirs,
        _ch_w=_ch_w,
        _ckpt=_ckpt,
        _critic_loss_fn=_critic_loss_fn,
        _empty_cache_at=_empty_cache_at,
        _epoch_steps=_epoch_steps,
        _epochs_needed=_epochs_needed,
        _es_check_every=_es_check_every,
        _guard=_guard,
        _inj_probe=_inj_probe,
        _inj_stats=_inj_stats,
        _mid_struct_loss=_mid_struct_loss,
        _skel_struct_loss_fn=_skel_struct_loss_fn,
        _std_soft_cond=_std_soft_cond,
        _stopper=_stopper,
        _strong_style=_strong_style,
        _style_anchor_k=_style_anchor_k,
        _style_anchor_mode=_style_anchor_mode,
        _style_anchor_target=_style_anchor_target,
        _style_anchor_w=_style_anchor_w,
        _style_ortho_w=_style_ortho_w,
        _style_rank_loss_fn=_style_rank_loss_fn,
        _style_table_param=_style_table_param,
        _train_vae_enc=_train_vae_enc,
        _vae_sf=_vae_sf,
        _vae_shift=_vae_shift,
        _writer_table_tuning=_writer_table_tuning,
        checkpoint_dir=checkpoint_dir,
        cond_mode=cond_mode,
        device=device,
        diffusion=diffusion,
        ema_model=ema_model,
        loader=loader,
        logger=logger,
        model=model,
        opt=opt,
        rank=rank,
        repa_loss_fn=repa_loss_fn,
        resume_start_step=resume_start_step,
        sampler=sampler,
        scheduler=scheduler,
        trainable_params_list=trainable_params_list,
        vae=vae))

    model.eval()
    # 等后台写盘收尾 —— 否则最后一个 ckpt 可能还没落盘进程就退了。
    if dist.get_rank() == 0:
        # ★ PR4: 收尾走 CheckpointManager ✓（内部仍是 drain_ckpt ✓
        #   strict=None ⇒ 默认**严格** ✓ 写盘失败会让进程非零退出 ✓ P1-3a ✓）
        _ckpt.finalize(timeout=600)
    if _train_fatal is not None:
        logger.error("训练**异常中止** ✗ —— 不打印 Done!, 并以非零码退出 ✓"
                     "（此前会静默打印 Done! 让失败看起来像成功 ✗）")
        cleanup()
        raise _train_fatal
    logger.info("Done!")
    cleanup()

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
