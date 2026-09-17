# -*- coding: utf-8 -*-
"""训练 CLI —— argparse 定义（从 train.py 抽出，2026-09-17）。

`train.py` 里有 **162 个** `add_argument`，连同"config 文件当默认值、CLI 覆盖"
的合并逻辑，一起占了主脚本约 500 行。抽到这里后 train.py 只留
`parse_args()` 一行调用。

对外只暴露两个函数：
    build_parser() -> ArgumentParser     # 只建 parser（不含 config 合并）
    parse_args(argv=None) -> Namespace   # 建 + 合并 config + 解析

**config 合并语义**（保持原样，勿改）：
    先用 `parse_known_args()` 只取 `--config`，读该 JSON，把其中出现的键
    **覆盖成对应 action 的 default** 并清掉 `required`；最后才真正 `parse_args()`。
    所以优先级是 **CLI > config > 代码默认值**。
    `_coerce` 负责把 JSON 里的值按 action.type 转换（JSON 没有 int/float 之分）。
"""
import argparse
import json
import os

from src.model import DiT_2Cond_models   # --model 的 choices 用它


def _coerce(value, template, target_type=None):
    """把 config JSON 里的值按 action 的 type 转换。

    JSON 只有 number/bool/string，而 argparse action 可能声明 type=int/float/
    _str_to_bool —— 不转换的话 "5000" 会以 str 形式进入训练代码（历史坑）。
    以 `template`（原 default）的类型为准；显式给了 `target_type` 则优先用它。
    """
    if value is None:
        return None
    if target_type is _str_to_bool:
        return _str_to_bool(value)
    if isinstance(template, bool):
        return _str_to_bool(value) if not isinstance(value, bool) else value
    if isinstance(template, int) and not isinstance(template, bool):
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    if isinstance(template, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    if isinstance(template, str) or template is None:
        if target_type in (int, float):
            try:
                return target_type(value)
            except (TypeError, ValueError):
                return value
        if isinstance(value, bool):
            return str(value)
        return value
    return value


def _str_to_bool(value):
    """argparse 的布尔类型 —— `type=bool` 会把 "False" 也判成 True。"""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def build_parser():
    """只构建 parser（不读 config、不解析）。"""
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
                        choices=["legacy", "factorized_add", "factorized_cat", "xl_highdim"],
                        default="legacy",
                        help="Cond fusion: legacy joint MLP | factorized_add (low-dim additive) | "
                             "factorized_cat (v12: 各向量因子 embedding 拼接后联合投影, ref/Moyun 式) | "
                             "xl_highdim (high-dim, XL-aligned, preserves pretrained adaLN).")
    parser.add_argument("--callig-embed-dim", type=int, default=None)
    parser.add_argument("--script-embed-dim", type=int, default=None)
    parser.add_argument("--char-embed-dim", type=int, default=None)
    parser.add_argument("--glyph-vec-cond", type=_str_to_bool, default=False,
                        help="v12: 把 g(标准字形) 池化成全局内容向量, 作为条件向量 c 的"
                             "第二个操作数。仅 factorized_add / factorized_cat 下生效。"
                             "作用: 让 adaLN 调制分支第一次能看到内容"
                             "(原先 c = t_emb + callig_proj, 完全不知道在写哪个字);"
                             "同时让 factorized_cat 不再退化成单层 Linear。")
    parser.add_argument("--glyph-vec-dim", type=int, default=128,
                        help="g 全局内容向量的维度 (concat/add 的第二个操作数)")
    parser.add_argument("--glyph-vec-pool", type=str, choices=["mean", "max"],
                        default="mean", help="g_tok 256 个 token 的池化方式")
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
    parser.add_argument("--use-checkpoint", type=_str_to_bool, default=False,
                        help="Enable gradient checkpointing on DiT blocks (cuts activation memory).")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-calligraphers", type=int, default=2021)
    parser.add_argument("--callig-id-map", default="",
                        help="干净书家词表映射 json 路径 (raw calligrapher_id -> 0..N-1)。"
                             "设置后 num_calligraphers 自动收紧为词表长度, 数据层查表映射。"
                             "见 tools/build_callig_map.py。空=保持现状(稀疏 id 直通)。")
    parser.add_argument("--callig-emb-pretrained", default="",
                        help="对比预训练的书家 embedding (.pt, 含 'embedding' (N,dim))。"
                             "加载到 y_callig_embedder 表前 N 行, 配合 --freeze-callig-table")
    parser.add_argument("--callig-spatial", action="store_true",
                        help="旧外挂(已证伪死重): 书家向量->r 系数 x (r,256,D) 空间基图 加到骨架。"
                             "保留为可配置开关以复评历史 ckpt。")
    parser.add_argument("--callig-spatial-rank", type=int, default=64,
                        help="外挂低秩基图数量 r (callig_spatial_net 输出维度)")
    parser.add_argument("--aux-latent-shards-dirs", type=str, default="",
                        help="moyi 式辅助目标通道: 逗号分隔 aux latent shard 目录 "
                             "(如 data/aux/aux_skel_latents_fame_e,data/aux/aux_canny_latents_fame_e)。"
                             "训练目标 x = cat(image, *aux), 对全部通道加噪/算 MSE; 推理只用前 4 通道。")
    parser.add_argument("--aux-loss-weight", type=float, default=1.0,
                        help="aux 通道 loss 权重 (1.0=等权/ref; <1 时图像主导)")
    parser.add_argument("--aux-loss-weights", type=str, default="",
                        help="per-group aux 权重, 逗号分隔, 与 aux-latent-shards-dirs 同序。"
                             "如 '0.3,0.8' -> canny×0.3, skel×0.8 (覆盖 --aux-loss-weight)。")
    # ⚠ 必须注册: 未注册的参数会被 config 加载器**静默丢弃**
    #   (main_from_cli 只把 config 的键套到已注册的 action 上), getattr 取到 False,
    #   于是 decode 前不加回白底 -> 全图发黄/发黑 (2026-09-14 事故根因)。
    parser.add_argument("--aux-zero-white", type=_str_to_bool, default=False,
                        help="白底归零: 训练目标已减去白底 latent (见 tools/rebuild_latents_wz.py), "
                             "评测/推理 decode 前**必须加回**, 否则 latent 的零向量会被 VAE 解成"
                             "灰黄棕色 (实测 RGB≈[129,110,89]) -> 整图发黄、更负处发黑。")
    parser.add_argument("--callig-style-attn", action="store_true",
                        help="callig 风格 cross-attention(书家化骨架正确形态): 书家向量 -> N_style 个 "
                             "style token, 骨架 token 内容寻址聚合风格, 产生'书家x字x位置'交互(结体差异)。"
                             "zero-init 可 resume。")
    parser.add_argument("--callig-n-style", type=int, default=8,
                        help="callig style token 数量 (配合 --callig-style-attn)")
    parser.add_argument("--glyph-inject-mode", choices=["adaln", "xattn"], default="adaln",
                        help="g 逐层注入方式: adaln=ZeroAdaLN 固定位置调制 (旧默认), "
                             "xattn=ZeroCrossAttention 空间寻址 (GlyphDraw 式, 新 ckpt 专用)")
    parser.add_argument("--cfg-glyph-scale", type=float, default=None, dest="cfg_glyph_scale",
                        help="[2026-09-17] 双轴 CFG 的**内容轴**(骨架 g)引导强度。\n"
                             "默认 None = 经典 2 路 CFG(只引导书家风格)。\n"
                             "非 None 时走 forward_with_2axis_cfg: 4 个 pass 分别\n"
                             "  full / style(y_callig, g=0) / content(null, g) / uncond(null, g=0)\n"
                             "⚠ **前提: 训练时 glyph_drop_prob > 0** —— g=0 必须是训练见过的\n"
                             "  条件, 否则内容轴未训练(与 callig null 行同一类问题)。\n"
                             "⚠ 4× NFE。")
    parser.add_argument("--cfg-w-inter", type=float, default=0.0, dest="cfg_w_inter",
                        help="双轴 CFG 的交互项权重 (full - content - style + uncond)。默认 0。")
    parser.add_argument("--xattn-q-pos", type=_str_to_bool, default=False, dest="xattn_q_pos",
                        help="[2026-09-17] xattn 的 **Q 是否也加 sincos 位置嵌入**。\n"
                             "默认 False = 旧行为(只给 K/V 加位置)。\n"
                             "⚠ 旧实现下 Q 无位置, 而 rope=True 时 x 残差流不加绝对位置\n"
                             "-> '空间寻址'退化成'内容寻址', 与 ZeroCrossAttention 的\n"
                             "docstring 声称的 2D 绑定保证不符。打开后 Q/K/V 都带位置。\n"
                             "建议与旧实现做 A/B (同预算、跑到平台)。")
    parser.add_argument("--style-token-n", type=int, default=0,
                        help="风格 token 数 (0=关闭): >0 时每层注入 context = "
                             "[书家化骨架(+2D位置); 风格token(+可学习role)], "
                             "使书家风格**直接参与每一层、每个空间位置**的内容寻址 "
                             "(风格局部化, 与局部字形共同调制最终生成)。"
                             "配 --glyph-inject-mode xattn 使用; 建议 16~64 并做容量扫描。")
    parser.add_argument("--style-role-init", type=float, default=0.02,
                        help="风格 token 的 role embedding 初始化 std (促 N 个 token 分化, "
                             "避免塌缩为同一向量)")
    parser.add_argument("--freeze-callig-table", action="store_true",
                        help="冻结书家表 [0,N) 行 (CFG null token 仍可训练)。"
                             "需先 --callig-emb-pretrained, 否则冻结随机初始化无意义")
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
                        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                        help="torch.compile mode: default / reduce-overhead (CUDA-graph, faster but "
                             "higher mem) / max-autotune / max-autotune-no-cudagraphs (autotuned GEMM "
                             "kernels without CUDA graphs — safe with the concurrent eval process).")
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
    parser.add_argument("--fresh-scheduler", type=_str_to_bool, default=False,
                        help="With --resume-full: ignore the restored scheduler state and "
                             "rebuild the LR schedule over the remaining fine-tune horizon "
                             "(max_steps - resume step) instead of continuing the old one.")
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
    # [2026-09-16] epoch 长度(步)。0 = 自动取 ckpt_every(推荐): epoch 与存盘/评测
    # 刻度重合, DataLoader 迭代器 reset 的代价被 ckpt+eval 吸收, 不再每 119 步抖一次。
    # 负值 = 强制退回旧行为(epoch = 数据集一遍)。
    parser.add_argument("--epoch-steps", type=int, default=0, dest="epoch_steps",
                        help="**已合并到 --ckpt-every（兼容别名，不再单独生效）**。\n"
                             "两者语义上本就是同一个刻度: epoch 长度 = 存盘周期 = 评测周期。\n"
                             "配了就**必须与 ckpt_every 相等**，不等直接拒绝启动；\n"
                             "只配这一个（ckpt_every 缺省）时会被采纳。\n"
                             "合并的好处: 存盘条件退化成一行, eval 点 == 存盘点恒成立,\n"
                             "deferred eval 的前置条件自动满足, 不再需要任何断言。\n"
                             "见 src/utils/samplers.py:LongEpochDistributedSampler。")
    parser.add_argument("--use-ema", type=_str_to_bool, default=False,
                        help="Maintain and evaluate a full-model exponential moving average.")
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--ema-interval", type=int, default=1, dest="ema_interval",
                        help="Update EMA every N steps with decay**N (mathematically "
                             "equivalent to per-step update, saves ~370MB/step of "
                             "memory bandwidth for a 46M-param model).")
    parser.add_argument("--ema-warmup", type=_str_to_bool, default=True,
                        help="Cap early EMA decay by update count to avoid random-init lag.")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--vae-path", type=str, default="data/pretrained/sd-vae-ft-ema", help="Local path to VAE weights")
    parser.add_argument("--vae-downscale", type=int, default=8, help="VAE spatial downsample factor (8=f8 sd-vae, 4=f4 kl-f4)")
    parser.add_argument("--latent-channels", type=int, default=4, help="VAE latent channel count (4=sd-vae, 3=kl-f4)")
    parser.add_argument("--image-channels", type=int, default=None,
                        help="CFG 作用域: 只对这些通道做 classifier-free guidance, 其余通道"
                             "(aux 结构通道) 保持 cond 分支原值。默认 None = 取 --latent-channels。"
                             "⚠ 12ch 下必须保持 4: aux 分布与 image 不同, 若被同一 cfg_scale "
                             "放大, 采样轨迹会跑飞 (doc59 §1.3 '墨团'根因)。"
                             "此键此前未注册 -> config 里写 image_channels 会被静默丢弃。")
    parser.add_argument("--vae-in-channels", type=int, default=3, help="VAE input image channels (3=RGB, 1=grayscale)")
    parser.add_argument("--vae-out-channels", type=int, default=3, help="VAE output image channels (3=RGB, 1=grayscale)")
    parser.add_argument("--vae-scaling-factor", type=float, default=0.18215, help="VAE latent scaling factor")
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
    parser.add_argument("--ckpt-every", type=int, default=10_000,
                        help="**唯一刻度**（2026-09-17 合并 epoch_steps 后）:\n"
                             "  ① 每多少步存一次 ckpt\n"
                             "  ② epoch 长度（DataLoader 迭代器 reset 点）\n"
                             "  ③ 评测点（in-mem / in-process eval 都挂在这个点上）\n"
                             "三者重合的意义: reset 代价被存盘+评测吸收（不再有独立锯齿），\n"
                             "且每个评测点必有 ckpt -> 可用 --eval-mode deferred。\n"
                             "0 = 不存盘, 且 epoch 退回\"数据集一遍\"的旧行为。")
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
    parser.add_argument("--w-glyph-cond", type=_str_to_bool, default=False,
                        help="Enable 甲2 standard-glyph token-add conditioning (use_glyph_cond).")
    parser.add_argument("--glyph-scale-init", type=float, default=0.4,
                        help="Initial glyph_scale (standard-glyph token-add strength).")
    parser.add_argument("--no-char-cond", type=_str_to_bool, default=False,
                        help="v10b: 移除 char 向量条件 (skel-g 即字条件, 因子分解=callig+skel)."
                             " 支持 factorized_add / factorized_cat。"
                             " 注意: 开启后向量因子只剩 callig 一个, factorized_cat 会退化为"
                             " 单层 Linear (与 factorized_add 等价), 见 dit.py 注释。")
    parser.add_argument("--skel-as-glyph-cond", type=_str_to_bool, default=False,
                        help="v10a: 用实例 skel latent (skel_latent_shards_dir) 走 g 通路"
                             " (use_glyph_cond 注入), 替代标准字形库——skel latent 即字条件, 从头预训练.")
    parser.add_argument("--skel-latent-shards-dir", default="",
                        help="实例 skel latent shards (--skel-as-glyph-cond 时必填)")
    parser.add_argument("--glyph-drop-prob", type=float, default=0.0,
                        help="g 条件训练期随机丢弃概率 (skel 模式建议 0.1, 保无 g 生成能力)")
    parser.add_argument("--glyph-inject-layers", type=int, default=0,
                        help="g 逐层注入层数 (0=仅输入层 token-add, s23 既有行为)")
    parser.add_argument("--glyph-embedder-depth", type=int, default=0,
                        help="g 编码器增强深度 (0=单层 Conv 现状; >0=降采样 Conv + N 层 "
                             "(SiLU+Conv3x3) 残形增强 std 骨架特征, 治 std-skel g 通路不激活).")
    # [v12+] FLOP 实测: depth=2 的两层满秩 3x3 conv 占全模型 9.7%。
    # depthwise-separable 同感受野只需 1/8.8 的代价 (省约 8.6% 总 FLOPs)。
    # ⚠ 这是**架构替换**, 理论上低风险(输入是近二值细线, 信息量极低), 但**未经验证**,
    #   必须 A/B 确认不伤质量 —— 不要因为"看起来等价"就直接切。
    parser.add_argument("--glyph-embedder-sep", type=_str_to_bool, default=False,
                        dest="glyph_embedder_sep",
                        help="g 编码器的 3x3 conv 用 depthwise-separable 实现 "
                             "(省 ~8.6%% 总 FLOPs, 需 A/B 验证质量)。")
    parser.add_argument("--glyph-init-mix", type=float, default=0.0,
                        help="HYBRID 初始点 alpha∈[0,1]: xT=alpha*randn+(1-alpha)*std字形latent。"
                             "0=纯噪声(现状); (0,1)=混合; 默认 0 保持当前行为, 收敛后按需设 e.g.0.6。"
                             "见 HYBRID_INIT_PLAN.md。")
    parser.add_argument("--w-std-mid", type=float, default=0.0,
                        help="MIDSTEP_STD 权重: 在中间噪声水平 sqrt(alpha_cumprod)∈[alo,ahi] 时,"
                             "额外监督 模型预测 clean latent 逼近标准字形 latent g, 让字形中段锚定。"
                             "需 w-glyph-cond 开启。权重明显小于主 loss(如 0.1~0.5), 防抹掉风格。0=关。")
    parser.add_argument("--w-latent-skel", type=float, default=0.0, dest="w_latent_skel",
                        help="实例骨架结构 loss 权重(辅助项, 建议 0.02~0.1, 绝不等权)。\n"
                             "用**冻结**的小 probe 把 pred_xstart 映射成实例骨架 latent, 与 GT\n"
                             "skel_latent 做 MSE; 只在 t<=--latent-skel-max-t 生效。\n"
                             "与 12ch 的本质区别: 不在扩散目标里 -> 4ch 主干/CFG 作用域/推理成本\n"
                             "全不受影响, 且**新增可训练参数 0**(probe 冻结)。0=关。")
    parser.add_argument("--latent-skel-probe", type=str, default="", dest="latent_skel_probe",
                        help="冻结 probe ckpt 路径 (LatentSkelProbe: 4ch img latent -> "
                             "4ch 实例骨架 latent)。由 tools/train_latent_skel_probe.py 训练。")
    parser.add_argument("--inst-skel-shards-dir", type=str, default="", dest="inst_skel_shards_dir",
                        help="**实例骨架** latent shards 目录 (结构 loss 的 target)。由 GT 图派生 "
                             "(tools/build_skel_latents.py), 与条件 g (skel_latent_shards_dir="
                             "shards_std) 完全解耦。w_latent_skel>0 时必填, 否则拒绝启动。")
    parser.add_argument("--latent-skel-max-t", type=float, default=0.3, dest="latent_skel_max_t",
                        help="结构 loss 的 t 门控上界。**flow 下 t in [0,1] 且 t=0 是干净端**\n"
                             "(flow_matching.py:14), 故 0.3 = 只监督最干净的 30%% 时刻。\n"
                             "⚠ 必须 float: 照抄 DDPM 的 500 会被 int() 截成 0 -> 门控恒假\n"
                             "-> loss 静默恒为 0 (静默失效惯犯)。")
    parser.add_argument("--std-mid-alo", type=float, default=0.35,
                        help="中间噪声带下界(sqrt_alpha_cumprod), 默认 0.35。")
    parser.add_argument("--std-mid-ahi", type=float, default=0.75,
                        help="中间噪声带上界(sqrt_alpha_cumprod), 默认 0.75。")
    parser.add_argument("--w-repa", type=float, default=0.0, help="Weight for Representation Alignment (REPA) Loss (0 = disabled, default)")
    parser.add_argument("--repa-teacher-ckpt", type=str, default="",
                        help="Local path to DINOv2 teacher weights (ModelScope safetensors). "
                             "Empty = auto-detect data/pretrained/dinov2_vits14_pretrain.safetensors or $DINO_WEIGHTS.")
    # ── 显存"空洞"回收（2026-09-17）────────────────────────────────────────
    # 背景: `torch.compile` 在前几十步会把 allocator 的高水位顶到远超真实活跃需求
    #   （实测 xattn @ batch240: 活跃 14.42G, 高水位 20.51G, **空洞 6.09G / 42%**）。
    #   这个空洞一直白占着，直到某个 `empty_cache()` 才还给驱动。
    #   后果: 决定能否上更大 batch 的是**高水位**而不是活跃需求 ——
    #   batch360 的活跃需求(22.2G)其实装得下，但高水位(~31G)装不下 -> OOM。
    # 修法: warmup 结束后调一次 `torch.cuda.empty_cache()`，把空洞还给驱动。
    parser.add_argument("--empty-cache-after-warmup", type=int, default=50,
                        dest="empty_cache_after_warmup",
                        help="在第 N 步之后调一次 torch.cuda.empty_cache()，回收 torch.compile "
                             "warmup 期间 allocator 囤积的显存空洞。0 = 关闭。"
                             "实测可回收 ~6G（高水位 20.51G -> 14.42G），"
                             "是把 batch 从 240 提到 360 的前提。")
    parser.add_argument("--repa-cache-dir", type=str, default="", dest="repa_cache_dir",
                        help="Dir with pre-extracted DINOv2 teacher features (feats.f16 + ids.npy, "
                             "built by tools/build_dino_cache.py). Hits skip the per-step DINO "
                             "forward (+15~20% throughput, -1.5GB VRAM); misses fall back to the "
                             "teacher forward (lazy-loaded). Empty = disabled (teacher every step).")
    # [v12 修复] 该键此前**未注册为 argparse 参数**, 代码里用
    #   getattr(args, "repa_layers", "") 读取 -> config JSON 里的值被**静默丢弃**,
    #   实际永远走 train.py 的硬编码兜底 `or (8,)`。v11/v12 恰好都想要 8 所以没暴露,
    #   但想改 REPA 层位时改了不生效。默认 "8" 与旧兜底**行为完全一致**。
    parser.add_argument("--repa-layers", type=str, default="8",
                        help="REPA 对齐的中间层, 单层 '8' 或多层 '8,11'。"
                             "⚠ 此键曾未注册导致 config 值被静默丢弃 (doc56/59 同类坑)。")
    parser.add_argument("--eval-skel-latent-shards-dir", type=str, default="",
                        dest="eval_skel_latent_shards_dir",
                        help="评测专用的标准字形(g) latent 目录。⚠ 必须与训练侧的 "
                             "--skel-latent-shards-dir 分开: 训练 shard 按 **train csv 的 img_id** 建, "
                             "而 eval 集的 img_id 是另一套。实测: 用 base_sym(train) 查 strict 命中 0/237 "
                             "-> g 全零 -> VAE decode(0) 解出灰黄棕 [129,110,89] -> poster 首行发黄, "
                             "且 strict 指标完全失真(曾出现 strict 0.4837 > seen 0.4577 的反常)。"
                             "留空则退回 --skel-latent-shards-dir。")
    parser.add_argument("--in-mem-eval", type=_str_to_bool, default=False, dest="in_mem_eval",
                        help="True in-mem eval inside the training process: at each ckpt point, "
                             "pause stepping, sample with the resident EMA model on the same GPU, "
                             "decode + compute SSIM/MSE in-memory, append eval_stdskel_*.csv "
                             "(batch_eval-compatible), then resume. No daemon, no PNGs, no ckpt reload.")
    parser.add_argument("--eval-mode", type=str, default="inline", dest="eval_mode",
                        choices=["inline", "deferred"],
                        help="**inline**: 训练中按 eval 周期就地评测（默认，现状）。\n"
                             "**deferred**: 训练循环**完全不 eval**，只存 ckpt；训练结束后用\n"
                             "  `python -m src.eval.batch_eval --results-dir <dir> --sets ...` 批量补跑。\n"
                             "动机（docs/system/70 §7）: 单卡时 eval 会独占 GPU 使训练停摆\n"
                             "（实测 40 次 x 42s = 28 分钟）；批量跑还能把 40 个 batch-60 的小作业\n"
                             "合成 1 个 batch-240 的大作业，GPU 利用率高得多。\n"
                             "⚠ 前提: **每个 eval 点都必须有 ckpt**（即 ckpt 周期整除 eval 周期）。\n"
                             "  启动时会断言 `ckpt_every % epoch_steps == 0`，不满足直接拒绝。")
    parser.add_argument("--in-mem-eval-sets", type=str, default="",
                        dest="in_mem_eval_sets",
                        help="Sets spec for --in-mem-eval: 'name:csv:n' comma-separated. "
                             "Default: seen:assets/eval_seen_v10.csv:10,"
                             "strict:assets/eval_fame3_strict_clean_v9.csv:50")
    parser.add_argument("--in-mem-eval-batch", type=int, default=16, dest="in_mem_eval_batch",
                        help="DiT sampling batch for --in-mem-eval (VRAM-safe: 16).")
    parser.add_argument("--in-mem-eval-save-samples", type=_str_to_bool, default=True,
                        dest="in_mem_eval_save_samples",
                        help="Save generated + GT PNGs to eval_samples_ctrl/step{N}/{set}/ "
                             "during --in-mem-eval (poster/sanity use).")
    # [v12+] LPIPS 此前是"声明了但从没算过"的空列。ssim 会被大面积白底匹配骗过
    # (doc59: 墨团也能拿 0.48), LPIPS 对结构细节敏感 —— 是区分"容量够不够"与
    # "ssim 饱和"的关键指标 (doc62 §3)。跑在 CPU, 不与训练争显存。
    parser.add_argument("--in-mem-eval-lpips", type=_str_to_bool, default=True,
                        dest="in_mem_eval_lpips",
                        help="计算 LPIPS(vgg, CPU) 并写入 summary/batch CSV 的 lpips 列。"
                             "不可用时自动留空且不影响评测。")
    parser.add_argument("--eval-self-cond", type=_str_to_bool, default=False, dest="eval_self_cond",
                        help="Two-pass self-conditioning sampling in --in-mem-eval "
                             "(pass-1 predicted skel channels fed back as g for pass-2).")
    parser.add_argument("--eval-blend-alpha", type=float, default=0.0, dest="eval_blend_alpha",
                        help="Self-conditioning skeleton blend (0=pure predicted, 0.5=half-half).")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon"],
                        help="Optimizer: adamw (default) or muon (matrix NS-orth + adamw for vec/embed).")
    parser.add_argument("--muon-lr", type=float, default=0.02,
                        help="Muon matrix-group LR (independent scale, ~0.01-0.1; adamw uses --lr).")
    parser.add_argument("--latent-shards-dir", type=str, default=None,
                        help="Dir of pre-built latent shards (shard_XXXXX.npz). If set, training reads "
                             "pre-encoded VAE latents instead of on-the-fly VAE encode.")
    parser.add_argument("--img-root", type=str, default="data/imgs/final_imgs_256",
                        help="Root dir of 256x256 gt images (used with latent-cached training for gt-losses).")
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

    return parser


def parse_args(argv=None):
    """建 parser -> 用 --config 的 JSON 覆盖默认值 -> 解析并返回 args。

    优先级: **CLI > config 文件 > 代码默认值**。
    """
    parser = build_parser()

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

    return parser.parse_args(argv)
