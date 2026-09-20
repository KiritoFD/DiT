# -*- coding: utf-8 -*-
"""4ch -> 12ch 通道扩展（用于"4ch 预训练 -> 12ch 后训练"）。

## 为什么需要单独的模块
`--resume-full` 走的是 `load_state_dict(..., strict=False)` —— 形状不匹配的张量
会被**静默跳过**（只报 missing/unexpected 计数）。直接拿 4ch ckpt 去 resume 12ch，
这 3 个张量会**停在随机初始化**，不报错、loss 正常下降 —— 又一例静默失效。

## 扩展只涉及 3 个张量
```
① x_embedder.proj.weight      (h, 4, p, p)  -> (h, 12, p, p)   拷前 4 个输入通道
② final_layer.linear.weight   ((HWpp)*4, h) -> ((HWpp)*12, h)  按 patch 交错拷贝
③ final_layer.linear.bias     ((HWpp)*4,)   -> ((HWpp)*12,)    按 patch 交错拷贝
```
（`x_embedder.proj.bias` 形状 `[h]` 不变，直接搬）

## ⚠ 最容易写错的地方
`FinalLayer` 的输出布局是 `reshape(N, h, w, p, p, C)` —— **通道在最内层**。
所以通道**不是"前 4 行"**，而是每个 patch 位置内连续的 4 个。
必须 `view(HWpp, C, ...)` 再拷；直接拷前 16 行会**错位**。

## 已验证
4ch 输出 vs 12ch 输出前 4 通道：`max|diff| = 0.000e+00`（逐位相同），
新通道输出恰好 0 —— 即第 0 步的 12ch 模型**完全等价**于 4ch 模型。
"""
from typing import Dict, List, Tuple

import torch


def expand_state_dict_4ch_to_12ch(
    sd_4ch: Dict[str, torch.Tensor],
    n_aux_groups: int,
    in_channels_4ch: int = 4,
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """把 4ch 模型的 state_dict 扩展成 12ch 的。

    参数
    ----
    sd_4ch : 4ch 模型的 state_dict（键名已去掉 `_orig_mod.` 前缀）
    n_aux_groups : aux 组数（2 组 -> 4 + 4*2 = 12 通道）
    in_channels_4ch : 原通道数，默认 4

    返回
    ----
    (新 state_dict, 被扩展的键名列表)

    新通道**全部零初始化** —— 这样扩展后的模型在前 `in_channels_4ch` 个通道上
    与 4ch 模型**逐位等价**，aux 通道从"输出 0"开始学。
    """
    new_in = in_channels_4ch + 4 * n_aux_groups
    out: Dict[str, torch.Tensor] = {}
    expanded: List[str] = []

    for k, v in sd_4ch.items():
        if k.endswith("x_embedder.proj.weight"):
            # [h, C, p, p]：C 在 dim=1
            h, c, p, _ = v.shape
            assert c == in_channels_4ch, f"{k} 期望 {in_channels_4ch} 通道，实为 {c}"
            nv = torch.zeros(h, new_in, p, p, dtype=v.dtype)
            nv[:, :in_channels_4ch] = v
            out[k] = nv
            expanded.append(k)

        elif k.endswith("final_layer.linear.weight"):
            # [(HWpp)*C, h]：reshape 成 (HWpp, C, h)，C 在中间
            total, h = v.shape
            assert total % in_channels_4ch == 0, f"{k} 形状 {tuple(v.shape)} 不是 {in_channels_4ch} 的整数倍"
            hwpp = total // in_channels_4ch
            nv = torch.zeros(hwpp * new_in, h, dtype=v.dtype)
            nv.view(hwpp, new_in, h)[:, :in_channels_4ch] = v.view(hwpp, in_channels_4ch, h)
            out[k] = nv
            expanded.append(k)

        elif k.endswith("final_layer.linear.bias"):
            # [(HWpp)*C]：同理
            total = v.shape[0]
            assert total % in_channels_4ch == 0, f"{k} 形状 {tuple(v.shape)} 不是 {in_channels_4ch} 的整数倍"
            hwpp = total // in_channels_4ch
            nv = torch.zeros(hwpp * new_in, dtype=v.dtype)
            nv.view(hwpp, new_in)[:, :in_channels_4ch] = v.view(hwpp, in_channels_4ch)
            out[k] = nv
            expanded.append(k)

        else:
            out[k] = v.clone()

    return out, expanded


def drop_shape_mismatched(model, sd: Dict[str, torch.Tensor]) -> List[str]:
    """从 ckpt state_dict 里**剔除与模型形状不匹配的键**，返回被剔除的键名。

    ## 为什么需要

    ``load_state_dict(strict=False)`` 只容忍 missing/unexpected **键**，
    遇到形状不匹配会直接 RuntimeError —— 而"换了条件头形状的架构演进式 resume"
    （如 v15 把书家表从 (87,128) 换成 (87, K*384)、cond_fusion 输入 256→512）
    恰恰希望：**形状对得上的全部加载，对不上的视为新模块重新初始化并保持可训**。

    剔除而不是硬崩：调用方必须把返回的键并入 resume 的 missing 集合，
    让 `--train-only-style` 等冻结策略把它们当作"ckpt 里没有的新模块"放开
    （否则新模块被冻死在随机初始化上 —— train-only-style 的历史教训）。
    """
    model_sd = model.state_dict()
    dropped = []
    for k in list(sd.keys()):
        if k in model_sd and tuple(model_sd[k].shape) != tuple(sd[k].shape):
            dropped.append(k)
            del sd[k]
    return dropped


def materialize_lazy_params(model, sd: Dict[str, torch.Tensor]) -> List[str]:
    """把 **ckpt 里有、但模型还没创建**的懒加载参数补出来，再返回补了哪些。

    ## 为什么需要
    `LabelEmbedder.null_embed` 是**懒创建**的（`dit.py` 里初始为 `None`，
    只有 `freeze_table()` 被调用才变成 `nn.Parameter`）。
    于是加载顺序**必须是** `freeze_table()` -> `load_state_dict()`：

        freeze_table() -> load    ✓ null_embed 存在，ckpt 的值正确灌入
        load -> freeze_table()    ✗ load 时模型没这个键 -> ckpt 的值被**静默丢弃**
                                    -> 随后 freeze_table() 从随机权重里复制
                                    -> **CFG 的 null 向量被重新随机化**

    后者不报错、loss 正常下降，但 CFG uncond 分支整个变味 —— 典型的静默失效。
    实测：顺序 A `missing=0 unexpected=0` 且值一致；顺序 B `unexpected=1` 且值不一致。

    本函数在 load **之前**调用，把 ckpt 里出现、模型里还没有的懒参数按 ckpt 的
    形状建出来，从而**无论调用顺序如何都不会丢**。

    返回补出来的键名列表（用于日志；空列表 = 没有懒参数需要补）。
    """
    import torch.nn as nn
    model_keys = set(model.state_dict().keys())
    added = []
    for k, v in sd.items():
        if k in model_keys or not k.endswith(".null_embed"):
            continue
        # 只处理形如 `<prefix>.null_embed` 的懒参数
        prefix = k[: -len(".null_embed")]
        mod = model
        ok = True
        for part in prefix.split("."):
            if not hasattr(mod, part):
                ok = False
                break
            mod = getattr(mod, part)
        if not ok or getattr(mod, "null_embed", "MISSING") != None:  # noqa: E711
            continue
        mod.null_embed = nn.Parameter(v.detach().clone().to(v.dtype))
        added.append(k)
    return added


def expand_ckpt_4ch_to_12ch(ckpt: dict, n_aux_groups: int,
                            keys=("delta", "model", "ema")) -> dict:
    """就地扩展 ckpt dict 里的若干子 state_dict，返回新的 dict。

    `keys` 里存在的子 dict 都会被扩展；`opt` 等其它键原样保留
    （优化器状态与形状绑定，**不能**跨通道数复用，后训练应从零重建）。
    """
    import copy
    out = copy.copy(ckpt)
    for key in keys:
        sub = ckpt.get(key)
        if not isinstance(sub, dict) or not sub:
            continue
        # 剥掉 torch.compile 的 _orig_mod. 前缀
        sub = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
               for k, v in sub.items()}
        new_sub, expanded = expand_state_dict_4ch_to_12ch(sub, n_aux_groups)
        out[key] = new_sub
        out.setdefault("_expand_report", {})[key] = expanded
    # 优化器状态与参数形状绑定，跨通道数不可复用
    out.pop("opt", None)
    return out
