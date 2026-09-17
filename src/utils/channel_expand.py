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
