# -*- coding: utf-8 -*-
"""optimizer.py — **优化器装配的纯决策与薄工厂**（2026-10-14 [重构 PR4] ✓）

审查 §4.2 把 optimizer 列为需要"明确状态与生命周期"的组件之一 ✓。

★ 为什么这里**只抽纯决策 + 薄工厂**，而不整体搬迁 ✗:
  `train.py` 的优化器区（约 120 行）**不是纯构造** ✗ —— 它中间夹着**模型突变** ✓
  （如 `_p.requires_grad_(deform_trainable)` ✓）与**三条顺序约束** ✓
  （原文: "① 必须在 `opt.load_state_dict` **之后** … ② 必须在
   `scheduler.load_state_dict` **之后** … ③ 拆完组要同步 append `scheduler.base_lrs`" ✓）。
  整体搬迁会把这些**隐式契约**变成跨文件的隐式契约 ✗ ⇒ 风险大于收益 ✗。
  ⇒ 本模块只接管"**没有副作用**"的部分 ✓：名字解析 ✓ fused 支持判定 ✓
    构造两个优化器 ✓ 身份摘要 ✓。**顺序与突变仍留在调用方** ✓（原地不动 ✓）。

★ 顺带消除一处**重复实现** ✗:
  `fused` 支持判定在 `train.py` 里**写了两遍** ✗（VAE 双轨分支与普通分支各一份 ✓
  `bool(torch.cuda.is_available()) and float(torch.__version__[:3]) >= 2.0` ✓）
  ⇒ 现在只有**一个** `fused_supported()` ✓（改一处即两处生效 ✓ 不会漂移 ✓）
"""
from __future__ import annotations

import torch

__all__ = ["resolve_name", "fused_supported", "build_adamw", "build_adamw_groups",
           "build_muon", "identity", "describe"]


def resolve_name(args) -> str:
    """优化器名 ✓（缺省 `adamw` ✓ 与旧实现 `getattr(args,'optimizer','adamw')` 一致 ✓）。"""
    return str(getattr(args, "optimizer", "adamw") or "adamw").lower()


def fused_supported() -> bool:
    """是否可用 **fused AdamW** ✓ —— 逐字等价于旧的两处内联判定 ✓。

    旧写法（重复两次 ✗）:
        bool(torch.cuda.is_available()) and float(torch.__version__[:3]) >= 2.0
    ⚠ 保留**完全相同的**版本比较语义 ✓（`[:3]` 截断 ✓）⇒ 行为不变 ✓
      （换成 `packaging.version` 会更稳健 ✓ 但那是**行为变更** ✗ ⇒ 不在本次重构范围 ✓）
    """
    return bool(torch.cuda.is_available()) and float(torch.__version__[:3]) >= 2.0


def build_adamw(params, lr: float, weight_decay: float, logger=None, tag: str = ""):
    """构造 **AdamW**（单一参数组 ✓）—— 带 `fused` 的 TypeError 回退 ✓ 与旧代码逐字一致 ✓。

    旧写法（普通分支 ✓）:
        _fused = bool(torch.cuda.is_available()) and float(torch.__version__[:3]) >= 2.0
        try:
            opt = torch.optim.AdamW(trainable_params_list, lr=args.lr,
                                    weight_decay=args.weight_decay, fused=_fused)
        except TypeError:
            opt = torch.optim.AdamW(trainable_params_list, lr=args.lr, weight_decay=args.weight_decay)
            _fused = False
    返回 `(opt, fused_used)` ✓ —— 与旧代码一样把**实际是否启用**如实回传 ✓。
    """
    fused = fused_supported()
    try:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=fused)
    except TypeError:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        fused = False
    if logger is not None and tag:
        logger.info(f"[optim] {tag} lr={lr} wd={weight_decay} fused={fused}")
    return opt, fused


def build_adamw_groups(groups, logger=None, tag: str = ""):
    """构造 **AdamW**（多参数组 ✓ **组内已自带 lr / weight_decay** ✓）。

    ⚠ **绝不**传 `lr` / `weight_decay` ✗ —— 传了会把各组的值**统一覆盖** ✗✗
      （VAE 双轨就靠三组各自的 lr 区分: DiT `args.lr` ✓ REPA `args.lr` ✓ VAE `vae_lr` ✓）
    旧写法（VAE 双轨分支 ✓）:
        opt = torch.optim.AdamW(optim_groups, fused=_fused)
        # TypeError 回退: torch.optim.AdamW(optim_groups)
    """
    fused = fused_supported()
    try:
        opt = torch.optim.AdamW(groups, fused=fused)
    except TypeError:
        opt = torch.optim.AdamW(groups)
        fused = False
    if logger is not None and tag:
        logger.info(f"[optim] {tag} groups={len(groups)} fused={fused}（组内自带 lr/wd ✓）")
    return opt, fused


def build_muon(params, args, logger=None):
    """构造 **Muon** ✓（矩阵组 NS-正交化 ✓ 向量/embedding 走 AdamW ✓ 与旧代码一致 ✓）。

    旧写法:
        _muon_lr = float(getattr(args, "muon_lr", 0.02))
        _adamw_lr = float(getattr(args, "lr", 3e-4))
        _adamw_wd = float(getattr(args, "weight_decay", 0.01))
        opt = _Muon(params, lr=_muon_lr, weight_decay=0.0,
                    adamw_lr=_adamw_lr, adamw_weight_decay=_adamw_wd)
    ⚠ 导入失败时的报错文本也与旧实现逐字一致 ✓（便于按日志检索历史 ✓）
    """
    try:
        from src.optim.muon import Muon
    except Exception as e:                                            # noqa: BLE001
        raise RuntimeError(f"src.optim.muon import failed: {e}") from e
    muon_lr = float(getattr(args, "muon_lr", 0.02))
    adamw_lr = float(getattr(args, "lr", 3e-4))
    adamw_wd = float(getattr(args, "weight_decay", 0.01))
    opt = Muon(params, lr=muon_lr, weight_decay=0.0,
               adamw_lr=adamw_lr, adamw_weight_decay=adamw_wd)
    if logger is not None:
        logger.info(f"[optim] Muon: 矩阵组 lr={muon_lr} (NS正交) / "
                    f"向量+embedding组 AdamW lr={adamw_lr} wd={adamw_wd}")
    return opt


def identity(args) -> dict:
    """优化器**身份** ✓（供 manifest / 日志 ✓ 审查 P1-7 要求"记录用了什么" ✓）。

    ⚠ 只记录**配置**（名字 / lr / wd / 各分组 lr ✓），不记录运行期状态 ✗
    """
    name = resolve_name(args)
    out = {
        "name": name,
        "lr": float(getattr(args, "lr", 0.0) or 0.0),
        "weight_decay": float(getattr(args, "weight_decay", 0.0) or 0.0),
        "fused_supported": fused_supported(),
        "train_vae_encoder": bool(getattr(args, "train_vae_encoder", False)),
    }
    if name == "muon":
        out["muon_lr"] = float(getattr(args, "muon_lr", 0.02))
    for k in ("vae_lr", "callig_table_lr", "triple_table_lr_scale",
              "deform_lr_scale", "deform_trainable"):
        if hasattr(args, k):
            out[k] = getattr(args, k)
    return out


def describe(args) -> str:
    """一行身份摘要 ✓（启动日志用 ✓）。"""
    d = identity(args)
    bits = [f"name={d['name']}", f"lr={d['lr']:g}", f"wd={d['weight_decay']:g}",
            f"fused={'yes' if d['fused_supported'] else 'no'}"]
    if d.get("train_vae_encoder"):
        bits.append(f"vae_lr={float(d.get('vae_lr', 2e-6)):g}")
    if d["name"] == "muon":
        bits.append(f"muon_lr={d.get('muon_lr', 0.02):g}")
    return " ".join(bits)
