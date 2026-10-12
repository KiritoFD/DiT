# -*- coding: utf-8 -*-
"""vae_scaling.py — VAE 缩放/位移常数的**唯一来源**（收口 460 处硬编码 0.18215）

背景（E5 事故）:
  SD-VAE 的 ``sf=0.18215`` 被抄进 460 处硬编码。当换成别的 VAE（physics / Calli-VAE /
  FLUX 16ch）时，这些 0.18215 不会自动跟着变 -> **静默按错的缩放系数编码/解码**，
  而且指标照样算得出来（实测潜伏 3 周）。

约定（见 docs/vae_latent_contract.md §1.1）:
  ``z_stored = (z_raw - shift) * sf``；还原 ``z_raw = z_stored / sf + shift``。
  => 编码时 ``z_stored = (z_raw - shift) * sf``，解码时 ``decode(z_stored / sf + shift)``。

用法:
  from src.utils.vae_scaling import resolve_vae_scaling
  sf, shift = resolve_vae_scaling(args)            # 显式配置优先，否则按路径查表
  sf, shift = resolve_vae_scaling(vae_path="data/pretrained/sd-vae-ft-ema")

行为保证（迁移安全）:
  * ``args.vae_scaling_factor`` / ``args.vae_shift_factor`` 一旦显式给出 -> **原样使用**，
    与本模块引入前**逐位一致** ✓（不会改变任何既有实验的数值）。
  * 未显式给出时按 VAE 路径查表；查不到则回落到 SD 值并**大声警告**（不静默）。
"""
from __future__ import annotations

# SD-VAE (stabilityai/sd-vae-ft-ema / -ft-mse) 的标准值
SD_SF = 0.18215
SD_SHIFT = 0.0

# 路径关键字 -> (sf, shift)。**新增 VAE 必须在这里登记**，否则会走 SD 回落并告警。
_VAE_TABLE = (
    ("sd-vae-ft-ema", (SD_SF, SD_SHIFT)),
    ("sd-vae-ft-mse", (SD_SF, SD_SHIFT)),
    ("sd-vae", (SD_SF, SD_SHIFT)),
    ("stabilityai/sd-vae", (SD_SF, SD_SHIFT)),
    # ★ 2026-10-12: Calli "physics" VAE —— 取值直接来自 v80_callivae_physics_noOT /
    #   v82_physics_noOT_stdsoft 两个**已跑过**的配置（不是我推算的 ✗）。
    #   ⚠ 它的 shift 非零、sf 与 SD 差 4.6× ⇒ 若某处仍按 0.18215 编解码，图必错 ✗
    ("calli_vae_physics", (0.843976921140543, -0.0334180024275045)),
    ("calli_vae_final", (0.843976921140543, -0.0334180024275045)),
)

_WARNED = set()


def lookup_by_path(vae_path):
    """按路径查 (sf, shift)。查不到返回 None。"""
    p = str(vae_path or "").replace("\\", "/").lower()
    for key, val in _VAE_TABLE:
        if key in p:
            return val
    return None


def resolve_vae_scaling(args=None, vae_path=None, verbose=True):
    """返回 ``(sf, shift)``。

    优先级: 显式 args.vae_scaling_factor/shift_factor  >  按 vae_path 查表  >  **报错** ✓（2026-10-13 [审查 P0]: 不再静默回退 SD ✗）
    """
    sf = getattr(args, "vae_scaling_factor", None) if args is not None else None
    sh = getattr(args, "vae_shift_factor", None) if args is not None else None
    if sf is not None:
        return float(sf), float(sh or 0.0)

    path = vae_path if vae_path is not None else getattr(args, "vae_path", "")
    hit = lookup_by_path(path)
    if hit is not None:
        return hit

    # ★ 2026-10-13 [审查 P0]: **未登记的 VAE 必须报错, 不许回退 SD** ✗✗
    #   回退 sf=0.18215 会让"换了 VAE 但缩放没跟着换"变成**静默错误 latent 契约** ——
    #   本项目已为此付出过代价 ✗。现在 fail loudly, 让人显式登记或显式给值 ✓
    raise ValueError(
        "[vae_scaling] 未登记的 VAE 路径: %r\n"
        "  拒绝回退 SD 值(sf=%.5f) —— 那会让 latent 契约静默错误 ✗\n"
        "  修法(任选其一):\n"
        "    1) 在 src/utils/vae_scaling.py 的 _VAE_TABLE 登记该 VAE(附 sf/shift + 出处 ✓);\n"
        "    2) 在 config/args 里**显式**给 vae_scaling_factor / vae_shift_factor ✓;\n"
        "    3) 从 checkpoint 的 args 读(评测端应优先用 ckpt 自带值 ✓)。"
        % (str(path or "<unknown>"), SD_SF))
