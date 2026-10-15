"""Parameter-gradient calibration, independent of writer/brush objective semantics.

★ 同步纪律（`docs/infra_1015.md` §二.2 / §二.3 ✓ 2026-10-16）:
  审查原文: "`messages` 每个 step 都构建，并不只发生在日志周期" ✗
    "应该将数值记录改为 `metrics["writer_loss"] = loss.detach()`，
     只在真正写日志时统一进行一次 CPU 读取" ✓
    "正常训练热路径中禁止 `.item()`、`float(cuda_tensor)`、`.tolist()`、
     `bool(cuda_tensor)`；只有**显式标记的同步边界**允许使用" ✓

  ⇒ 本模块的**两条路径**（已显式分离 ✓）:
    · **热路径**（每步 ✓ `reference_norm is None` ✓）: **零同步** ✓
      只做 GPU 张量运算 ✓ 不建字符串 ✗ 不读标量 ✗
    · **诊断路径**（probe 步 ✓ 低频 ✓ `reference_norm is not None` ✓）:
      **允许**同步 ✓ —— 每处都带 `# sync-ok:` 显式标记 ✓（§二.3 的规则 ✓）

  ★ 实测收益（改前 ✓）: `balance()` 每步对**每个 part** 做一次 `%.6g` 格式化 ✗
    ⇒ N 次 CUDA→CPU 同步/步 ✗（`strong_style` 有 writer/cf/brush 三路 ⇒ 3 次/步 ✗）
    而 `strong_style.py:226` 只在 `probe or step % 50 == 0` 才消费该字符串 ✓
    ⇒ 49/50 次的同步**纯属浪费** ✗✗
  ⇒ 改后: 热路径 **0 次** ✓；格式化推迟到 `format_metrics()` ✓（仅日志周期调用 ✓）
"""
from dataclasses import dataclass
from typing import Any

import torch


def norm(loss, parameters):
    """Measure one objective without accumulating .grad or releasing its graph."""
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    values = [g.detach().float().square().sum() for g in grads if g is not None]
    if not values:
        return loss.new_zeros(())
    return torch.stack(values).sum().sqrt()


@dataclass
class PartMetrics:
    """一个 Loss 部件的**待记录指标** ✓ —— 全部保持为 **GPU 张量** ✓（不触发同步 ✓）。

    ★ 只在 `format_metrics()` 里才转 CPU ✓（`docs/infra_1015.md` §二.2 ✓）。
    """

    loss: torch.Tensor          # GPU ✓
    weight: torch.Tensor        # GPU ✓（`scale` 本身可能是 0-dim 张量 ✓）
    ratio: float                # Python 常数 ✓（来自 args ✓ 不触发同步 ✓）
    actual: float | None = None  # 仅 probe 步有 ✓（诊断路径 ✓ 允许同步 ✓）


@dataclass
class GradientBudget:
    max_weight: float
    scales: dict[str, float]
    weak_probes: dict[str, int]
    logger: Any

    def balance(self, parts, parameters, reference_norm, step):
        """Return weighted objectives and **GPU-resident** per-part metrics.

        ★ 返回 `(total, dict[str, PartMetrics])` ✓ —— **不再**返回预格式化字符串 ✗
          （旧实现每步对每个 part 做 `%.6g` ⇒ N 次 CUDA→CPU 同步 ✗✗）
        ★ 调用方在**日志周期**用 `format_metrics()` 一次性转 CPU ✓
        ★ 语义不变 ✓: 加权方式、`weak` 判定、连续 10 次报错**全部逐字保留** ✓
        """
        first_loss = next(iter(parts.values()))[0]
        total = first_loss.new_zeros(())
        metrics: dict[str, PartMetrics] = {}
        for name, (loss, ratio) in parts.items():
            actual = None
            if reference_norm is not None:
                # ── ★ 诊断路径（probe 步 ✓ 低频 ✓）—— 允许同步 ✓（§二.3 ✓）──
                auxiliary_norm = norm(loss, parameters)
                # sync-ok: probe 步的低频诊断 ✓ 需要标量以决定 scale ✓（§二.2 明确允许 ✓）
                scale = min(self.max_weight, float(ratio * reference_norm / auxiliary_norm.clamp_min(1e-8))) if auxiliary_norm > 1e-8 else 0.
                self.scales[name] = scale
                # sync-ok: 同上 ✓ 低频诊断 ✓
                actual = float(scale * auxiliary_norm / reference_norm.clamp_min(1e-8))
                # sync-ok: 同上 ✓ 低频诊断 ✓（bool(cuda_tensor) ✓）
                weak = ratio > 0 and loss.detach() > 1e-3 and actual < .8 * ratio
                self.weak_probes[name] = self.weak_probes.get(name, 0) + 1 if weak else 0
                if weak:
                    self.logger.info('[strong-style-underpowered] step=%d part=%s actual=%.4f target=%.4f consecutive=%d',
                                     step, name, actual, ratio, self.weak_probes[name])
                if self.weak_probes[name] >= 10:
                    raise RuntimeError('strong-style remains underpowered for 10 probes: ' + name)
            scale = self.scales[name]
            total = total + scale * loss
            # ★ 热路径: **只存张量** ✓ 零同步 ✓（旧实现此处 `'%s=%.6g' % (name, loss.detach())` ✗）
            metrics[name] = PartMetrics(loss=loss.detach(), weight=scale,
                                        ratio=ratio, actual=actual)
        return total, metrics


def format_metrics(metrics: dict[str, PartMetrics]) -> str:
    """把 `balance()` 的 GPU 指标转成日志字符串 ✓ —— **只在日志周期调用** ✓。

    ★ 输出格式与旧实现**逐字相同** ✓（`'%s=%.6g weight=%.6g ratio=%s'` ✓）
      ⇒ 既有日志与解析脚本**不受影响** ✓（不为了省同步而改日志格式 ✗）
    ★ 这里是**唯一**一次 CPU 读取 ✓（§二.2 ✓）
    """
    out = []
    for name, m in metrics.items():
        # sync-ok: 日志周期唯一一次读取 ✓（§二.2 明确要求"统一进行一次 CPU 读取" ✓）
        ratio = 'unprobed' if m.actual is None else '%.4f' % m.actual
        out.append('%s=%.6g weight=%.6g ratio=%s'
                   % (name, float(m.loss), float(m.weight), ratio))
    return ' | '.join(out)
