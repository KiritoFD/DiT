# -*- coding: utf-8 -*-
"""早停 —— 从 train.py 抽出的独立模块（2026-09-17）。

判据来源：ckpt 目录下由评测写出的 ``eval_auto_<step>.json``（字段 mse/ssim/skel_iou/lpips）。
每次 ``check()`` 只处理**新的** eval 文件（按 step 去重），避免同一份结果被重复计入 stale。

支持四种 metric：
  ``ssim`` / ``mse``    单指标
  ``combo``             ssim↑ + skel_iou↑  （skel_iou 已被证实不敏感，不推荐）
  ``ssim_lpips``        ssim↑ + lpips↓     （推荐：像素结构 + 感知距离互补）

多指标组合是**双保险**语义：各自追踪 best/stale，**全部** stale 才停。
用 ``(key, higher_better)`` 描述方向，因此天然支持方向混合。

⚠ min_delta 不可省：ssim 在 256² 二值字形上对笔画粗细/亚像素位移极敏感，
不设阈值的话指标噪声会不停重置 stale 计数器，早停实际由噪声驱动。
"""
import glob
import json
import os

# 组合 metric 的规格: (子指标, 是否越大越好)
_ES_SPECS = {
    "combo": (("ssim", True), ("skel_iou", True)),
    "ssim_lpips": (("ssim", True), ("lpips", False)),
}

# 需要"越大越好"的 metric
_HIGHER_BETTER = ("ssim", "combo", "ssim_lpips")


def _min_deltas(args):
    """各指标的 min_delta 默认值（按经验噪声量级选取）。

    ssim / skel_iou / lpips 都是 ~0~1 量级；mse 的量级随数据分布变化，
    默认不设阈值，需要时显式配置。
    """
    return {
        "ssim": float(getattr(args, "early_stop_min_delta", 0.002)),
        "skel_iou": float(getattr(args, "early_stop_min_delta_iou", 0.005)),
        "lpips": float(getattr(args, "early_stop_min_delta_lpips", 0.003)),
        "mse": float(getattr(args, "early_stop_min_delta_mse", 0.0)),
    }


class EarlyStopper:
    """早停状态机。每次 ``check()`` 返回 True 表示"该停了"。"""

    def __init__(self, args, checkpoint_dir, logger):
        self.args = args
        self.checkpoint_dir = checkpoint_dir
        self.logger = logger

        self.metric = getattr(args, "early_stop_metric", "ssim")
        self.patience = int(getattr(args, "early_stop_patience", 5))
        self.delta = _min_deltas(args)
        self.higher_better = self.metric in _HIGHER_BETTER

        self.spec = _ES_SPECS.get(self.metric)
        # 单指标状态
        self.best = None
        self.stale = 0
        # 组合指标状态
        self.combo_best = {k: None for k, _ in (self.spec or ())}
        self.combo_stale = {k: 0 for k, _ in (self.spec or ())}
        # 去重: 已计入过的最大 eval step
        self.last_eval_step = -1

        logger.info(f"[early-stop] metric={self.metric}, patience={self.patience}, "
                    f"min_delta={self.delta}")

    # ── 检查周期 ────────────────────────────────────────────────────────────
    @property
    def check_every(self):
        """多少步检查一次。未显式配置时取 ckpt 周期的一半（至少 1000）。"""
        n = int(getattr(self.args, "early_stop_check_every", 0))
        if n <= 0:
            n = max(int(getattr(self.args, "ckpt_every", 5000)) // 2, 1000)
        return n

    # ── 主入口 ──────────────────────────────────────────────────────────────
    def check(self, force=False):
        if not getattr(self.args, "early_stop", False):
            return False
        latest = self._latest_eval_file()
        if latest is None:
            return False
        ev_step, last_ev = latest
        if not force and ev_step <= self.last_eval_step:
            return False
        self.last_eval_step = ev_step

        val = self._read_metric(last_ev)
        if val is None:
            return False

        if self.spec is not None:
            return self._check_combo(ev_step, val)
        return self._check_single(ev_step, val)

    # ── 内部 ────────────────────────────────────────────────────────────────
    def _latest_eval_file(self):
        """取**步数最大**的 eval_auto_*.json -> (step, path)，没有则 None。

        ⚠ 必须按**数值**排序，不能 `sorted(glob(...))[-1]`：
        文件名是原始步数（`eval_auto_95000.json` / `eval_auto_100000.json`），
        字符串序下 `'9' > '1'` -> **95000 会排在 100000 之后**，
        于是跨过 10 万步之后早停会一直读**旧文件**，stale 计数与 best 全部失真。
        （原实现就是 `sorted(...)[-1]`，属于静默错误：不报错、只是判据永远滞后。）
        """
        best = None
        for f in glob.glob(os.path.join(self.checkpoint_dir, "eval_auto_*.json")):
            try:
                s = int(os.path.basename(f)
                        .replace("eval_auto_", "").replace(".json", ""))
            except ValueError:
                continue
            if best is None or s > best[0]:
                best = (s, f)
        return best

    def _read_metric(self, path):
        """读 eval json 并取出当前 metric 的数值（组合 metric 返回 tuple）。

        lpips 取**负号**统一成"越大越好"，从而复用同一套比较逻辑；
        打印时会还原（见 `_fmt`）。
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            s, k, lp = d.get("ssim"), d.get("skel_iou"), d.get("lpips")
            if self.metric == "ssim_lpips":
                return ((float(s), -float(lp))
                        if s is not None and lp is not None else None)
            if self.metric == "combo":
                return ((float(s), float(k))
                        if s is not None and k is not None else None)
            if self.metric == "ssim":
                return float(s) if s is not None else None
            m = d.get("mse")
            return float(m) if m is not None else None
        except Exception:                                     # noqa: BLE001
            # eval json 可能正被写入（读到半个文件）-> 当次跳过，不算失败
            return None

    @staticmethod
    def _fmt(key, v):
        """显示还原: lpips 在内部取了负号。"""
        return -v if key == "lpips" else v

    def _check_combo(self, ev_step, val):
        improved = False
        for (key, _hi), v in zip(self.spec, val):
            b = self.combo_best[key]
            dlt = self.delta.get(key, 0.0)
            if b is None or v > b + dlt:
                self.combo_best[key] = v
                self.combo_stale[key] = 0
                improved = True
            else:
                self.combo_stale[key] += 1

        shown = ", ".join(f"{k}={self._fmt(k, v):.4f}"
                          for (k, _), v in zip(self.spec, val))
        best_shown = ", ".join(f"best_{k}={self._fmt(k, self.combo_best[k]):.4f}"
                               for k, _ in self.spec)
        if improved:
            self.stale = 0
            self.logger.info(f"[early-stop] eval step {ev_step}: {self.metric} {shown} "
                             f"-> NEW BEST ({best_shown})")
            return False

        self.stale += 1
        self.logger.info(f"[early-stop] eval step {ev_step}: {self.metric} {shown} "
                         f"({best_shown}, stale {self.stale}/{self.patience})")
        if self.stale >= self.patience:
            self.logger.info(f"[early-stop] {self.metric} no improvement for "
                             f"{self.stale} evals; early stopping.")
            return True
        return False

    def _check_single(self, ev_step, val):
        d = self.delta.get(self.metric, 0.0)
        improved = (self.best is None
                    or ((val > self.best + d) if self.higher_better
                        else (val < self.best - d)))
        if improved:
            self.best = val
            self.stale = 0
            self.logger.info(f"[early-stop] eval step {ev_step}: "
                             f"{self.metric}={val:.4f} (new best)")
            return False

        self.stale += 1
        self.logger.info(f"[early-stop] eval step {ev_step}: {self.metric}={val:.4f} "
                         f"(best {self.best:.4f}, stale {self.stale}/{self.patience})")
        if self.stale >= self.patience:
            self.logger.info(f"[early-stop] {self.metric} no improvement for "
                             f"{self.stale} evals; early stopping.")
            return True
        return False
