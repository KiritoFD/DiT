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
import csv
import glob
import json
import os

# 组合 metric 的规格: (子指标, 是否越大越好)
# 语义: 各子指标**各自**追踪 best/stale, **全部** stale 才停 (双保险)。
_ES_SPECS = {
    "combo": (("ssim", True), ("skel_iou", True)),
    "ssim_lpips": (("ssim", True), ("lpips", False)),
    # ★ iou_lpips: 结构 (骨架 IoU) + 感知 (LPIPS) 互补, 不比 ssim ——
    #   ssim 在 256² 二值字形上对笔画粗细/亚像素位移极敏感(见文件头说明),
    #   且全图 ssim 被 ~90% 白底抬高 (实测 0.5745 vs 墨迹框 0.3161)。
    "iou_lpips": (("skel_iou", True), ("lpips", False)),
}

# 需要"越大越好"的 metric
_HIGHER_BETTER = ("ssim", "combo", "ssim_lpips", "iou_lpips")


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
        # ★ 阶段内早停 (2026-10-03): 采样课程的**阶段边界会故意让指标变差**
        #   (GT 采样比例下降 -> 部署条件下的分数先掉一截), 所以绝不能拿上一阶段的
        #   最好成绩当基线。两种定界方式:
        #     --early-stop-from-step N > 0 : 显式给本段起点(级联运行器用这个, 最清晰)
        #     否则自动: 首次 check 时把"当前可见的最新评测 step + 1"当作本段起点
        #            (这样改 early_stop.py 后, 已在跑的运行器下一段就自动生效)
        self.from_step = int(getattr(self.args, "early_stop_from_step", 0) or 0)
        self._auto_scope = self.from_step <= 0
        if self._auto_scope:
            # ★ 定界必须在**构造时**做 (而不是首次 check 时): 此刻数据源里只可能有
            #   上一阶段的评测, 本段自己的评测还没产生 -> 定界确定, 且不会把本段
            #   第一条评测误当成"上一段末尾"丢掉 (首次-check 定界会丢, 实测)。
            _now = self._max_step_seen()
            self._scope_step = (_now + 1) if _now is not None else 1
        else:
            self._scope_step = self.from_step

        logger.info(f"[early-stop] metric={self.metric}, patience={self.patience}, "
                    f"min_delta={self.delta}, 阶段内早停: 只看 step >= {self._scope_step}"
                    + (" (自动定界)" if self._auto_scope else " (--early-stop-from-step)"))

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
        if latest is not None:
            ev_step, last_ev = latest
            val = self._read_metric(last_ev)
        else:
            # ★ 回退数据源: 现代在训评测 (--in-mem-eval) 只写
            #   <run_dir>/eval_stdskel_summary.csv, **不写** eval_auto_<step>.json
            #   (后者是 legacy 外部 eval 进程的产物) -> 只认 json 时早停永远不触发,
            #   而且静默(不报错、只是永远不停)。实测: 只挂 in-mem-eval 的 run 目录里
            #   eval_auto_*.json 数量为 0。
            got = self._latest_summary()
            if got is None:
                return False
            ev_step, val = got
        if val is None:
            return False
        # ★ 阶段内早停: 只认本段起点(含)之后的评测 —— 上一段的最好成绩不作基线,
        #   因为课程的阶段边界会**故意**让指标变差(GT 采样比例下降)。
        if self._scope_step and ev_step < self._scope_step:
            return False
        if not force and ev_step <= self.last_eval_step:
            return False
        self.last_eval_step = ev_step

        if self.spec is not None:
            return self._check_combo(ev_step, val)
        return self._check_single(ev_step, val)

    def _max_step_seen(self):
        """两个数据源里当前可见的最大评测 step (供自动定界用)。"""
        steps = []
        j = self._latest_eval_file()
        if j is not None:
            steps.append(j[0])
        s = self._latest_summary()
        if s is not None:
            steps.append(s[0])
        return max(steps) if steps else None

    # ── 回退数据源: in_mem_eval 的摘要 CSV ──────────────────────────────────
    def _latest_summary(self):
        """从 <run_dir>/eval_stdskel_summary.csv 取最近一次在训评测。

        列 (22): exp, step, set, n, ssim_mean..p90, mse_mean, lpips_mean,
                 ink_ssim_mean, ink_iou_mean, skel_iou_mean, frag_ratio,
                 hole_pred, hole_gt, nn_ssim, nn_mean, tgt_spec, cal_enrich
        set 选取: --early-stop-set 优先; 未给则取 **n 最大** 的 set (样本最多最可信)。
        返回 (step, val), val 的形状与 `_read_metric` 一致 (组合 metric 统一"越大越好",
        故 lpips 取负号)。
        """
        # 摘要落盘位置 = args.results_dir (in_mem_eval 就是这么调的);
        # 兜底再看 ckpt 目录的上一级 (某些调用路径传的是 run 子目录)。
        cands = [os.path.join(str(getattr(self.args, "results_dir", "") or ""),
                              "eval_stdskel_summary.csv"),
                 os.path.join(os.path.dirname(os.path.abspath(self.checkpoint_dir)),
                              "eval_stdskel_summary.csv")]
        p = next((c for c in cands if c and os.path.exists(c)), None)
        if p is None:
            return None
        rows = []
        try:
            with open(p, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    try:
                        rows.append((int(r["step"]), r))
                    except (KeyError, TypeError, ValueError):
                        continue
        except Exception:                                     # noqa: BLE001
            return None
        if not rows:
            return None
        want = str(getattr(self.args, "early_stop_set", "") or "")
        if not want:
            by_set = {}
            for _, r in rows:
                s = str(r.get("set", ""))
                try:
                    n = float(r.get("n") or 0)
                except (TypeError, ValueError):
                    n = 0.0
                if s not in by_set or n > by_set[s]:
                    by_set[s] = n
            want = max(by_set, key=lambda k: by_set[k]) if by_set else ""
        rows = [x for x in rows if not want or str(x[1].get("set", "")) == want]
        if not rows:
            return None
        ev_step, r = max(rows, key=lambda x: x[0])

        def _g(key):
            try:
                return float(r.get(key, ""))
            except (TypeError, ValueError):
                return None

        vals = {"ssim": _g("ssim_mean"), "mse": _g("mse_mean"),
                "lpips": _g("lpips_mean"), "skel_iou": _g("skel_iou_mean")}
        if self.spec is not None:
            out = []
            for k, hi in self.spec:
                v = vals.get(k)
                if v is None:
                    return None
                out.append(v if hi else -v)
            return ev_step, tuple(out)
        v = vals.get(self.metric)
        if v is None:
            return None
        return ev_step, v

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
            # ★ 按 spec 通用取数 (原来只硬编码了 ssim_lpips / combo,
            #   新增 iou_lpips 会落到 mse 分支 -> 返回 float -> _check_combo 崩)。
            _raw = {"ssim": d.get("ssim"), "skel_iou": d.get("skel_iou"),
                    "lpips": d.get("lpips"), "mse": d.get("mse")}
            if self.spec is not None:
                out = []
                for key, hi in self.spec:
                    v = _raw.get(key)
                    if v is None:
                        return None
                    out.append(float(v) if hi else -float(v))
                return tuple(out)
            v = _raw.get(self.metric)
            return None if v is None else float(v)
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
