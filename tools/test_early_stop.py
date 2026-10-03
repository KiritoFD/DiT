"""早停逻辑单测 (纯 CPU, 不碰 GPU):
  1) iou_lpips 组合 (skel_iou↑ + lpips↓), 双指标全 stale 才停
  2) json 主数据源 (eval_auto_<step>.json)
  3) CSV 回退数据源 (eval_stdskel_summary.csv)
  4) ★ 阶段内早停: 上一段的好成绩不得当基线 (显式 --early-stop-from-step 与自动定界两条路)
"""
import csv
import json
import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, "/root/Workspace/xy/DiT")
from src.train.early_stop import EarlyStopper  # noqa: E402


class L:
    def info(self, m):
        print("     ", m)

    def error(self, m):
        print("      !", m)


def mk(tmp, patience=1, **kw):
    ns = SimpleNamespace(
        early_stop=True, early_stop_metric="iou_lpips", early_stop_patience=patience,
        early_stop_min_delta=0.002, early_stop_min_delta_iou=0.005,
        early_stop_min_delta_lpips=0.003, early_stop_min_delta_mse=0.0,
        early_stop_set="eval200", ckpt_every=2500, early_stop_check_every=2500,
        early_stop_from_step=0,
        results_dir=os.path.join(tmp, "run"),
        checkpoint_dir=os.path.join(tmp, "run", "checkpoints"))
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def w_json(ck, step, iou, lp, ssim=0.60):
    with open(os.path.join(ck, f"eval_auto_{step}.json"), "w", encoding="utf-8") as f:
        json.dump({"step": step, "set": "eval200", "n": 200, "ssim": ssim,
                   "mse": 1.0, "skel_iou": iou, "lpips": lp}, f)


# ── 1/2: json 主源 + 组合判据 ────────────────────────────────────────────
d = tempfile.mkdtemp()
ck = os.path.join(d, "run", "checkpoints")
os.makedirs(ck)
st = EarlyStopper(mk(d), ck, L())
print("[1] 无任何数据 -> 应停=False :", st.check(force=True))
assert st.check(force=True) is False

print("[2] json 主源: 逐步变好 -> False ; 双指标同时退化 -> True")
for step, iou, lp in ((2500, 0.10, 0.50), (5000, 0.12, 0.45), (7500, 0.10, 0.52)):
    w_json(ck, step, iou, lp)
    r = st.check()
    print(f"     step {step} iou={iou} lpips={lp} -> stop={r}")
    if step == 7500:
        assert r is True, "双指标退化后应停"

for f in os.listdir(ck):
    os.remove(os.path.join(ck, f))

# ── 3: CSV 回退源 ────────────────────────────────────────────────────────
# 显式 from_step=1 (等同运行器给第 1 段传的值) -> 关闭自动定界, 直接检验 CSV 读取
print("[3] CSV 回退源 (显式 from_step=1)")
with open(os.path.join(d, "run", "eval_stdskel_summary.csv"), "w", newline="",
          encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["exp", "step", "set", "n", "ssim_mean", "lpips_mean", "skel_iou_mean"])
    w.writerow(["v46", 12500, "eval200", 200, 0.6300, 0.57000, 0.09500])
st2 = EarlyStopper(mk(d, early_stop_from_step=1), ck, L())
r1 = st2.check(force=True)
print(f"     step12500 -> stop={r1} (应 False: 首次即 best)")
assert r1 is False
with open(os.path.join(d, "run", "eval_stdskel_summary.csv"), "a", newline="",
          encoding="utf-8") as f:
    csv.writer(f).writerow(["v46", 15000, "eval200", 200, 0.6300, 0.63000, 0.09000])
r2 = st2.check()
print(f"     step15000 (双指标退化) -> stop={r2} (应 True)")
assert r2 is True

# ── 4: 阶段内早停 ────────────────────────────────────────────────────────
print("[4] 阶段内早停: 上一段的好成绩(iou 0.20/lpips 0.30)不得当基线")
for tag, extra in (("显式 from_step", {"early_stop_from_step": 30001}), ("自动定界", {})):
    d2 = tempfile.mkdtemp()
    ck2 = os.path.join(d2, "run", "checkpoints")
    os.makedirs(ck2)
    # results_dir 里先有上一段的评测 (step 30000, 指标好得多)
    with open(os.path.join(d2, "run", "eval_stdskel_summary.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["exp", "step", "set", "n", "ssim_mean", "lpips_mean",
                    "skel_iou_mean"])
        w.writerow(["v46", 30000, "eval200", 200, 0.66, 0.30000, 0.20000])
    s = EarlyStopper(mk(d2, patience=2, **extra), ck2, L())
    # 本段三条: 全都不如上一段 (iou 0.05 / lpips 0.50), 且彼此不退步
    res = []
    for step in (32500, 35000, 37500):
        w_json(ck2, step, 0.05, 0.50)
        res.append(s.check())
    print(f"     [{tag}] 32500/35000/37500 -> {res} (应 [False, False, True])")
    assert res == [False, False, True], res

print("\n全部通过 ✅  (iou_lpips 组合 / json 主源 / CSV 回退源 / 阶段内早停)")
