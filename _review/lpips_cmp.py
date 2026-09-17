"""用 LPIPS 直接比较 v11(M/2) 与 v12(S/2) 在 step 20000 的 strict 集。

这正是 ssim 答不了的问题: 两者 ssim 几乎相同 (0.5125 vs 0.5067),
但肉眼上 v12 笔画更粘连。LPIPS 对结构细节敏感, 应该能分辨。
"""
import os, sys, glob, re, statistics, math
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/root/Workspace/xy/DiT")
from src.eval.in_mem_eval import _lpips_per_sample, _get_lpips

ROOT = "assets/results"
RUNS = {
    "v11_M2": f"{ROOT}/v11_pretrain_M432_adaln4_fame_kxl_tj_px60/eval_samples_ctrl/step0020000/strict",
    "v12_S2": f"{ROOT}/v12_pretrain_S_cat_fame_kxl_tj_px60/eval_samples_ctrl/step0020000/strict",
}

fn = _get_lpips()
print("lpips fn:", "loaded" if fn is not None else "UNAVAILABLE")
if fn is None:
    sys.exit(1)


def load(run_dir):
    # 注意: glob("g*.png") 也会匹配 gt*.png -> 必须用 fullmatch 过滤
    idx = []
    for p in glob.glob(os.path.join(run_dir, "g*.png")):
        m = re.fullmatch(r"g(\d+)\.png", os.path.basename(p))
        if m:
            idx.append(int(m.group(1)))
    idx = sorted(idx)
    pred, gt = [], []
    for i in idx:
        p = Image.open(os.path.join(run_dir, f"g{i}.png")).convert("RGB")
        q = Image.open(os.path.join(run_dir, f"gt{i}.png")).convert("RGB")
        pred.append(np.asarray(p, dtype=np.float32) / 255.0)
        gt.append(np.asarray(q, dtype=np.float32) / 255.0)
    return idx, np.stack(pred), np.stack(gt)


res = {}
for name, d in RUNS.items():
    if not os.path.isdir(d):
        print(f"{name}: dir missing {d}")
        continue
    idx, pred, gt = load(d)
    lp = _lpips_per_sample(pred, gt, enabled=True)
    if lp is None:
        print(f"{name}: lpips failed")
        continue
    # ssim 对照 (用同一实现)
    from src.eval.inference import _ssim, _mse
    ss = [_ssim(pred[i], gt[i]) for i in range(len(idx))]
    res[name] = (idx, lp, ss)
    _mse_mean = statistics.mean([_mse(pred[i], gt[i]) for i in range(len(idx))])
    print(f"{name}: n={len(idx)}  lpips={statistics.mean(lp):.5f}  "
          f"ssim={statistics.mean(ss):.4f}  mse={_mse_mean:.5f}")

# 配对比较 (同一批样本)
if len(res) == 2:
    a, b = list(res.keys())
    ia, la, sa = res[a]
    ib, lb, sb = res[b]
    common = sorted(set(ia) & set(ib))
    da = {i: la[ia.index(i)] for i in common}
    db = {i: lb[ib.index(i)] for i in common}
    d = [db[i] - da[i] for i in common]
    n = len(d); m = statistics.mean(d); se = statistics.stdev(d) / math.sqrt(n)
    print(f"\n=== 配对比较 ({b} - {a}), n={n} ===")
    print(f"  LPIPS 差 = {m:+.5f}  se={se:.5f}  t={m/se:+.2f}  "
          f"({'显著' if abs(m/se) > 2 else '不显著'})")
    print(f"  -> {'v12 更差 (LPIPS 更低=更差? 注意方向)' if False else ''}")
    print(f"  解读: LPIPS 越低越好。差值为正 = {b} 的 LPIPS 更高 = {b} 更差")
    print(f"  {b} 更差的样本数: {sum(1 for x in d if x > 0)}/{n}")
