"""同一个 ckpt, 扫 CFG 与采样步数 —— 用来分离"瘦/碎"是**采样锐度**问题还是**损失**问题。

为什么先做这个: 改损失(加 L1/图像域 loss)要重训几小时且不可回滚; 而 cfg 是免费旋钮。
如果 frag 随 cfg 大幅下降 -> 模型其实会写粗笔画, 只是被低 cfg 抹细了 -> 先别动损失。
如果 cfg 拉高几乎不变 -> 确认是"学的时候就被均值回归抹平" -> 再上 L1 / 图像域 loss。

用法: python tools/cfg_sweep.py --ckpt <path> --cfgs 0.7,1.0,1.3,1.6 --steps 50,100
"""
import argparse
import os
import sys
import time

import torch as th

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="exp-std/runs_purestd/20261003-163412-v46-std-adaln4-top10-p1.0"
                                   "/checkpoints/0025000.pt")
ap.add_argument("--cfgs", default="0.7,1.0,1.3,1.6")
ap.add_argument("--steps", default="50")
ap.add_argument("--set", default="eval200fix")
ap.add_argument("--csv", default="exp-std/csv/eval200_fixed.csv")
ap.add_argument("--n", type=int, default=187)
ap.add_argument("--shards", default="exp-std/data/shards_std_w7_fixed_eval200")
ap.add_argument("--out-dir", default="exp-std/reeval/cfgsweep")
# ★ 训练在跑时同卡共存: 必须把 batch 压小(实测 batch16 吃 3.4G 会 OOM, 训练占 20G)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--vae-batch", type=int, default=4)
a = ap.parse_args()

import src.eval.in_mem_eval as IME                     # noqa: E402
from src.eval.model_io import load_model_from_ckpt     # noqa: E402

dev = th.device("cuda" if th.cuda.is_available() else "cpu")
cfgs = [float(x) for x in a.cfgs.split(",") if x.strip()]
stepss = [int(x) for x in a.steps.split(",") if x.strip()]

print(f"[ckpt] {a.ckpt}")
model, args = load_model_from_ckpt(a.ckpt, device=dev, use_ema=True, verbose=False)
args.eval_skel_latent_shards_dir = a.shards
args.skel_latent_shards_dir = "exp-std/data/shards_std_w7"
args.in_mem_eval_samechar_nn = False        # 扫描不需要, 省时间
args.in_mem_eval_save_samples = False
args.in_mem_eval_lpips = True
args.in_mem_eval_batch = a.batch            # 与训练共存时压小
args.in_mem_eval_vae_batch = a.vae_batch

rows = []
for ns in stepss:
    for cfg in cfgs:
        args.eval_cfg = cfg
        args.eval_steps = ns
        IME._DIFF = None                    # ★ 步数变了必须重建 ODE solver(模块级全局)
        tag = f"s{ns}_cfg{cfg}"
        d = os.path.join(a.out_dir, tag)
        t0 = time.time()
        res = IME.run_in_mem_eval(model, args, 25000, dev, d,
                                  sets=[(a.set, a.csv, a.n)], logger=lambda *x: None)
        m = res.get(a.set, {})
        rows.append((ns, cfg, m))
        th.cuda.empty_cache()
        print(f"[{tag}] ssim={m.get('ssim', float('nan')):.4f} "
              f"lpips={m.get('lpips', float('nan')):.4f} "
              f"ink_ssim={m.get('ink_ssim', float('nan')):.4f} "
              f"ink_iou={m.get('ink_iou', float('nan')):.4f} "
              f"frag={m.get('frag', float('nan')):.3f} "
              f"hole={m.get('hole', float('nan')):.3f}  ({time.time() - t0:.0f}s)")

print("\n================ 扫描汇总 (n=%d, ckpt step 25000) ================" % a.n)
print(f"{'steps':>6} {'cfg':>5} {'ssim':>8} {'lpips':>8} {'ink_ssim':>9} "
      f"{'ink_iou':>8} {'frag':>7} {'hole':>7}")
for ns, cfg, m in rows:
    print(f"{ns:>6} {cfg:>5.2f} {m.get('ssim', 0):>8.4f} {m.get('lpips', 0):>8.4f} "
          f"{m.get('ink_ssim', 0):>9.4f} {m.get('ink_iou', 0):>8.4f} "
          f"{m.get('frag', 0):>7.3f} {m.get('hole', 0):>7.3f}")
print("\n判读: frag 随 cfg 明显下降 -> 采样锐度问题(cfg 太低), 先别改损失;"
      "\n      frag 基本不动   -> 生成被均值回归抹平, 该上图像域 loss / L1")
