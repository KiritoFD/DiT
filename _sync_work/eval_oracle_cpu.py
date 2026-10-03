"""CPU-only: 用指定骨架条件目录评 v46 ckpt 的 eval200 子集 (完全不碰 GPU)。

目的: 量"这个模型自己的天花板" —— 把评测条件从 std 标准骨架换成 GT 真迹骨架,
      看同一个 ckpt 在同一批 held-out 目标上能好多少。两者之差 = 条件的价值;
      天花板本身离 1.0 的距离 = 欠训/设计的余地。

与 in_mem_eval 的口径对齐: cfg=0.7, 50 步 heun, decode(/0.18215), 再算
ssim / ink_ssim / ink_iou / skel_iou / lpips / frag。
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch as th

ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--cond-dir", default="exp-std/data/shards_gtskel_w7")
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--cfg", type=float, default=0.7)
ap.add_argument("--cache", default="exp-std/data/eval_real200_cache.pt")
ap.add_argument("--vae", default="data/pretrained/pretrained_models/sd-vae-ft-ema")
ap.add_argument("--batch", type=int, default=200)   # 采样批大小 (GPU 上开大, 快)
ap.add_argument("--decode-batch", type=int, default=16)   # VAE 解码批 (256² 解码吃显存)
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--device", default="cuda")   # 卡空就用 cuda(快); 训练占卡时用 cpu
ap.add_argument("--tag", default="")
a = ap.parse_args()

dev = th.device(a.device) if (a.device != "cuda" or th.cuda.is_available()) \
    else th.device("cpu")
th.set_num_threads(a.threads or max(8, (os.cpu_count() or 8)))
t0 = time.time()

from src.eval import inference                       # noqa: E402
from src.eval.in_mem_eval import _get_vae, _lpips_per_sample  # noqa: E402
from src.eval.metrics import frag_ratio, hole_ratio, skel_iou, ssim_torch  # noqa: E402
from src.eval.metrics_ink import ink_iou, ink_ssim   # noqa: E402
from src.model.dit import DiT_2Cond_S_2              # noqa: E402

c = th.load(a.cache, map_location="cpu", weights_only=False)
ids = [int(i) for i in c["img_ids"]][:a.n]
conds = list(c["conds"])[:a.n]
noise = c["noise"][:a.n].to(dev)
std_lats = c["std_lats"][:a.n].to(dev)
gt_lats = c["gt_lats"][:a.n].to(dev)
gt_png = c["gt_pngs"][:a.n].float().to(dev)
std_png = c["std_pngs"][:a.n].float().to(dev)
print(f"[data] eval200 子集 n={len(ids)}", flush=True)

# ── 条件: 从 shards 按 img_id 取 ─────────────────────────────────────────
want = set(ids)
lut = {}
for f in sorted(glob.glob(os.path.join(a.cond_dir, "shard_*.npz"))):
    with np.load(f) as z:
        for j, i in enumerate(z["img_ids"]):
            ii = int(i)
            if ii in want:
                lut[ii] = np.asarray(z["latents"][j], np.float32)
miss = [i for i in ids if i not in lut]
print(f"[cond] {a.cond_dir}: 命中 {len(lut)}/{len(ids)}"
      + (f" 缺 {miss[:5]}" if miss else ""), flush=True)
assert not miss, "条件目录缺样本"
cond = th.from_numpy(np.stack([lut[i] for i in ids])).to(dev)

# ── 模型: 用仓库自己的 model_io (按 ckpt 里存的 args 复刻, 不手搭、不猜结构) ──
from src.eval import model_io                            # noqa: E402

m, margs = model_io.load_model_from_ckpt(a.ckpt, device=dev, use_ema=True)
m = m.eval()
print(f"[model] model={getattr(margs, 'model', None)} "
      f"inject={getattr(margs, 'glyph_inject_mode', None)}"
      f"×{getattr(margs, 'glyph_inject_layers', None)} "
      f"n_callig={getattr(margs, 'num_calligraphers', None)} "
      f"fusion={getattr(margs, 'condition_fusion', None)} "
      f"glyph_vec_cond={getattr(margs, 'glyph_vec_cond', None)} "
      f"use_char={not getattr(margs, 'no_char_cond', False)}", flush=True)

vae = _get_vae(dev, a.vae).eval()
diff = inference.build_diffusion(a.steps, "flow")


def run(cond_lat, tag):
    t = time.time()
    with th.no_grad():
        g = inference.sample_latents(m, diff, noise, conds, cfg_scale=a.cfg,
                                     batch=a.batch, device=dev, skel=cond_lat)
        g = g.to(dev)          # sample_latents 返回 CPU 张量 -> decode 前搬回设备
        dec = th.cat([((vae.decode(g[s:s + a.decode_batch] / 0.18215).sample
                        .clamp(-1, 1)) + 1) / 2
                      for s in range(0, len(g), a.decode_batch)], 0)
    pn, gn = dec.cpu().numpy(), gt_png.cpu().numpy()
    s = float(np.mean([ssim_torch(dec[i:i + 1], gt_png[i:i + 1]).item()
                       for i in range(len(dec))]))
    ins = float(np.mean([ink_ssim(pn[i], gn[i]) for i in range(len(dec))]))
    ini = float(np.mean([ink_iou(pn[i], gn[i]) for i in range(len(dec))]))
    sk = float(np.mean([skel_iou(pn[i], gn[i], thresh=0.5) for i in range(len(dec))]))
    fr = float(np.mean([frag_ratio(pn[i], gn[i], thresh=0.5) for i in range(len(dec))]))
    hp = float(np.mean([hole_ratio(pn[i], thresh=0.5) for i in range(len(dec))]))
    lp = _lpips_per_sample(pn.transpose(0, 2, 3, 1), gn.transpose(0, 2, 3, 1))
    lp = float(np.mean(lp)) if lp else float("nan")
    print(f"[{tag}] ssim={s:.4f} ink_ssim={ins:.4f} ink_iou={ini:.4f} "
          f"skel_iou={sk:.4f} frag={fr:.3f} hole={hp:.3f} lpips={lp:.4f} "
          f"({time.time() - t:.0f}s)", flush=True)
    return dict(ssim=s, ink_ssim=ins, ink_iou=ini, skel_iou=sk, frag=fr,
                hole=hp, lpips=lp)


r = {}
r["cond_GT"] = run(cond, "GT 骨架条件 (oracle)")
r["cond_std"] = run(std_lats, "std 骨架条件 (部署)")
r["baseline_std_png"] = None
print(f"[ref] 什么都不做(直接拿 std 图当输出): "
      f"ssim={float(np.mean([ssim_torch(std_png[i:i+1], gt_png[i:i+1]).item() for i in range(len(ids))])):.4f}",
      flush=True)
print(f"=== {a.tag} n={len(ids)} 用时 {time.time()-t0:.0f}s ===")
for k, v in r.items():
    if v:
        print(f"  {k:>16}: ssim={v['ssim']:.4f} skel_iou={v['skel_iou']:.4f} "
              f"lpips={v['lpips']:.4f} ink_iou={v['ink_iou']:.4f}")
