"""探针二: 单 batch 极限记忆 (Micro-Training, ~2 分钟/臂)。

问的是"注入头的**特征容量**够不够" —— 不跑扩散, 只当一个自编码器:
  ① 造一个 hard batch: **同一字 × 尽可能多的书家** (风格跨度最大)
  ② 冻结 DiT 主干, 只解冻"注入链"; t≈0(直接预测清晰图)
  ③ 就这一批死磕 500 步 Adam
裁决: loss 断崖下跌 -> 容量够 ✓; 早早平底 -> 注入头是瓶颈, 跑 150k 步也是浪费电 ✗

用法: python tools/injection_overfit_probe.py --ckpt <ckpt> --arm xattn|adaln|k4 --steps 500
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch as th
import torch.nn.functional as F

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="")
ap.add_argument("--arm", default="xattn", choices=["xattn", "adaln", "k4", "all"])
ap.add_argument("--csv", default="exp-std/csv/train.csv")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--steps", type=int, default=500)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--t-fixed", type=float, default=0.02)
a = ap.parse_args()
dev = th.device("cuda" if th.cuda.is_available() else "cpu")

if not a.ckpt:
    import glob
    cks = sorted(glob.glob("exp-std/runs_purestd/*p1.0/checkpoints/*.pt"))
    a.ckpt = max(cks, key=lambda p: int(os.path.basename(p)[:-3]))
print(f"[ckpt] {a.ckpt}   arm={a.arm}")

from src.eval.model_io import load_model_from_ckpt            # noqa: E402
# 该 ckpt 顶层只有 args/ema/delta/opt/scheduler (无 "model" 键) -> 必须用 EMA 权重
model, args = load_model_from_ckpt(a.ckpt, device=dev, use_ema=True, verbose=False)
model.train()

# ── 冻结主干, 只留注入链 ──────────────────────────────────────────────
PATS = {
    "xattn": ("glyph_injections", "glyph_embedder", "y_callig_embedder", "cond_fusion"),
    "adaln": ("glyph_injections", "glyph_embedder", "y_callig_embedder", "cond_fusion"),
    "k4": ("y_callig_embedder", "callig_style_ca", "glyph_injections", "cond_fusion",
           "style_proj", "style_role"),
    "all": ("",),
}[a.arm]
n_tr = n_fr = 0
for name, p in model.named_parameters():
    ok = (a.arm == "all") or any(pat in name for pat in PATS)
    p.requires_grad_(ok)
    n_tr += ok
    n_fr += (not ok)
print(f"[freeze] 可训张量 {n_tr} / 冻结 {n_fr}")

# ── 造 hard batch: 同一字 × 最多书家 ─────────────────────────────────
rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
by_char = {}
for r in rows:
    by_char.setdefault(r["character"], []).append(r)
char, cand = max(by_char.items(), key=lambda kv: len({x["calligrapher"] for x in kv[1]}))
uniq = {}
for r in cand:                                   # 每个书家留一张
    uniq.setdefault(r["calligrapher"], r)
picked = list(uniq.values())[:a.batch]
while len(picked) < a.batch:                     # 不够就补(重复无害)
    picked.append(picked[len(picked) % len(uniq)])
print(f"[batch] 字='{char}'  书家 {len(uniq)} 个  取 {len(picked)} 条 "
      f"(同字异风格, 最跨度的 hard batch)")

# ── 取 latent (目标 16ch/4ch 由 shard 决定; 条件 = std 骨架) ──────────
def shard_index(sdir):
    import glob
    idx, cache = {}, {}
    for sp in sorted(glob.glob(os.path.join(sdir, "shard_*.npz"))):
        d = np.load(sp)
        for j, x in enumerate(d["img_ids"].tolist()):
            idx[int(x)] = (sp, j)
    return idx, cache


XI, XC = shard_index(args.latent_shards_dir)
SI, SC = shard_index(args.skel_latent_shards_dir)


def get(idict, cache, sdir, i):
    sp, j = idict[int(i)]
    if sp not in cache:
        cache[sp] = np.load(sp)["latents"]
    return th.from_numpy(cache[sp][j].astype(np.float32))


x0 = th.stack([get(XI, XC, args.latent_shards_dir, r["img_id"]) for r in picked]).to(dev)
g = th.stack([get(SI, SC, args.skel_latent_shards_dir, r["img_id"]) for r in picked]).to(dev)
print(f"[latent] x0{tuple(x0.shape)}  g{tuple(g.shape)}")

# ── 装配 model_kwargs (与训练一致; 缺键给兜底, 不静默) ────────────────
model_kwargs = {}
try:
    model_kwargs = model.build_cond_kwargs_from_rows(picked)     # 若模型自带
except Exception:                                                # noqa: BLE001
    pass
if not model_kwargs:
    from src.utils.callig_script_map import load_callig_script_map
    cmap_p = getattr(args, "callig_script_map", "") or "exp-std/csv/callig_script_id_map_top10.json"
    cm = load_callig_script_map(cmap_p)
    y = th.tensor([int(cm["pair_map"][f"{r['calligrapher_id']}:{r['script_id']}"])
                   for r in picked], device=dev)
    model_kwargs = {"y_callig": y, "g": g}
    # ★ 即使 no_char_cond=true, forward 仍要求 y_char 位置参数(实测 TypeError) -> 给全
    model_kwargs["y_char"] = th.tensor(
        [int(r.get("character_id") or 0) for r in picked], device=dev)
print(f"[kw] {[(k, tuple(v.shape) if hasattr(v, 'shape') else v) for k, v in model_kwargs.items()]}")

# ── flow loss (直接手写, 不依赖 diffusion 封装) ───────────────────────
t = th.full((a.batch,), a.t_fixed, device=dev)
noise = th.randn_like(x0)
x_t = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * noise
v_tgt = noise - x0

opt = th.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr)
losses = []
t0 = time.time()
for s in range(1, a.steps + 1):
    opt.zero_grad(set_to_none=True)
    with th.autocast("cuda", dtype=th.bfloat16):
        out = model(x_t, t, **model_kwargs)
    if out.shape[1] == 2 * v_tgt.shape[1]:
        out = out[:, : v_tgt.shape[1]]
    loss = F.mse_loss(out.float(), v_tgt.float())
    loss.backward()
    opt.step()
    losses.append(float(loss.detach()))
    if s % 50 == 0 or s == 1:
        print(f"  step {s:>4}  loss={losses[-1]:.5f}  "
              f"({losses[-1] / losses[0] * 100:.1f}% of init)  {time.time() - t0:.0f}s",
              flush=True)

print(f"\n[探针二] arm={a.arm}  字='{char}'  书家 {len(uniq)}")
print(f"  loss: init={losses[0]:.5f} -> final={losses[-1]:.5f}  "
      f"降幅 {100 * (1 - losses[-1] / losses[0]):.1f}%")
print(f"  最后 100 步均值={np.mean(losses[-100:]):.5f}   "
      f"平台期相对噪声={np.std(losses[-100:]) / (np.mean(losses[-100:]) + 1e-12):.4f}")
print("判读: 降幅>70% 且平台期低 -> 容量够 ✓; 早早平底(降幅<30%) -> 注入头是瓶颈 ✗")
