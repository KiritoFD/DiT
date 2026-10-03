# -*- coding: utf-8 -*-
"""
lr_analyze.py — 解析法估算 LR (不跑训练实验).

A) UWR (update/weight ratio, Adam 精确): 从 ckpt 的优化器二阶态直接算
   m̂/(√v̂+ε) (与 lr 无关) × 候选 lr = 实际单步位移; / RMS(w) = UWR.
   经验稳区: AdamW 微调 UWR ≈ 1e-3 ~ 2e-3 (数据分布偏移可取上限).
B) GNS (gradient noise scale): K 个批次的扩散 velocity loss 全量梯度,
   B_crit = B × mean_j||g_j - ḡ||² / ||ḡ||²; batch 128 vs B_crit 判据:
   B << B_crit → 梯度噪声主导, 可升 lr/降 batch; B >> B_crit → 数据极限.

用法: /opt/conda/envs/cu121/bin/python _sync_work/lr_analyze.py <ckpt.pt>
"""
import json
import math
import sys
import time

import numpy as np
import torch

ROOT = "/root/Workspace/xy/DiT"
import os

os.chdir(ROOT)
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CKPT = sys.argv[1]
BATCH = 128
K_GNS = 12
CFG = "src/train/configs/v10b_stdskel_fame3_c41x_cos_e.json"
args = json.load(open(CFG, encoding="utf-8"))

print(f"ckpt: {CKPT}")
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
opt_sd = ck["opt"]

# ---------- A) UWR ----------
from src.model import DiT_2Cond_models

model = DiT_2Cond_models["DiT-2Cond-Sp/2"](
    num_calligraphers=args["num_calligraphers"], num_characters=args["num_characters"],
    condition_fusion=args["condition_fusion"], callig_embed_dim=args["callig_embed_dim"],
    char_embed_dim=args["char_embed_dim"], char_proj_mode=args["char_proj_mode"],
    learn_sigma=False, use_glyph_cond=True, use_char_cond=False,
    glyph_inject_layers=args["glyph_inject_layers"], glyph_inject_mode=args["glyph_inject_mode"],
    glyph_embedder_depth=args["glyph_embedder_depth"], callig_spatial=False).cpu()

model_sd = ck.get("delta", ck.get("model", {}))
model_sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in model_sd.items()}
model.load_state_dict(model_sd, strict=False)
named = [n for n, _ in model.named_parameters()]
n_params = list(model.parameters())
flat_w = {n: p.detach().float() for n, p in model.named_parameters()}

# opt.state 的 key 是 param 在创建时的索引; param_groups 记录 indices
groups = opt_sd["param_groups"]
states = opt_sd["state"]
print("\n=== A) UWR (来自优化器二阶态, 无需前向) ===")
print(f"{'group':>6} {'n_params':>10} {'RMS(w)':>10} {'mean|m/√v|':>11} {'lr=3e-5':>9} {'lr=5e-5':>9} {'lr=1e-4':>9} {'lr=1.5e-4':>9}")
tot_w2, tot_u, tot_n = 0.0, 0.0, 0
for g in groups:
    idxs = g["params"]
    ws, us = [], []
    for i in idxs:
        st = states.get(i)
        if st is None or "exp_avg_sq" not in st or i >= len(n_params):
            continue  # REPA proj 等追加参数不在模型里, 跳过
        w = n_params[i].detach().float()
        m = st["exp_avg"].float()
        v = st["exp_avg_sq"].float()
        t = float(st.get("step", 1000))
        m_hat = m / (1 - 0.9 ** t)
        v_hat = v / (1 - 0.999 ** t)
        u = (m_hat.abs() / (v_hat.sqrt() + 1e-8)).mean().item()
        ws.append(w.pow(2).mean().item())
        us.append(u)
        tot_w2 += w.pow(2).sum().item()
        tot_u += (m_hat.abs() / (v_hat.sqrt() + 1e-8)).sum().item()
        tot_n += w.numel()
    rms_w = float(np.sqrt(np.mean(ws))) if ws else 0.0
    mu = float(np.mean(us)) if us else 0.0
    row = " ".join(f"{lr * mu / max(rms_w, 1e-12):9.2e}" for lr in (3e-5, 5e-5, 1e-4, 1.5e-4))
    print(f"{g.get('gid', '-')!s:>6} {len(idxs):>10} {rms_w:>10.3e} {mu:>11.3f} {row}")

mu_all = tot_u / max(tot_n, 1)
rms_all = math.sqrt(tot_w2 / max(tot_n, 1))
print(f"\n全局: RMS(w)={rms_all:.3e}  mean|m̂/√v̂|={mu_all:.3f}")
for lr in (5e-6, 1.5e-5, 3e-5, 5e-5, 1e-4, 1.5e-4):
    print(f"  lr={lr:.1e} -> 单步位移/RMS(w) = {lr * mu_all / rms_all:.2e}")

# ---------- B) GNS ----------
print("\n=== B) GNS (K 批次扩散 velocity 梯度) ===")
dev = torch.device("cuda")
model = model.to(dev)
model.requires_grad_(True)
from src.utils.latent_dataset import MCCDLatentDataset
from src.loss import create_diffusion_or_flow, flow_kwargs_from

from src.utils.callig_map import load_callig_id_map
_cmap, _ = load_callig_id_map(args["callig_id_map"])
ds = MCCDLatentDataset(csv_file=args["data_csv"],
                       latent_shards_dir=args["latent_shards_dir"],
                       img_root=args["img_root"],
                       image_size=256, load_canny=False, load_skel=False,
                       skel_latent_shards_dir=args["skel_latent_shards_dir"],
                       preload=True, load_image=False,
                       callig_id_map=_cmap)
dl = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=4,
                                 drop_last=True, prefetch_factor=2)
_kw = flow_kwargs_from(type("A", (), {**args, "diffusion_type": args.get("diffusion_type", "flow"),
                                      "t_mean": args.get("t_mean", 0.0), "t_std": args.get("t_std", 1.0),
                                      "shift": args.get("shift", 1.0), "flow_sampler": args.get("flow_sampler", "heun")}))
diff = create_diffusion_or_flow(timestep_respacing="", diffusion_type=args.get("diffusion_type", "flow"), **_kw)

params = [p for p in model.parameters() if p.requires_grad]
grads = []
it = iter(dl)
t0 = time.time()
for k in range(K_GNS):
    try:
        batch = next(it)
    except StopIteration:
        it = iter(dl)
        batch = next(it)
    x = batch["latent"].to(dev).float()
    y = batch["y_callig"].to(dev)
    g = batch["skel_latent"].to(dev).float()
    t = diff.sample_t(x.shape[0], dev)
    for p in params:
        p.grad = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ld = diff.training_losses(model, x, t, dict(y_callig=y, y_char=None, g=g))
        loss = ld["loss"].mean()
    loss.backward()
    gf = torch.cat([(p.grad.detach().float().reshape(-1)) for p in params if p.grad is not None])
    grads.append(gf.cpu())
    print(f"  batch {k}: loss={loss.item():.4f} ||g||={gf.norm().item():.4f} ({time.time() - t0:.0f}s)")

G = torch.stack(grads)            # (K, P)
gbar = G.mean(0)
num = (G - gbar).pow(2).sum(1).mean()   # E||g_j - gbar||²
den = gbar.pow(2).sum()
B_crit = BATCH * float(num) / max(float(den), 1e-12)
cosm = torch.nn.functional.cosine_similarity(G, gbar.unsqueeze(0), dim=1)
print(f"\n||gbar||={den.sqrt().item():.4f}  mean cos(g_j, gbar)={cosm.mean().item():.3f}")
print(f"B_crit ≈ {B_crit:.1f}  (batch={BATCH})")
if B_crit > 2 * BATCH:
    print("  → batch 远小于临界值: 梯度噪声主导, LR 有上调空间")
elif B_crit < BATCH:
    print("  → batch 已超过临界值: 更大 batch/更小 lr 收益有限")
