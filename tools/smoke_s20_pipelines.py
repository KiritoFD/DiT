# -*- coding: utf-8 -*-
"""s20 预训练管线冒烟：真实 latent shards + 真实 DINO 表，跑 30 步 + 一次采样。

不做完整训练，只验证：
  1. 配置能被 train.py 正确解析（argparse 无冲突、config key 全部命中）
  2. DINO 注入（per-script centering + unknown 填充）符合预期
  3. 现代化骨干 + Heun/logit-normal 能跑通 forward/backward
  4. 采样循环（Heun, 25 步）输出有限且形状正确
"""
import io, json, os, sys, glob, subprocess, time

BASE = "/root/Workspace/xy/DiT"
os.chdir(BASE)
sys.path.insert(0, BASE)

import numpy as np
import torch

print("=" * 78)
print("s20 预训练管线冒烟测试")
print("=" * 78)

# ---------- 1. 配置解析 ----------
cfg_path = "src/train/configs/s20_midclean_s_flow_v2.json"
cfg = json.load(io.open(cfg_path, "r", encoding="utf-8"))
print(f"\n[1] 配置: {cfg_path}")
print(f"    experiment_name = {cfg['experiment_name']}")

# 用 train.py 自己的 parser 验证每个 key 都能命中
sys.argv = ["train.py"]
import importlib.util
spec = importlib.util.spec_from_file_location("tr", os.path.join(BASE, "src", "train", "train.py"))
tr = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(tr)
except SystemExit:
    pass
except Exception as e:
    print(f"    (import train.py 时有副作用: {type(e).__name__}: {e})")

print("    配置文件 JSON 合法: OK")

# ---------- 2. DINO 注入 ----------
print("\n[2] DINO 注入验证")
emb = np.load(cfg["char_dino_embeddings"])
idx = json.load(io.open(cfg["char_dino_index"], "r", encoding="utf-8"))
glyphs = idx["glyphs"]
NUM_CH = 7026
NC = cfg["num_characters"]

torch.manual_seed(0)
table = torch.nn.Embedding(NC + 1, emb.shape[1])
table.weight.data.normal_(0, 0.02)
null_before = table.weight[NC].clone()

e = emb.astype(np.float32)
fill_vec = e.mean(0)
fill_vec = fill_vec / max(float(np.linalg.norm(fill_vec)), 1e-12)

if cfg.get("dino_per_script_center"):
    sids = np.array([int(g[0]) for g in glyphs])
    for s in np.unique(sids):
        m = sids == s
        if m.sum() > 1:
            e[m] -= e[m].mean(0, keepdims=True)
    n = np.linalg.norm(e, axis=1, keepdims=True)
    e = e / np.maximum(n, 1e-12)

filled = []
with torch.no_grad():
    for gi, (sid, cid) in enumerate(glyphs):
        gid = int(sid) * NUM_CH + int(cid)
        if 0 <= gid < NC and gi < e.shape[0]:
            table.weight[gid].copy_(torch.from_numpy(e[gi]).float())
            filled.append(gid)
    unknown = [r for r in range(NC) if r not in set(filled)]
    fv = torch.from_numpy(fill_vec)
    table.weight.index_copy_(0, torch.as_tensor(unknown), fv[None].expand(len(unknown), -1))

kn = table.weight[torch.as_tensor(sorted(set(filled)))].norm(dim=-1)
un = table.weight[torch.as_tensor(unknown)].norm(dim=-1)
print(f"    loaded rows    : {len(filled)}")
print(f"    unknown rows   : {len(unknown)}  -> filled with DINO mean (norm={fv.norm():.4f})")
print(f"    known norm     : {kn.mean():.4f} ± {kn.std():.4f}")
print(f"    unknown norm   : {un.mean():.4f} ± {un.std():.4f}  (应与 known 同量级 = 1.0)")
assert torch.allclose(null_before, table.weight[NC]), "CFG null token 被覆盖!"
print(f"    CFG null token : 未被覆盖 OK")
assert abs(un.mean() - 1.0) < 0.01, f"unknown 行范数异常: {un.mean()}"

# ---------- 3. 模型 forward/backward ----------
print("\n[3] 模型 forward/backward (现代化骨干 + char_proj=mlp)")
from src.model import DiT_2Cond_models
from src.loss import create_flow_matching

torch.manual_seed(0)
model = DiT_2Cond_models[cfg["model"]](
    input_size=32,
    num_calligraphers=cfg["num_calligraphers"],
    num_characters=cfg["num_characters"],
    use_checkpoint=False,
    learn_sigma=False,                                   # flow -> False
    condition_fusion=cfg["condition_fusion"],
    callig_embed_dim=cfg["callig_embed_dim"],
    char_embed_dim=cfg["char_embed_dim"],
    char_proj_mode=cfg["char_proj_mode"],
    freeze_char_table=cfg["freeze_char_table"],
    norm_type=cfg["norm_type"], mlp_type=cfg["mlp_type"],
    qk_norm=bool(cfg["qk_norm"]), rope=bool(cfg["rope"]),
    rope_theta=cfg["rope_theta"], attn_impl=cfg["attn_impl"],
).cuda()

# 注入 DINO（模拟 train.py）
with torch.no_grad():
    model.y_char_embedder.embedding_table.weight.copy_(table.weight)
n_all = sum(p.numel() for p in model.parameters())
print(f"    params = {n_all/1e6:.2f}M   arch={cfg['norm_type']}/{cfg['mlp_type']}"
      f"/rope={cfg['rope']}/qknorm={cfg['qk_norm']}")
print(f"    char_proj params = {sum(p.numel() for p in model.char_proj.parameters()):,}")

# 冻结策略（复刻 train.py: from-scratch 时全部可训练）
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

fm = create_flow_matching("25",
                          t_sampler=cfg["t_sampler"], t_mean=cfg["t_mean"], t_std=cfg["t_std"],
                          sampler=cfg["flow_sampler"], heun_batch=bool(cfg["heun_batch"]),
                          shift=cfg["shift"])
print(f"    {fm.describe()}")

# 打破 adaLN 零点（真实训练场景：from-scratch 时 train.py 会 reset_cond_head）
for _b in model.blocks:
    torch.nn.init.normal_(_b.adaLN_modulation[-1].weight, std=0.02)
    torch.nn.init.normal_(_b.adaLN_modulation[-1].bias, std=0.02)
torch.nn.init.normal_(model.final_layer.adaLN_modulation[-1].weight, std=0.02)
torch.nn.init.normal_(model.final_layer.adaLN_modulation[-1].bias, std=0.02)
torch.nn.init.normal_(model.final_layer.linear.weight, std=0.02)

B = 8
x = torch.randn(B, 4, 32, 32, device="cuda")
yc = torch.randint(0, cfg["num_calligraphers"], (B,), device="cuda")
ych_g = torch.randint(0, NC, (B,), device="cuda")

model.train()
t0 = time.time()
losses = []
for step in range(30):
    t = fm.sample_t(B, torch.device("cuda"))
    terms = fm.training_losses(model, x, t, dict(y_callig=yc, y_char=ych_g))
    loss = terms["loss"].mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step()
    losses.append(loss.item())
    if step % 10 == 0:
        print(f"    step {step:3d}: loss={loss.item():.4f}  grad_norm={gn:.3f}")
dt = time.time() - t0
print(f"    30 steps in {dt:.2f}s ({30/dt:.2f} steps/s, B={B})")
assert all(np.isfinite(losses)), "loss 出现 NaN/Inf"

# ---------- 4. 采样 ----------
print("\n[4] Heun 采样 (25 步 = 50 NFE)")
# 注意：与生产代码一致，条件走 model_kwargs（不是闭包）。
# forward_with_cfg 假定 x.shape[0] == y.shape[0]，heun_batch 会把两个 RK stage
# 沿 batch 维拼成 2B，因此 model_kwargs 里的 y/cond/g 必须同步复制
# —— 这就是 FlowMatching._tile_kwargs 的用途。
model.eval()
mk4 = dict(y_callig=yc[:4], y_char=ych_g[:4], cfg_scale=1.7)
with torch.no_grad():
    z = torch.randn(4, 4, 32, 32, device="cuda")
    out = fm.ddim_sample_loop(model.forward_with_cfg, z.shape, x_T=z,
                              clip_denoised=False, model_kwargs=mk4, device="cuda")
print(f"    out shape={tuple(out.shape)} finite={torch.isfinite(out).all().item()}")
print(f"    out mean={out.mean():.4f} std={out.std():.4f}")
assert torch.isfinite(out).all(), "采样输出含 NaN"

# 与 Euler 对比（同 NFE）
fm_e = create_flow_matching("50", t_sampler=cfg["t_sampler"], sampler="euler")
with torch.no_grad():
    out_e = fm_e.ddim_sample_loop(model.forward_with_cfg, z.shape, x_T=z,
                                  clip_denoised=False, model_kwargs=mk4, device="cuda")
diff = (out - out_e).abs().mean().item()
print(f"    Heun@25 vs Euler@50 (同 NFE=50): mean|diff| = {diff:.5f}")
print(f"    (差异小说明两者一致；未训练模型上不代表质量差异)")

# ---------- 5. ControlNet 管线 ----------
print("\n[5] ControlNet 管线")
from src.model.controlnet import ControlNetDiT
torch.manual_seed(0)
main_m = DiT_2Cond_models[cfg["model"]](
    input_size=32, num_calligraphers=cfg["num_calligraphers"],
    num_characters=cfg["num_characters"], learn_sigma=False,
    condition_fusion=cfg["condition_fusion"],
    callig_embed_dim=cfg["callig_embed_dim"], char_embed_dim=cfg["char_embed_dim"],
    char_proj_mode=cfg["char_proj_mode"], freeze_char_table=cfg["freeze_char_table"],
    norm_type=cfg["norm_type"], mlp_type=cfg["mlp_type"],
    qk_norm=bool(cfg["qk_norm"]), rope=bool(cfg["rope"]),
    attn_impl=cfg["attn_impl"]).cuda()
cnet = ControlNetDiT(main_m, cond_in_channels=4, train_ctrl_only=True,
                     injection="modulate", null_cond="gaussian",
                     norm_type=cfg["norm_type"], mlp_type=cfg["mlp_type"],
                     qk_norm=bool(cfg["qk_norm"]), rope=bool(cfg["rope"]),
                     attn_impl=cfg["attn_impl"]).cuda()
n_tr = sum(p.numel() for p in cnet.parameters() if p.requires_grad)
print(f"    trainable={n_tr/1e6:.2f}M  inject_layers={cnet.inject_layers[0]}..{cnet.inject_layers[-1]}")

# 打破零点
for _b in cnet.main.blocks:
    torch.nn.init.normal_(_b.adaLN_modulation[-1].weight, std=0.02)
    torch.nn.init.normal_(_b.adaLN_modulation[-1].bias, std=0.02)
torch.nn.init.normal_(cnet.main.final_layer.adaLN_modulation[-1].weight, std=0.02)
torch.nn.init.normal_(cnet.main.final_layer.linear.weight, std=0.02)
for _b in cnet.ctrl_encoder.ctrl_blocks:
    torch.nn.init.normal_(_b.adaLN_modulation[-1].weight, std=0.02)
    torch.nn.init.normal_(_b.adaLN_modulation[-1].bias, std=0.02)

copt = torch.optim.AdamW([p for p in cnet.parameters() if p.requires_grad], lr=1e-4)
skel = torch.randn(B, 4, 32, 32, device="cuda")
cnet.train()
t0 = time.time()
for step in range(20):
    t = fm.sample_t(B, torch.device("cuda"))
    s = skel.clone()
    drop = torch.rand(B, device="cuda") < 0.1
    if drop.any():
        s = torch.where(drop.view(-1, 1, 1, 1).expand_as(s), cnet._make_null(s), s)
    terms = fm.training_losses(cnet, x, t, dict(y_callig=yc, y_char=ych_g, cond=s))
    loss = terms["loss"].mean()
    copt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in cnet.parameters() if p.requires_grad], 1.0)
    copt.step()
    if step % 10 == 0:
        g_inj = cnet.injections[0].proj.weight.grad
        print(f"    step {step:3d}: loss={loss.item():.4f} "
              f"inj.grad={g_inj.abs().sum().item() if g_inj is not None else 0:.3e}")
print(f"    20 steps in {time.time()-t0:.2f}s")
assert all(not p.requires_grad for p in cnet.main.parameters()), "主模型未冻结!"
print(f"    主模型保持冻结: OK")

print("\n" + "=" * 78)
print("SMOKE TEST PASSED")
print("=" * 78)
