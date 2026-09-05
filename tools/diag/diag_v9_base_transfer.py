# -*- coding: utf-8 -*-
"""diag_v9_base_transfer.py — 为什么更强的 v9a base 让 ctrl 后训练变差?

受控对比 (全部同 batch / 同 flow loss / fp32):
  D1 主干特征范数轮廓: v8a vs v9a 逐 block (注入的相对尺度)
  D2 零初始注入梯度种子: fresh ctrl 在两个主干上的逐层注入梯度范数 (学习信号强度)
  D3 已训练 ctrl 的注入输出幅度 |s|,|t| 与条件响应 |Δout| (v8b/v8e/v9b)
  D4 骨架输入梯度 ||dL/d skel|| (骨架信号被放大还是衰减)
  D5 主干条件损失 (sanity: 无 ctrl 时 v8a vs v9a)

用法: /opt/conda/envs/cu121/bin/python tools/diag/diag_v9_base_transfer.py
"""
import os, sys, glob, json
import numpy as np
import torch

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")
from src.model.controlnet import load_main_model, ControlNetDiT
from src.eval.inference import make_eval_cache
from src.loss import create_diffusion_or_flow

dev = torch.device("cuda")
ARCH = dict(norm_type="rms", mlp_type="swiglu", qk_norm=True, rope=True,
            rope_theta=100.0, attn_impl="sdpa")
COMMON = dict(model_name="DiT-2Cond-S/2", device=dev, num_calligraphers=1013,
              num_characters=35130, condition_fusion="factorized_add",
              callig_embed_dim=128, char_embed_dim=384, char_proj_mode="mlp",
              freeze_char_table=True, learn_sigma=False, **ARCH)


def strip(sd):
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}


def find_ckpt(base_glob, step):
    """在多个 run 目录里找指定 step 的 ckpt (取 eval 指标最高的 run).
    兼容 eval json 的补零/不补零命名与 ssim_mean/ssim 键名."""
    best, bestf = -1, None
    for f in glob.glob(base_glob):
        step_int = int("".join(c for c in os.path.basename(f).split(".")[0] if c.isdigit()))
        if step_int != step:
            continue
        rd = os.path.dirname(f)
        s = 0.0
        for ej in (os.path.join(rd, f"eval_auto_ctrl_{step_int:07d}.json"),
                   os.path.join(rd, f"eval_auto_ctrl_{step_int}.json")):
            if os.path.exists(ej):
                d = json.load(open(ej)).get("ctrl", {})
                s = d.get("ssim_mean", d.get("ssim", 0.0))
                break
        if s >= best:
            best, bestf = s, f
    return bestf


V8A = "5script/results/v8_3stage/A_main_final.pt"
V9A = sorted(glob.glob("5script/results/v9a_repa_pretrain/*/checkpoints/0130000.pt"))[-1]
PAIRS = {  # tag -> (base_ckpt, ctrl_ckpt, trained_step)
    "v8b": (V8A, find_ckpt("5script/results/v8_3stage/v8b/*/checkpoints/*.pt", 35000), 35000),
    "v8e": (V8A, find_ckpt("5script/results/v8_3stage/v8e/*/checkpoints/*.pt", 22500), 22500),
    "v9b": (V9A, find_ckpt("5script/results/v9b_ctrl_strong/*/checkpoints/*.pt", 35000), 35000),
}
print("v8a =", V8A, "\nv9a =", V9A)
for k, (b, c, s) in PAIRS.items():
    print(f"pair {k}: ctrl={c}")

# ---- 固定评测 batch (16 样本, 与训练 eval 同源) ----
N = 16
cache = make_eval_cache("5script/eval_fame_strict_clean_v8.csv", "final_imgs_fame_v8", None,
                        256, N, 8, 4, 0.18215,
                        skel_latent_shards_dir="final_skel_latents_fame_1px_v8")
yc = torch.tensor([c[0] for c in cache["conds"]], device=dev)
yh = torch.tensor([c[1] for c in cache["conds"]], device=dev)
skel = cache["skels_latent"].float().to(dev)
g = torch.Generator().manual_seed(123)
noise = torch.randn(N, 4, 32, 32, generator=g).to(dev)
t05 = torch.full((N,), 0.5, device=dev)
flow = create_diffusion_or_flow(timestep_respacing="", diffusion_type="flow")


def main_fwd(main, x_t, t, cond=None):
    if cond is None:
        return main(x_t, t * 1000.0, yc, yh)
    raise RuntimeError("main 不吃 cond")


def block_hooks(m):
    feats = {}
    hs = []
    for i, blk in enumerate(m.blocks):
        def fn(mod, inp, out, i=i):
            if isinstance(out, tuple):
                out = out[0]
            feats[i] = out.detach()
        hs.append(blk.register_forward_hook(fn))
    return feats, hs


def loss_fn(model, x_start, with_cond, skel_in=None):
    t = t05
    kwargs = dict(y_callig=yc, y_char=yh)
    if with_cond:
        kwargs["cond"] = skel_in
    terms = flow.training_losses(model, x_start, t, kwargs, noise=noise)
    return terms["loss"].mean()


print("\n========== D5 主干条件损失 (无 ctrl, 同 batch) ==========")
trunk_loss = {}
for tag, ck in [("v8a", V8A), ("v9a", V9A)]:
    m = load_main_model(ckpt_path=ck, **COMMON)
    m.eval()
    x_start = torch.randn(N, 4, 32, 32, generator=torch.Generator().manual_seed(7)).to(dev)
    with torch.no_grad():
        tl = loss_fn(m, x_start, with_cond=False).item()
    trunk_loss[tag] = tl
    print(f"  {tag}: flow loss = {tl:.5f}")
    del m
    torch.cuda.empty_cache()

print("\n========== D1 主干逐 block 特征范数 (t=0.5, 同 batch) ==========")
norms = {}
for tag, ck in [("v8a", V8A), ("v9a", V9A)]:
    m = load_main_model(ckpt_path=ck, **COMMON)
    m.eval()
    x_t = (1 - 0.5) * torch.randn(N, 4, 32, 32, generator=torch.Generator().manual_seed(7)).to(dev) \
        + 0.5 * noise  # 与 D5 同 x_start 的 x_t
    feats, hs = block_hooks(m)
    with torch.no_grad():
        main_fwd(m, x_t, t05)
    for h in hs:
        h.remove()
    norms[tag] = {i: f.norm(dim=-1).mean().item() for i, f in feats.items()}
    del m, feats, hs
    torch.cuda.empty_cache()
print(f"  {'block':>6} {'v8a':>9} {'v9a':>9} {'ratio':>7}")
for i in sorted(norms["v8a"]):
    r = norms["v9a"][i] / max(norms["v8a"][i], 1e-9)
    mark = " ←注入层" if i in (5, 8, 11) else ""
    print(f"  {i:>6} {norms['v8a'][i]:>9.2f} {norms['v9a'][i]:>9.2f} {r:>7.3f}{mark}")

print("\n========== D2 零初始注入梯度种子 (fresh ctrl, 逐注入层) ==========")
for tag, ck in [("v8a", V8A), ("v9a", V9A)]:
    m = load_main_model(ckpt_path=ck, **COMMON)
    m.eval()
    torch.manual_seed(0)
    c = ControlNetDiT(m, cond_in_channels=4, train_ctrl_only=True,
                      injection="modulate", null_cond="gaussian").to(dev)
    x_start = torch.randn(N, 4, 32, 32, generator=torch.Generator().manual_seed(7)).to(dev)
    loss = loss_fn(c, x_start, with_cond=True, skel_in=skel)
    loss.backward()
    inj_g = []
    for i, inj in enumerate(c.injections):
        gn = inj.proj.weight.grad.norm().item()
        inj_g.append(gn)
    tot = sum(p.grad.norm().item() ** 2 for p in c.ctrl_encoder.parameters()
              if p.grad is not None) ** 0.5
    print(f"  {tag}: loss={loss.item():.5f} ctrl_enc_grad={tot:.3f} "
          f"inj_grads={['%.2f' % v for v in inj_g]}")
    del c, m
    torch.cuda.empty_cache()

print("\n========== D3/D4 已训练 ctrl: 注入幅度 / 条件响应 / 骨架梯度 ==========")
print(f"  {'model':>6} {'|s|mean':>9} {'|t|mean':>9} {'|Δout|':>9} {'|Δout|rel':>10} {'dL/dskel':>10}")
results = {}
for tag, (bck, cck, step) in PAIRS.items():
    m = load_main_model(ckpt_path=bck, **COMMON)
    m.eval()
    c = ControlNetDiT(m, cond_in_channels=4, train_ctrl_only=True,
                      injection="modulate", null_cond="gaussian").to(dev)
    ckd = torch.load(cck, map_location="cpu", weights_only=False)
    sd = strip(ckd.get("ema") or ckd.get("ctrl"))
    c.load_state_dict(sd, strict=False)
    c.eval()
    # 注入输出幅度 (hook proj 输出 = [s|t])
    caps = {}
    hs = []
    for i, inj in enumerate(c.injections):
        def fn(mod, inp, out, i=i):
            caps[i] = out.detach()
        hs.append(inj.proj.register_forward_hook(fn))
    x_start = torch.randn(N, 4, 32, 32, generator=torch.Generator().manual_seed(7)).to(dev)
    with torch.no_grad():
        out_sk = c(x_start, t05 * 1000.0, yc, yh, cond=skel)
        out_no = c(x_start, t05 * 1000.0, yc, yh, cond=None)
    for h in hs:
        h.remove()
    s_abs = np.mean([caps[i].float().abs().mean().item() for i in caps])
    d_out = (out_sk - out_no).abs().mean().item()
    d_rel = d_out / out_no.abs().mean().item()
    # D4: 骨架输入梯度
    sk = skel.clone().requires_grad_(True)
    loss = loss_fn(c, x_start, with_cond=True, skel_in=sk)
    loss.backward()
    gsk = sk.grad.norm().item() / N
    results[tag] = (s_abs, d_out, d_rel, gsk)
    print(f"  {tag:>6} {s_abs:>9.4f} {s_abs:>9.4f} {d_out:>9.4f} {d_rel:>10.4f} {gsk:>10.4f}")
    del c, m, ckd, sd
    torch.cuda.empty_cache()

print("\n========== 汇总比值 (v9b/v8e, 受控: 同配方不同 base) ==========")
if "v8e" in results and "v9b" in results:
    a, b = results["v8e"], results["v9b"]
    print(f"  注入幅度比 {b[0]/a[0]:.3f} | 条件响应比 {b[2]/a[2]:.3f} | 骨架梯度比 {b[3]/a[3]:.3f}")
    print(f"  主干特征范数比 (block8) {norms['v9a'][8]/norms['v8a'][8]:.3f}")
