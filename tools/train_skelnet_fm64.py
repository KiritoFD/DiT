#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_skelnet_fm64.py — **64x64 像素域 Flow Matching** 骨架生成 (复刻主干口径)。

用户 2026-09-30 裁定:
  · 像素域就做 diffusion/flow, 残差+旁路那一套全部扔掉;
  · 参考**主干 flow 模型的现代化**, 不要 class-ic DDPM。

口径逐条对齐主干 (src/loss/flow_matching.py + configs/*.json):
  · 线性插值:  x_t = (1-t)*x0 + t*noise,  t ∈ [0,1]  (t=0 数据, t=1 纯噪声)
  · 预测目标:  v = noise - x0 (velocity), loss = **纯 MSE**(无加权, 与主干一致)
  · t 采样:    logit_normal, t = sigmoid(N(mean=0, std=1))   ← SD3, 主干默认
  · 时间喂入:  model(..., t * TIME_SCALE), TIME_SCALE=1000  (与 DDPM 相位对齐)
  · 采样器:    **Heun**(二阶, 每步 2 NFE), 网格 s=linspace(1,0,steps+1),
               t = shift*s/(1+(shift-1)*s)                    ← SD3 timestep shift
  · CFG:       全局条件(书家风格)做 drop(0.1) + 采样期 CFG; **骨架 g 两支都给**
               (与 in_process_eval.py 的约定一致)

数据: tools/build_skel64_dataset.py 落盘的 64² 数据集
  x0   = gt_skel64  (书法家骨架, 1px@64, [-1,1], 墨=-1)
  cond = std_skel64 (标准骨架, 1px@64) 拼通道 + 书家x书体风格(128d) 走 FiLM

刻意不做: g 硬旁路(λ)、latent 域预测、残差目标 —— 前几轮实测都失败:
  λ=4 时 sigmoid'(4)≈0.018 压死有墨处梯度(eval 完全冻结); latent MSE 会回归成淡影。

用法:
  python tools/train_skelnet_fm64.py --steps 10000 --batch 2048
  python tools/train_skelnet_fm64.py --dump --resume <ckpt> --dump-width 5
"""
import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

TIME_SCALE = 1000.0


class SinEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        f = th.exp(-math.log(10000) * th.arange(half, dtype=th.float32,
                                                device=t.device) / half)
        a = t.float()[:, None] * f[None]
        return th.cat([a.sin(), a.cos()], 1)


def cbr(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False),
                         nn.GroupNorm(8, o), nn.SiLU(inplace=True))


class FiLM(nn.Module):
    def __init__(self, cd, ch):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(cd, ch * 2), nn.SiLU(),
                                 nn.Linear(ch * 2, ch * 2))
        nn.init.zeros_(self.mlp[-1].weight)      # ★ 零初始化: 起点是恒等调制
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, c):
        s, b = self.mlp(c).chunk(2, 1)
        return x * (1 + s[:, :, None, None]) + b[:, :, None, None]


class SelfAttn2d(nn.Module):
    """瓶颈处的全局自注意力 —— 卷积 UNet 的感受野不足以协调整字笔画。"""

    def __init__(self, ch, heads=4):
        super().__init__()
        self.n = nn.GroupNorm(8, ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        t = self.n(x).reshape(b, c, h * w).transpose(1, 2)
        o, _ = self.attn(t, t, t, need_weights=False)
        return x + o.transpose(1, 2).reshape(b, c, h, w)


def soft_dice_prob(p, tgt, eps=1.0):
    num = 2 * (p * tgt).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + tgt.sum(dim=(1, 2, 3)) + eps
    return 1 - (num / den).mean()


class CondUNet(nn.Module):
    """输入 concat[x_t, g] (2ch) -> v。条件 = 时间嵌入 ⊕ 风格向量 -> FiLM。"""

    def __init__(self, base=32, adim=128, sdim=128, tdim=128, in_ch=2):
        super().__init__()
        b = base
        self.tmlp = nn.Sequential(SinEmb(tdim), nn.Linear(tdim, tdim), nn.SiLU(),
                                  nn.Linear(tdim, tdim))
        self.cmlp = nn.Sequential(nn.Linear(tdim + sdim, adim * 2), nn.SiLU(),
                                  nn.Linear(adim * 2, adim * 2))
        cd = adim * 2
        self.e1, self.e2, self.e3, self.e4 = (
            cbr(in_ch, b), cbr(b, b * 2), cbr(b * 2, b * 4), cbr(b * 4, b * 8))
        self.f1, self.f2, self.f3, self.f4 = (FiLM(cd, b), FiLM(cd, b * 2),
                                              FiLM(cd, b * 4), FiLM(cd, b * 8))
        self.pool = nn.MaxPool2d(2)
        self.mid = cbr(b * 8, b * 8)
        self.attn = SelfAttn2d(b * 8)                 # 瓶颈 (8x8) 全局注意力
        self.attn3 = SelfAttn2d(b * 4)                # ★ 16x16 也加一层: 笔画间的协调
        self.fm = FiLM(cd, b * 8)
        self.u3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.d3 = cbr(b * 8, b * 4)
        self.u2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.d2 = cbr(b * 4, b * 2)
        self.u1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.d1 = cbr(b * 2, b)
        self.out = nn.Conv2d(b, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.null_style = nn.Parameter(th.zeros(sdim))     # CFG 的 null 风格

    def cond(self, t, s):
        return self.cmlp(th.cat([self.tmlp(t * TIME_SCALE), s], 1))

    def forward(self, x, t, g, s, cvec=None):
        c = self.cond(t, s) if cvec is None else cvec
        x = th.cat([x, g], 1)
        e1 = self.f1(self.e1(x), c)
        e2 = self.f2(self.e2(self.pool(e1)), c)
        e3 = self.attn3(self.f3(self.e3(self.pool(e2)), c))
        e4 = self.fm(self.attn(self.mid(self.e4(self.pool(e3)))), c)
        d3 = self.d3(th.cat([self.u3(e4), e3], 1))
        d2 = self.d2(th.cat([self.u2(d3), e2], 1))
        d1 = self.d1(th.cat([self.u1(d2), e1], 1))
        return self.out(d1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/top10_style23/skel64")
    ap.add_argument("--style-emb", default="assets/callig_script_emb_top10.pt")
    ap.add_argument("--callig-map", default="assets/callig_script_id_map_top10.json")
    ap.add_argument("--vae", default="data/pretrained/pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--val-n", type=int, default=128)
    ap.add_argument("--es-patience", type=int, default=5)
    ap.add_argument("--t-sampler", choices=["logit_normal", "uniform", "cosmap"],
                    default="logit_normal")
    ap.add_argument("--t-std", type=float, default=1.0)
    ap.add_argument("--shift", type=float, default=1.0)
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--sampler", choices=["heun", "euler"], default="heun")
    ap.add_argument("--cfg", type=float, default=1.0, help="采样 CFG (1.0=关)")
    ap.add_argument("--cond-drop", type=float, default=0.1)
    ap.add_argument("--w-struct", type=float, default=0.5,
                    help="★ 结构 loss 权重 (对 x0_pred 的 Dice+BCE)。**必须有**:\n"
                         "  像素是稀疏二值(墨 5.5%), 纯 MSE 被 94.5% 平凡背景像素主导,\n"
                         "  模型只需要在背景上预测 v≈ε-1 就能把 loss 压到 0.14 —— 实测\n"
                         "  训练 loss 掉到 0.14 但采样结构仍等于噪声。主干同款在\n"
                         "  src/train/v11_struct-loss.json (w_latent_skel/canny)")
    ap.add_argument("--struct-min-t", type=float, default=0.4,
                    help="★ 结构 loss 只在 t>=此值生效。方向与主干相反, 原因**:\n"
                         "  x0_pred = x_t - t*v。t 很小时 x_t 已≈x0 -> 即使 v=0 也\n"
                         "  Dice≈0.9(实测 t=0.02: 0.891, t=0.2: 0.423), 是**免费假信号**,\n"
                         "  还与速度目标(要预测 eps-x0)互相打架。高噪声段才真正考结构:\n"
                         "  实测 v=0 的 Dice t=0.5: 0.195 / t=0.7: 0.130 / t=0.9: 0.093,\n"
                         "  而正确 v 在所有 t 上 Dice=0.9989 —— 处处可满足, 高 t 才有梯度。\n"
                         "  (主干用 max_t=0.3 是因为它走 probe, probe 在高噪声下不准; 我们\n"
                         "   直接在像素域算 x0_pred, 门控方向必须反过来)")
    ap.add_argument("--struct-max-t", type=float, default=1.0,
                    help="结构 loss 上界 (默认 1.0 不限)")
    ap.add_argument("--ink-w", type=float, default=0.0,
                    help="⚠ 墨像素在 MSE 里的额外权重。**实测有害**: ink_w=10 会把模型\n"
                         "  推向'到处画墨' (墨量 7.7~8.5x GT, precision 一路掉) —— 保持 0")
    ap.add_argument("--w-ink-ratio", type=float, default=0.3,
                    help="★ **单边**墨量约束: 只罚超过 ink-ratio-tol 倍的过墨")
    ap.add_argument("--ink-ratio-tol", type=float, default=1.5,
                    help="墨量容忍倍数 (默认 1.5 = 允许多画 50%% 以保住 recall)")
    ap.add_argument("--max-ink", type=float, default=0.5,
                    help="过滤原迹(64²)墨比高于此值的退化样本 (近似纯黑块, 骨架无意义)")
    ap.add_argument("--ema-decay", type=float, default=0.99)
    ap.add_argument("--resume", default="")
    ap.add_argument("--poster", type=int, default=0,
                    help="出采样对比 poster 的行数 (条件g | 采样 | 目标x0), 0=关")
    ap.add_argument("--poster-out", default="_ot_scratch/fm_poster.png")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--dump-width", type=int, default=3)
    ap.add_argument("--dump-tag", default="")
    ap.add_argument("--out", default="assets/skelnet_fm64.pt")
    ap.add_argument("--log", default="logs/skelnet_fm64.log")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    R = 64

    def log(m):
        print(m, flush=True)

    csmap = json.load(open(a.callig_map, encoding="utf-8"))
    from src.utils.callig_script_map import map_callig_script as _mcs
    st_ = th.load(a.style_emb, map_location="cpu", weights_only=False)
    style_tab = (st_["embedding"] if isinstance(st_, dict) else st_).float().to(dev)

    # ── 数据 (落盘 64² 数据集) ───────────────────────────────────────────
    def load_ds(split):
        p = os.path.join(a.dataset, f"{split}.npz")
        if not os.path.exists(p):
            raise SystemExit(f"[FATAL] 缺落盘数据集 {p}\n"
                             f"  先跑: python tools/build_skel64_dataset.py --res {R}")
        z = np.load(p)
        ids = [int(i) for i in z["ids"]]
        return ({i: k for k, i in enumerate(ids)}, z["std_skel"], z["gt_skel"],
                z["img_ink"])

    def rows_with_y(csvp):
        import csv
        out = []
        for r in csv.DictReader(open(csvp, encoding="utf-8")):
            try:
                i = int(r["img_id"])
            except Exception:                                  # noqa: BLE001
                m = re.search(r"(\d+)\.png$", r["image_path"])
                if not m:
                    continue
                i = int(m.group(1))
            out.append((i, int(_mcs(int(r["calligrapher_id"]),
                                    int(r["script_id"]), csmap))))
        return out

    tr_mp, tr_std, tr_gt, tr_img = load_ds("train")
    va_mp, va_std, va_gt, va_img = load_ds("val")
    _sc0 = 255.0 if tr_img.max() > 1.5 else 1.0
    # ★ 过滤退化样本: 原迹(64²)墨比过高 = 近似纯黑块, 其"骨架"无意义 (train 232/34749)
    _ok_tr = set(int(i) for i in np.array(list(tr_mp))[
        tr_img.reshape(len(tr_img), -1).mean(1) / _sc0 <= a.max_ink])
    _ok_va = set(int(i) for i in np.array(list(va_mp))[
        va_img.reshape(len(va_img), -1).mean(1) / _sc0 <= a.max_ink])
    tr_rows = [r for r in rows_with_y("assets/train_top10_style23_minusval.csv")
               if r[0] in tr_mp and r[0] in _ok_tr]
    va_rows = [r for r in rows_with_y("assets/val_skelnet.csv")
               if r[0] in va_mp and r[0] in _ok_va]
    log(f"[1] 退化样本过滤 (原迹墨比>{a.max_ink}): 训练 "
        f"{len(tr_mp)-len(_ok_tr)} 条, 验证 {len(va_mp)-len(_ok_va)} 条")
    TR_G = th.from_numpy(tr_std[[tr_mp[r[0]] for r in tr_rows]].astype(np.float32))
    TR_T = th.from_numpy(tr_gt[[tr_mp[r[0]] for r in tr_rows]].astype(np.float32))
    TR_Y = np.array([r[1] for r in tr_rows], np.int64)
    n_va = min(a.val_n, len(va_rows))
    va_ids = [r[0] for r in va_rows[:n_va]]
    VA_G = th.from_numpy(va_std[[va_mp[i] for i in va_ids]].astype(np.float32))
    VA_T = th.from_numpy(va_gt[[va_mp[i] for i in va_ids]].astype(np.float32))
    VA_Y = np.array([r[1] for r in va_rows[:n_va]], np.int64)
    _sc = 255.0 if float(TR_T.max()) > 1.5 else 1.0
    log(f"[1] 落盘数据集: 训练 {len(tr_rows)} 验证 {n_va} | 量纲 0/{int(_sc)} | "
        f"x0 墨比 {float(TR_T.mean())/_sc:.4f} | g 墨比 "
        f"{float(TR_G.mean())/_sc:.4f}   (应 ~0.055)")

    def to_x(v):
        """(N,64,64) 图 -> [-1,1], 墨=-1, 背景=+1。

        ★ 量纲自适应: 若出现 >1.5 的值就按 0/255 处理, 否则按 0/1。
          这样无论数据集存哪种口径都不会静默出错 (今天已因此废过一轮:
          把 0/1 当 0/255, 背景被当成墨)。
        """
        v = v.float()
        if float(v.max()) > 1.5:
            v = v / 255.0
        return 1.0 - 2.0 * v

    model = CondUNet(base=a.base, sdim=style_tab.shape[1]).to(dev)
    log(f"[2] CondUNet base={a.base} -> "
        f"{sum(p.numel() for p in model.parameters()):,} 参数 | flow: "
        f"t={a.t_sampler} sampler={a.sampler} shift={a.shift}")

    opt = th.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = th.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / max(a.warmup, 1), 1.0) *
        (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * min(1.0, s / max(a.steps, 1))))))
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def sample_t(n):
        if a.t_sampler == "logit_normal":
            return th.sigmoid(th.randn(n, device=dev) * a.t_std)
        if a.t_sampler == "cosmap":
            u = th.rand(n, device=dev).clamp_(1e-6, 1 - 1e-6)
            return 1.0 - 1.0 / (th.tan(u * math.pi / 2) + 1.0)
        return th.rand(n, device=dev)

    @th.no_grad()
    def heun(model_, g, s, steps, cfg):
        """主干同款: Heun 二阶 + shift 网格 + (可选)CFG。g 两支都给。"""
        b = g.shape[0]
        x = th.randn_like(g)
        ss = np.linspace(1.0, 0.0, steps + 1)
        if a.shift != 1.0:
            ss = a.shift * ss / (1.0 + (a.shift - 1.0) * ss)
        for k in range(steps):
            t_i, t_n = float(ss[k]), float(ss[k + 1])
            dt = t_n - t_i
            t1 = th.full((b,), t_i, device=dev)

            def vel(xx, tt):
                v = model_(xx, tt, g, s)
                if cfg and cfg != 1.0:
                    c_n = model_.cond(tt, model_.null_style.expand(b, -1))
                    v_c = model_(xx, tt, g, None, cvec=c_n)
                    v = v_c + cfg * (v - v_c)
                return v

            v1 = vel(x, t1)
            if a.sampler == "heun":
                t2 = th.full((b,), t_n, device=dev)
                v2 = vel(x + dt * v1, t2)
                x = x + dt * 0.5 * (v1 + v2)
            else:
                x = x + dt * v1
        return x

    def predict(g_u8, ys, use_ema=True):
        model.eval()
        ck = None
        if use_ema:
            ck = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict({k: v.to(dev) for k, v in ema.items()})
        out = []
        with th.no_grad():
            for s0 in range(0, g_u8.shape[0], 64):
                # ★ as_tensor: 训练路径传张量, dump 路径传 numpy, 统一处理
                gg = to_x(th.as_tensor(g_u8[s0:s0 + 64])).unsqueeze(1).to(dev)
                ss = style_tab[th.tensor(ys[s0:s0 + 64], device=dev)]
                out.append((heun(model, gg, ss, a.sample_steps, a.cfg) < 0
                            ).float()[:, 0].cpu().numpy())
        if ck is not None:
            model.load_state_dict(ck)
        model.train()
        return np.concatenate(out).astype(np.uint8)

    def eval_val():
        p = predict(VA_G, VA_Y)
        t = (VA_T.numpy() > 0.5).astype(np.uint8)      # ⚠ 0/1 数据, 不是 >127
        g = (VA_G.numpy() > 0.5).astype(np.uint8)
        from scipy.ndimage import binary_dilation
        st8 = np.ones((3, 3), bool)
        dice, prec, rec = [], [], []
        for i in range(p.shape[0]):
            pi, ti = p[i] > 0, t[i] > 0
            dice.append(2.0 * (pi & ti).sum() / max(int(pi.sum() + ti.sum()), 1))
            prec.append(float((pi & binary_dilation(ti, st8, 1)).sum())
                        / max(int(pi.sum()), 1))
            rec.append(float((ti & binary_dilation(pi, st8, 1)).sum())
                       / max(int(ti.sum()), 1))
        return {"dice64": float(np.mean(dice)), "prec64": float(np.mean(prec)),
                "rec64": float(np.mean(rec)),
                "lostInk": float(1.0 - np.mean(rec)),
                "inkRatio": float(p.mean() / max(t.mean(), 1e-9))}

    # ── poster: 采样式样 (条件 g | 采样 | 目标 x0) + 逐行墨量/Dice ───────────
    if a.poster > 0:
        if not a.resume:
            raise SystemExit("[FATAL] --poster 需要 --resume")
        rf = th.load(a.resume, map_location="cpu", weights_only=False)
        ema.update({k: v.float() for k, v in rf["ema"].items()})
        n = max(1, min(a.poster, n_va))
        idxs = np.linspace(0, n_va - 1, n).astype(int)
        Gsel, Tsel, Ysel = VA_G[idxs], VA_T[idxs], VA_Y[idxs]
        P = predict(Gsel, Ysel)                      # 0/1
        tt = (Tsel.numpy() > 127)
        gg = (Gsel.numpy() > 127)
        print(f"[poster] ckpt={a.resume} step={rf.get('step')} n={n} "
              f"cfg={a.cfg} steps={a.sample_steps}")
        print(f"  {'#':>3} {'id':>8} {'采样墨比':>9} {'目标墨比':>9} {'墨量倍':>7} "
              f"{'Dice':>7} {'rec(1px)':>9} {'prec(1px)':>9}")
        from scipy.ndimage import binary_dilation as _bd
        st8 = np.ones((3, 3), bool)
        r_g, r_t, r_m = [], [], []
        for r in range(n):
            pi, ti = P[r] > 0.5, tt[r]
            d = 2.0 * (pi & ti).sum() / max(int(pi.sum() + ti.sum()), 1)
            rec = float((ti & _bd(pi, st8, 1)).sum()) / max(int(ti.sum()), 1)
            pre = float((pi & _bd(ti, st8, 1)).sum()) / max(int(pi.sum()), 1)
            gmr = float(pi.mean()) / max(float(ti.mean()), 1e-9)
            r_g.append(gmr); r_t.append(d); r_m.append(pre)
            print(f"  {r:>3} {va_ids[idxs[r]]:>8} {pi.mean():9.4f} {ti.mean():9.4f} "
                  f"{gmr:7.2f} {d:7.4f} {rec:9.4f} {pre:9.4f}")
        print(f"  均值: 墨量倍 {np.mean(r_g):.2f}  Dice {np.mean(r_t):.4f}  "
              f"prec {np.mean(r_m):.4f}")
        from PIL import Image, ImageDraw, ImageFont
        cell = 130
        cv = Image.new("L", (3 * cell, n * (cell + 20)), 255)
        dr = ImageDraw.Draw(cv)
        try:
            fnt = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        except Exception:                                      # noqa: BLE001
            fnt = ImageFont.load_default()
        for c, nm in enumerate(["cond g", "采样", "目标 x0"]):
            dr.text((c * cell + 4, 3), nm, fill=0, font=fnt)
        for r in range(n):
            yy = r * (cell + 20) + 18
            for c, arr in enumerate([gg[r], P[r] > 0.5, tt[r]]):
                im = Image.fromarray(np.where(arr, 0, 255).astype(np.uint8))
                cv.paste(im.resize((cell, cell), Image.NEAREST), (c * cell, yy))
        os.makedirs(os.path.dirname(a.poster_out) or ".", exist_ok=True)
        cv.save(a.poster_out)
        print(f"[poster] -> {a.poster_out}")
        return

    # ── dump: 采样 -> 升采样4x -> 骨架化 -> 膨胀到 W px -> VAE encode ──────
    if a.dump:
        if not a.resume:
            raise SystemExit("[FATAL] --dump 需要 --resume")
        rf = th.load(a.resume, map_location="cpu", weights_only=False)
        ema.update({k: v.float() for k, v in rf["ema"].items()})
        log(f"[dump] 载入 {a.resume} (step={rf.get('step')}) width={a.dump_width}px "
            f"cfg={a.cfg}")
        for name in ("seen20", "strict84"):
            mp_, std_sk, _gt, _ink = load_ds(name)
            csvp = ("assets/eval_top10_seen_20.csv" if name == "seen20"
                    else "assets/eval_top10_strict_subset84.csv")
            ids = [r[0] for r in rows_with_y(csvp) if r[0] in mp_]
            ys = [r[1] for r in rows_with_y(csvp) if r[0] in mp_]
            outd = (f"data/top10_style23/predskel_fm_seen20"
                    if name == "seen20" else
                    f"data/top10_style23/predskel_fm_strict84")
            if a.dump_tag:
                outd += "_" + a.dump_tag
            from skimage.morphology import skeletonize as _sk
            from scipy.ndimage import binary_dilation as _bd
            from diffusers.models import AutoencoderKL
            p64 = predict(std_sk[[mp_[i] for i in ids]], ys)
            up = F.interpolate(th.from_numpy(p64).float()[:, None], size=(256, 256),
                               mode="nearest")[:, 0].numpy() > 0.5
            _it = max(0, (a.dump_width - 1) // 2)
            proc = np.empty_like(up)
            for i in range(up.shape[0]):
                sk = _sk(up[i])
                proc[i] = (_bd(sk, structure=np.ones((3, 3), bool),
                               iterations=_it) if _it > 0 else sk)
            vae = AutoencoderKL.from_pretrained(a.vae).to(dev).eval()
            img = th.from_numpy((1.0 - proc.astype(np.float32)) * 2 - 1)
            img = img.unsqueeze(1).repeat(1, 3, 1, 1)
            zl = []
            for s0 in range(0, img.shape[0], 8):
                with th.no_grad(), th.autocast("cuda", dtype=th.float16):
                    zl.append(vae.encode(img[s0:s0 + 8].to(dev)).latent_dist.mode())
                th.cuda.empty_cache()
            z = (th.cat(zl) * 0.18215).cpu().numpy()
            del vae
            th.cuda.empty_cache()
            os.makedirs(outd, exist_ok=True)
            np.savez_compressed(os.path.join(outd, "shard_00000.npz"),
                                latents=z.astype(np.float16),
                                img_ids=np.array(ids, dtype=np.int64))
            log(f"[dump] {name} -> {outd} ({len(ids)} 条) 预测墨比 {p64.mean():.4f}")
        return

    # ── 训练 ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)
    lf = open(a.log, "a", encoding="utf-8")

    def wlog(m):
        log(m)
        lf.write(m + "\n")
        lf.flush()

    best, bad, t0 = -1.0, 0, time.time()
    wlog(f"[3] 训练 {a.steps} 步 batch={a.batch} lr={a.lr} | flow "
         f"t={a.t_sampler} sampler={a.sampler} cfg={a.cfg}")
    for step in range(1, a.steps + 1):
        idx = np.random.randint(0, len(tr_rows), a.batch)
        x0 = to_x(TR_T[idx].to(dev)).unsqueeze(1)          # 目标骨架 (墨=-1)
        g = to_x(TR_G[idx].to(dev)).unsqueeze(1)
        y = th.from_numpy(TR_Y[idx]).to(dev)
        s = style_tab[y].clone()
        dr = th.rand(a.batch, device=dev) < a.cond_drop
        if dr.any():
            s[dr] = model.null_style
        t = sample_t(a.batch)
        eps = th.randn_like(x0)
        tc = t[:, None, None, None]
        xt = (1 - tc) * x0 + tc * eps
        v_tgt = eps - x0                                   # ★ 主干同款速度目标
        v = model(xt, t, g, s)
        if a.ink_w > 0:                                    # 墨像素加权 MSE
            w_ = 1.0 + a.ink_w * ((1.0 - x0) * 0.5)
            l_mse = (((v - v_tgt) ** 2) * w_).sum() / w_.sum()
        else:
            l_mse = F.mse_loss(v, v_tgt)
        # ★ 结构 loss: 只在低噪声步监督 x0_pred = x_t - t*v (主干 latent_struct_max_t=0.3)
        l_struct = th.zeros((), device=dev)
        l_inkr = th.zeros((), device=dev)
        if a.w_struct > 0 or a.w_ink_ratio > 0:
            sel = (t >= a.struct_min_t) & (t <= a.struct_max_t)
            if bool(sel.any()):
                tc_ = t[sel][:, None, None, None]
                x0p = xt[sel] - tc_ * v[sel]
                pin = ((1.0 - x0p) * 0.5).clamp(1e-4, 1 - 1e-4)   # 1=墨
                tin = (1.0 - x0[sel]) * 0.5
                l_struct = soft_dice_prob(pin, tin) + 0.5 * F.binary_cross_entropy(
                    pin, tin)
                ps = pin.sum(dim=(1, 2, 3))
                ts = tin.sum(dim=(1, 2, 3)).clamp_min(1.0)
                # ★ **单边**墨量约束: 只罚超过 1.5x 的过墨。
                #   双边(|Σp-Σt|/Σt) 会把模型逼到"不敢画", recall 直接崩
                #   (实测 fix3: recall 0.94 -> 0.30); 单边则允许适当多画以保住覆盖。
                l_inkr = (th.relu(ps - a.ink_ratio_tol * ts) / ts).mean()
        loss = l_mse + a.w_struct * l_struct + a.w_ink_ratio * l_inkr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with th.no_grad():
            for k, v_ in model.state_dict().items():
                if v_.dtype.is_floating_point:
                    ema[k].mul_(a.ema_decay).add_(v_.detach(), alpha=1 - a.ema_decay)
                else:
                    ema[k].copy_(v_)
        if step % 200 == 0:
            wlog(f"    step {step:6d}  mse {float(l_mse):.5f}  struct "
                 f"{float(l_struct):.4f}  inkr {float(l_inkr):.4f}  "
                 f"{int(time.time()-t0)}s")
        if a.eval_every and step % a.eval_every == 0:
            m = eval_val()
            wlog(f"    [eval] step {step} | " + "  ".join(
                f"{k} {v_:.4f}" for k, v_ in m.items()))
            score = m["dice64"]
            if score > best:
                best, bad = score, 0
                th.save({"ema": {k: v_.cpu() for k, v_ in ema.items()},
                         "step": step, "args": vars(a), "metrics": m}, a.out)
                wlog(f"    [es] ★ 新最佳 {score:.4f} @ {step}")
            else:
                bad += 1
                wlog(f"    [es] 未改善 {bad}/{a.es_patience}")
                if bad >= a.es_patience:
                    wlog(f"    [es] ★★ 早停 @ {step} (best {best:.4f})")
                    break
        if a.save_every and step % a.save_every == 0:
            th.save({"ema": {k: v_.cpu() for k, v_ in ema.items()},
                     "step": step, "args": vars(a)}, a.out + f".step{step:06d}")
    wlog(f"[4] DONE best={best:.4f}")


if __name__ == "__main__":
    main()
