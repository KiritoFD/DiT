#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_skelnet_pix.py — **64x64 骨架域**（1px 中心线）骨架形变网。

为什么换掉 DiT-latent 路线 (2026-09-30 诊断, 实测证据):
  · DiT-latent 输出的骨架解码墨量只有 GT 的 0.32~0.50x, 大量笔画**纯白丢失**
    —— 信息不在 latent 里, 推理期缩放/阈值造不出来;
  · 四个设计死结: (1) patch_size=2 把 32² latent 折成 16², 3px 骨架只占 0.375 格,
    patch 的 2x2 平均抹平细笔画, 线性 patch 输出头又出不了稀疏高频;
    (2) latent MSE 下墨只占 ~5% 权重 -> 丢笔画几乎不花 loss;
    (3) 没有 g 的全分辨率旁路; (4) VAE 对 mean latent 解码天然发灰。
  反证: oracle(GT 骨架 latent) 1.00x 墨量且干净 —— 因为它是从**干净像素骨架 encode**
        出来的。

本实现 (2026-09-30 采纳「降采样到 64 再取 1px 骨架」口径):
  · 分辨率 **64x64**。先把 256² 骨架用 **any-pool**(块均值>0) 降采样到 64²
    —— "这条线经过该格", 细线不会因亚像素而消失; 再在 64² 上 **skeletonize** 取
    **1px 中心线**。1px@64 回到 256² 约等于 4px 宽, 正好落在 3~7px 的目标档内。
  · 输入 = 标准骨架同样口径的 1px@64; 风格向量(书家x书体 128d) 走 FiLM
  · 输出 64x64 二值 logits;  **g 硬旁路** logit += λ*g -> g 的笔画不可能丢
  · 损失 = Dice + BCE: Dice 对"少一笔"极敏感, 正是 MSE 的盲区
  · 确定性单次前向; 无扩散; 无 VAE 参与训练
  · --dump: 最近邻升采样 x4 回 256 (≈4px) -> 骨架化 -> 膨胀到 `--dump-width`
    -> VAE encode -> predskel shards。**宽度是自由旋钮, 按下游效果扫**。

缓存: 一次解码就把 256² 二值骨架存成 packed 位图 (34.7k x 8KB = 285MB),
      之后换 prep / 分辨率都不需要再解码 (解码是唯一贵的步骤)。

用法:
  python tools/train_skelnet_pix.py --prep skel --res 64
  python tools/train_skelnet_pix.py --dump --resume assets/skelnet_pix64.pt --dump-width 5
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


def cldice_np(pred_b, gt_b, tol=3):
    """容差版 clDice (与 train_skelnet_dit.py 一致)"""
    from skimage.morphology import skeletonize
    from scipy.ndimage import binary_dilation
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return 0.0
    sp, sg = skeletonize(pred_b), skeletonize(gt_b)
    if sp.sum() == 0 or sg.sum() == 0:
        return 0.0
    st = np.ones((3, 3), bool)
    gt_t = binary_dilation(gt_b, structure=st, iterations=tol)
    pr_t = binary_dilation(pred_b, structure=st, iterations=tol)
    tp = float((sp & gt_t).sum()) / float(sp.sum())
    ts = float((sg & pr_t).sum()) / float(sg.sum())
    return 2.0 * tp * ts / (tp + ts) if (tp + ts) > 0 else 0.0


# ── 网络 ────────────────────────────────────────────────────────────────────
def cbr(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False),
                         nn.BatchNorm2d(o), nn.SiLU(inplace=True))


class FiLM(nn.Module):
    def __init__(self, sd, ch):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(sd, ch * 2), nn.SiLU(),
                                 nn.Linear(ch * 2, ch * 2))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, s):
        sc, sh = self.mlp(s).chunk(2, 1)
        return x * (1 + sc[:, :, None, None]) + sh[:, :, None, None]


class SkelUNet(nn.Module):
    def __init__(self, base=32, sd=128, in_ch=1):
        super().__init__()
        b = base
        self.e1, self.e2, self.e3, self.e4 = (
            cbr(in_ch, b), cbr(b, b * 2), cbr(b * 2, b * 4), cbr(b * 4, b * 8))
        self.f1, self.f2, self.f3, self.f4 = (FiLM(sd, b), FiLM(sd, b * 2),
                                              FiLM(sd, b * 4), FiLM(sd, b * 8))
        self.pool = nn.MaxPool2d(2)
        self.u3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.d3 = cbr(b * 8, b * 4)
        self.u2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.d2 = cbr(b * 4, b * 2)
        self.u1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.d1 = cbr(b * 2, b)
        self.out = nn.Conv2d(b, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, s):
        e1 = self.f1(self.e1(x), s)
        e2 = self.f2(self.e2(self.pool(e1)), s)
        e3 = self.f3(self.e3(self.pool(e2)), s)
        e4 = self.f4(self.e4(self.pool(e3)), s)
        d3 = self.d3(th.cat([self.u3(e4), e3], 1))
        d2 = self.d2(th.cat([self.u2(d3), e2], 1))
        d1 = self.d1(th.cat([self.u1(d2), e1], 1))
        return self.out(d1)


def soft_dice(logit, tgt, eps=1.0):
    p = th.sigmoid(logit)
    num = 2 * (p * tgt).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + tgt.sum(dim=(1, 2, 3)) + eps
    return 1 - (num / den).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/top10_style23/skel64",
                    help="★ 落盘数据集目录 (tools/build_skel64_dataset.py 产出):\n"
                         "  原始图/标准字图先降到 64 落盘, 再在 64 上提 1px 骨架")
    ap.add_argument("--prep", choices=["skel", "density"], default="skel",
                    help="skel: any-pool 降采样后取 1px 骨架 (默认, 细线不丢);\n"
                         "density: 块均值密度图 (保留粗细但会亚像素模糊)")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--train-csv", default="assets/train_top10_style23_minusval.csv")
    ap.add_argument("--val-csv", default="assets/val_skelnet.csv")
    ap.add_argument("--std-shards", default="data/top10_style23/shards_std")
    ap.add_argument("--gt-png-dir", default="data/top10_style23/gt_skel_png")
    ap.add_argument("--style-emb", default="assets/callig_script_emb_top10.pt")
    ap.add_argument("--callig-map", default="assets/callig_script_id_map_top10.json")
    ap.add_argument("--vae", default="data/pretrained/pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--cache-dir", default="_ot_scratch/pixcache")
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--val-n", type=int, default=64)
    ap.add_argument("--es-patience", type=int, default=8)
    ap.add_argument("--skip-lambda", type=float, default=1.0,
                    help="g 硬旁路: logit += λ*g。⚠ 别设大: λ=4 时 sigmoid'(4)≈0.018,\n"
                         "  恰好在有墨处把梯度压死, 网络学不动 (实测 eval 完全冻结)。\n"
                         "  λ=1: 有墨处 logit≥1 -> p=0.73, 阈值下 g 的笔画仍保住, 且梯度健康")
    ap.add_argument("--dice-w", type=float, default=1.0)
    ap.add_argument("--bce-w", type=float, default=1.0)
    ap.add_argument("--ema-decay", type=float, default=0.99,
                    help="⚠ 0.999 地平线 1000 步 -> 前几百步的 EMA 约等于初始化,\n"
                         "  评测会'看起来冻住'。0.99 地平线 100 步, 能及时反映进展")
    ap.add_argument("--resume", default="")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--dump-width", type=int, default=3,
                    help="★ 收尾膨胀宽度 px (自由旋钮): 1->3px, 2->5px, 3->7px")
    ap.add_argument("--dump-tag", default="")
    ap.add_argument("--out", default="assets/skelnet_pix64.pt")
    ap.add_argument("--log", default="logs/skelnet_pix64.log")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    R, K = a.res, 256 // a.res

    def log(m):
        print(m, flush=True)

    csmap = json.load(open(a.callig_map, encoding="utf-8"))
    from src.utils.callig_script_map import map_callig_script as _mcs
    os.makedirs(a.cache_dir, exist_ok=True)

    def rows_of(p):
        return list(csv.DictReader(open(p, encoding="utf-8")))

    def iid_of(r):
        if r.get("img_id"):
            return int(r["img_id"])
        m = re.search(r"(\d+)\.png$", r["image_path"])
        return int(m.group(1)) if m else None

    # ── 缓存: 256² 二值骨架 (packed) -> 任意 prep ─────────────────────────
    def _prep(packed, tag):
        cp = os.path.join(a.cache_dir, f"{a.prep}_{tag}_{R}.npy")
        if os.path.exists(cp) and np.load(cp, mmap_mode="r").shape[0] == packed.shape[0]:
            return np.load(cp)
        from skimage.morphology import skeletonize
        n = packed.shape[0]
        out = np.zeros((n, R, R), np.float32)
        for s0 in range(0, n, 512):
            b = np.unpackbits(packed[s0:s0 + 512], axis=1)
            b = b[:, :256 * 256].reshape(-1, 256, 256).astype(np.float32)
            dens = b.reshape(-1, R, K, R, K).mean(axis=(2, 4))     # 块均值
            if a.prep == "density":
                out[s0:s0 + dens.shape[0]] = dens
            else:
                for i in range(dens.shape[0]):                     # >0 = 线经过
                    out[s0 + i] = skeletonize(dens[i] > 0)         # -> 1px 中心线
        arr = ((out * 255).astype(np.uint8) if a.prep == "density"
               else (out > 0).astype(np.uint8) * 255)
        np.save(cp, arr)
        return arr

    def load_targets(ids, tag):
        cp = os.path.join(a.cache_dir, f"bin256_t_{tag}.npy")
        if not (os.path.exists(cp)
                and np.load(cp, mmap_mode="r").shape[0] == len(ids)):
            from PIL import Image

            def one(i):
                fp = os.path.join(a.gt_png_dir, f"{i:06d}.png")
                if not os.path.exists(fp):
                    return np.zeros(256 * 256 // 8, np.uint8)
                b = (np.asarray(Image.open(fp).convert("L")) < 128)
                return np.packbits(b.reshape(-1))
            with ThreadPoolExecutor(16) as ex:
                pk = np.stack(list(ex.map(one, ids)))
            np.save(cp, pk)
            log(f"    [cache] GT 骨架 256² 打包 {pk.shape} -> {cp}")
        return _prep(np.load(cp), tag)

    def load_inputs(ids, std_shards, tag):
        cp = os.path.join(a.cache_dir, f"bin256_g_{tag}.npy")
        ok = (os.path.exists(cp)
              and np.load(cp, mmap_mode="r").shape[0] == len(ids))
        if not ok:
            from diffusers.models import AutoencoderKL
            vae = AutoencoderKL.from_pretrained(a.vae).to(dev).eval()
            for p_ in vae.parameters():
                p_.requires_grad_(False)
            idx = {}
            for sp in sorted(glob.glob(os.path.join(std_shards, "shard_*.npz"))):
                with np.load(sp) as z:
                    for j, i in enumerate(z["img_ids"]):
                        idx[int(i)] = (sp, j)
            pk = np.zeros((len(ids), 256 * 256 // 8), np.uint8)
            by_f = {}
            for k, i in enumerate(ids):
                if i in idx:
                    by_f.setdefault(idx[i][0], []).append((k, idx[i][1]))
            done = 0
            for f, lst in by_f.items():
                with np.load(f) as z:
                    lat = z["latents"]
                # ⚠ 必须小块: 256² VAE 解码 fp32 每层激活 ~1GB, batch32 峰值 ~20GB
                for s0 in range(0, len(lst), 8):
                    ch = lst[s0:s0 + 8]
                    arr = np.stack([np.asarray(lat[j], dtype=np.float32)
                                    for _, j in ch])
                    t = th.from_numpy(arr).to(dev)
                    with th.no_grad(), th.autocast("cuda", dtype=th.float16):
                        d = (vae.decode((t / 0.18215).half()).sample.mean(1) < 0)
                    d = d.cpu().numpy()          # 保持 bool (packbits 只吃整型/布尔)
                    for k, (k0, _) in enumerate(ch):
                        pk[k0] = np.packbits(d[k].reshape(-1))
                    done += len(ch)
                    del t, d
                    th.cuda.empty_cache()
                    if done % 2048 < 8:
                        log(f"    [decode] {done}/{len(ids)}")
            del vae
            th.cuda.empty_cache()
            np.save(cp, pk)
            log(f"    [cache] std 骨架 256² 打包 {pk.shape} -> {cp}")
        return _prep(np.load(cp), tag)

    def build(csvp):
        out = []
        for r in rows_of(csvp):
            i = iid_of(r)
            if i is None:
                continue
            out.append((i, int(_mcs(int(r["calligrapher_id"]),
                                    int(r["script_id"]), csmap))))
        return out

    tr_rows, va_rows = build(a.train_csv), build(a.val_csv)
    log(f"[1] 配对: 训练 {len(tr_rows)} / 验证 {len(va_rows)}; prep={a.prep} res={R}²")
    if not tr_rows:
        raise SystemExit("[FATAL] 训练集为空")

    t0 = time.time()

    def load_ds(split):
        p = os.path.join(a.dataset, f"{split}.npz")
        if not os.path.exists(p):
            raise SystemExit(
                f"[FATAL] 缺落盘数据集 {p}\n"
                f"  先跑: python tools/build_skel64_dataset.py --res {R}")
        z = np.load(p)
        ids = [int(i) for i in z["ids"]]
        return ({i: k for k, i in enumerate(ids)}, z["std_skel"], z["gt_skel"])

    tr_mp, tr_std, tr_gt = load_ds("train")
    va_mp, va_std, va_gt = load_ds("val")
    # ★ 对齐: 只保留落在落盘数据集里的行 (npz 顺序与 csv 顺序无关)
    tr_rows = [r for r in tr_rows if r[0] in tr_mp]
    va_rows = [r for r in va_rows if r[0] in va_mp]
    if not tr_rows:
        raise SystemExit("[FATAL] 训练集与落盘数据集无交集")
    TR_G = tr_std[[tr_mp[r[0]] for r in tr_rows]] * 255       # uint8 0/255
    TR_T = tr_gt[[tr_mp[r[0]] for r in tr_rows]] * 255
    n_va = min(a.val_n, len(va_rows))
    va_ids = [r[0] for r in va_rows[:n_va]]
    VA_G = va_std[[va_mp[i] for i in va_ids]] * 255
    VA_T = va_gt[[va_mp[i] for i in va_ids]] * 255
    log(f"[1] 落盘数据集就绪 {int(time.time()-t0)}s | "
        f"g 墨比 {TR_G.mean()/255:.4f} / GT 墨比 {TR_T.mean()/255:.4f} | "
        f"训练 {len(tr_rows)} 验证 {n_va}")

    TR_Y = np.array([r[1] for r in tr_rows], np.int64)
    VA_Y = np.array([r[1] for r in va_rows[:n_va]], np.int64)
    st = th.load(a.style_emb, map_location="cpu", weights_only=False)
    style_tab = (st["embedding"] if isinstance(st, dict) else st).float().to(dev)

    model = SkelUNet(base=a.base, sd=style_tab.shape[1]).to(dev)
    log(f"[2] SkelUNet base={a.base} -> "
        f"{sum(p.numel() for p in model.parameters()):,} 参数; g 旁路 λ={a.skip_lambda}")

    opt = th.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = th.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / max(a.warmup, 1), 1.0) *
        (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(1.0, s / max(a.steps, 1))))))
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def fwd(g, y):
        lg = model(g, style_tab[y])
        return lg + a.skip_lambda * g if a.skip_lambda > 0 else lg

    def finish_to_latent(pred):
        """★ 与 GT 骨架同配方: 升采样回 256 -> 骨架化 -> 膨胀到 --dump-width -> encode"""
        from skimage.morphology import skeletonize
        from scipy.ndimage import binary_dilation
        from diffusers.models import AutoencoderKL
        if a.prep == "skel":        # 1px@64 -> 最近邻 x4 = 4px@256
            up = F.interpolate(th.from_numpy(pred)[:, None], size=(256, 256),
                               mode="nearest")[:, 0]
            b256 = (up > 0.5).numpy()
        else:
            up = F.interpolate(th.from_numpy(pred)[:, None], size=(256, 256),
                               mode="bilinear", align_corners=False)[:, 0]
            b256 = (up > 0.5).numpy()
        _it = max(0, (a.dump_width - 1) // 2)
        proc = np.empty_like(b256)
        for i in range(b256.shape[0]):
            sk = skeletonize(b256[i])
            proc[i] = (binary_dilation(sk, structure=np.ones((3, 3), bool),
                                       iterations=_it) if _it > 0 else sk)
        vae = AutoencoderKL.from_pretrained(a.vae).to(dev).eval()
        img = th.from_numpy((1.0 - proc.astype(np.float32)) * 2 - 1)
        img = img.unsqueeze(1).repeat(1, 3, 1, 1)
        zl = []
        for s0 in range(0, img.shape[0], 8):
            with th.no_grad(), th.autocast("cuda", dtype=th.float16):
                zl.append(vae.encode(img[s0:s0 + 8].to(dev)).latent_dist.mode())
            th.cuda.empty_cache()
        z = th.cat(zl) * 0.18215
        del vae
        th.cuda.empty_cache()
        return z.cpu().numpy()

    def predict(dens_u8, ys):
        model.eval()
        ck = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: v.to(dev) for k, v in ema.items()})
        outs = []
        with th.no_grad():
            for s0 in range(0, len(dens_u8), 32):
                g = th.from_numpy(dens_u8[s0:s0 + 32]).float().div_(255)[:, None].to(dev)
                y = th.tensor(ys[s0:s0 + 32], device=dev)
                outs.append(th.sigmoid(fwd(g, y))[:, 0].cpu().numpy())
        model.load_state_dict(ck)
        model.train()
        return np.concatenate(outs)

    # ── dump ─────────────────────────────────────────────────────────────
    if a.dump:
        if not a.resume:
            raise SystemExit("[FATAL] --dump 需要 --resume")
        rf = th.load(a.resume, map_location="cpu", weights_only=False)
        ema.update({k: v.float() for k, v in rf["ema"].items()})
        log(f"[dump] 载入 {a.resume} (step={rf.get('step')}) width={a.dump_width}px")
        SETS = {
            "seen20": ("assets/eval_top10_seen_20.csv",
                       "data/top10_style23/shards_std",
                       "data/top10_style23/predskel_pix_seen20"),
            "strict84": ("assets/eval_top10_strict_subset84.csv",
                         "data/50k_v2_glyph15k/shards_std",
                         "data/top10_style23/predskel_pix_strict84"),
        }
        for name, (csvp, stdd, outd) in SETS.items():
            rs = rows_of(csvp)
            ids = [iid_of(r) for r in rs]
            ys = [_mcs(int(r["calligrapher_id"]), int(r["script_id"]), csmap)
                  for r in rs]
            mp_, std_sk, _gt = load_ds(name)
            keep = [k for k, i in enumerate(ids) if i in mp_]
            ids = [ids[k] for k in keep]
            ys = [ys[k] for k in keep]
            G = std_sk[[mp_[i] for i in ids]] * 255
            pred = predict(G, ys)
            z = finish_to_latent(pred)
            if a.dump_tag:
                outd = outd + "_" + a.dump_tag
            os.makedirs(outd, exist_ok=True)
            np.savez_compressed(os.path.join(outd, "shard_00000.npz"),
                                latents=z.astype(np.float16),
                                img_ids=np.array(ids, dtype=np.int64))
            log(f"[dump] {name} -> {outd} ({len(ids)} 条) 预测墨比 {pred.mean():.4f}")
        return

    # ── 训练 ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)
    lf = open(a.log, "a", encoding="utf-8")

    def wlog(m):
        log(m)
        lf.write(m + "\n")
        lf.flush()

    def eval_val():
        from scipy.ndimage import binary_dilation
        pred = predict(VA_G, VA_Y)
        t64 = VA_T > 127
        g64 = VA_G > 127
        if a.prep == "skel":
            p64 = pred > 0.5
            st_ = np.ones((3, 3), bool)
            dice, prec, rec = [], [], []
            for i in range(p64.shape[0]):
                p, t = p64[i], t64[i]
                dice.append(2.0 * (p & t).sum() / max(int(p.sum() + t.sum()), 1))
                # 容忍 1px@64 ≈ 4px@256
                prec.append(float((p & binary_dilation(t, st_, 1)).sum())
                            / max(int(p.sum()), 1))
                rec.append(float((t & binary_dilation(p, st_, 1)).sum())
                           / max(int(t.sum()), 1))
            return {"dice64": float(np.mean(dice)), "prec64": float(np.mean(prec)),
                    "rec64": float(np.mean(rec)),
                    "lostInk": float(1.0 - np.mean(rec)),
                    "inkRatio": float(p64.mean() / max(t64.mean(), 1e-9))}
        p256 = (F.interpolate(th.from_numpy(pred)[:, None], size=(256, 256),
                              mode="bilinear", align_corners=False)[:, 0] > 0.5).numpy()
        t256 = (F.interpolate(th.from_numpy(VA_T.astype(np.float32) / 255)[:, None],
                              size=(256, 256), mode="bilinear",
                              align_corners=False)[:, 0] > 0.5).numpy()
        cls = [cldice_np(p256[i], t256[i]) for i in range(p256.shape[0])]
        lost = [float((t256[i] & ~p256[i]).sum()) / max(float(t256[i].sum()), 1)
                for i in range(p256.shape[0])]
        return {"clDice": float(np.mean(cls)), "lostInk": float(np.mean(lost)),
                "inkRatio": float(p256.mean() / max(t256.mean(), 1e-9))}

    best, bad, t0 = -1.0, 0, time.time()
    wlog(f"[3] 训练 {a.steps} 步 batch={a.batch} lr={a.lr} prep={a.prep} res={R}")
    for step in range(1, a.steps + 1):
        idx = np.random.randint(0, len(tr_rows), a.batch)
        g = th.from_numpy(TR_G[idx]).float().div_(255)[:, None].to(dev)
        t = th.from_numpy(TR_T[idx]).float().div_(255)[:, None].to(dev)
        y = th.from_numpy(TR_Y[idx]).to(dev)
        logit = fwd(g, y)
        l_dice = soft_dice(logit, t)
        l_bce = F.binary_cross_entropy_with_logits(logit, t)
        loss = a.dice_w * l_dice + a.bce_w * l_bce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with th.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(a.ema_decay).add_(v.detach(), alpha=1 - a.ema_decay)
                else:
                    ema[k].copy_(v)
        if step % 500 == 0:
            wlog(f"    step {step:6d}  dice {float(l_dice):.4f}  bce {float(l_bce):.4f}"
                 f"  {int(time.time()-t0)}s")
        if a.eval_every and step % a.eval_every == 0:
            m = eval_val()
            wlog(f"    [eval] step {step} | " + "  ".join(
                f"{k} {v:.4f}" for k, v in m.items()))
            if a.prep == "skel":
                score = m["dice64"]
            else:
                score = m["clDice"] * (1 - m["lostInk"])
            if score > best:
                best, bad = score, 0
                th.save({"ema": {k: v.cpu() for k, v in ema.items()},
                         "step": step, "args": vars(a), "metrics": m}, a.out)
                wlog(f"    [es] ★ 新最佳 {score:.4f} @ {step}")
            else:
                bad += 1
                wlog(f"    [es] 未改善 {bad}/{a.es_patience}")
                if bad >= a.es_patience:
                    wlog(f"    [es] ★★ 早停 @ {step} (best {best:.4f})")
                    break
        if a.save_every and step % a.save_every == 0:
            th.save({"ema": {k: v.cpu() for k, v in ema.items()},
                     "step": step, "args": vars(a)}, a.out + f".step{step:06d}")
    wlog(f"[4] DONE best={best:.4f}")


if __name__ == "__main__":
    main()
