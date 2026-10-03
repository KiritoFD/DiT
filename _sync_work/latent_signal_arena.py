"""隐空间信号竞技场: 6 个候选信号 × 2 个 VAE(SD-4ch / FLUX-16ch) × 2 个判据, 全程免训练。

为什么要做: 我们要在 DiT 的 endpoint(t∈[0.05,0.25]) 挂一个 loss, 让它"把书家推开"。
任何候选信号必须先在**离线沙盒**里过两关, 否则会往模型里灌毒梯度。

候选信号 (统一签名 fn(a, b) -> float, a/b 均为 (N,C,H,W) 的 latent, 越小越像):
  S1 raw_l1 / raw_l2     : 裸 latent 距离 (基线: 验证 VAE 本身是不是真的瞎)
  S2 lap_l1 / sobel_l1   : latent 空间高通(lap / sobel 幅值)后 L1 (专提隐空间里的边缘)
  S3 gram_fro            : 通道间协方差 Gram 的尺度无关 Frobenius 距离 (风格表征)
  S4 swd                 : 切片 Wasserstein (P 个随机 1D 投影 + 排序 + L1) —— 空间失明

判据1 排他性 margin (区分度): N 个三元组
  anchor = (字 c, 书家 A) 的一张真迹; pos = 同 (字 c, 书家 A) 的另一张(无则微扰);
  neg = (字 c, 书家 B) 的真迹. Margin = d(anchor,neg) - d(anchor,pos) > 0 才算"撕得开".
  ⚠ 各信号量纲差几个数量级 -> 只比**相对统计量**: frac>0 与 z = mean/std.

判据2 白盒梯度反演 (梯度健康度): x=randn(requires_grad) -> Adam 100 步最小化 d(x,target)
  -> decode -> 量化伪影: 8px 周期 Nyquist 块状能量比 + 与 target 解码图的 PSNR/SSIM/ink-IoU
  + 收敛比 final/init. 另存海报供目检。

输出: exp-std/signal_arena/{margin.csv, inversion.csv, poster_<vae>.png}
用法: python tools/latent_signal_arena.py --n-triplets 1000 --steps 100
"""
import argparse
import csv
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch as th
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--train-csv", default="exp-std/csv/train.csv")
ap.add_argument("--sd-shards", default="exp-std/data/shards_img")
ap.add_argument("--flux-ckpt", default="data/pretrained/flux_vae/ae.safetensors")
ap.add_argument("--flux-diff-dir", default="data/pretrained/flux_vae_diffusers",
                help="diffusers 格式 FLUX VAE 目录(优先)")
ap.add_argument("--flux-shards", default="",
                help="预先编好的 16ch latent shard 目录(有则直接读, 跳过 PNG 解码与编码, 快得多)")
ap.add_argument("--n-triplets", type=int, default=1000)
ap.add_argument("--n-unique", type=int, default=3000)
ap.add_argument("--swd-proj", type=int, default=32)
ap.add_argument("--steps", type=int, default=100)
ap.add_argument("--targets", type=int, default=1, help="判据2 反演用几个目标取均值")
ap.add_argument("--seeds", type=int, default=1, help="判据2 每个目标几个随机起点")
ap.add_argument("--enc-batch", type=int, default=32)
ap.add_argument("--n-pairs", type=int, default=4000, help="pairwise AUROC 的随机对数/类")
ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--out-dir", default="exp-std/signal_arena")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
dev = th.device("cuda" if th.cuda.is_available() else "cpu")
rng = random.Random(a.seed)
th.manual_seed(a.seed)

from diffusers import AutoencoderKL                                             # noqa: E402
from safetensors import safe_open                                               # noqa: E402
from src.eval.in_mem_eval import _get_vae                                       # noqa: E402

SD_SF = 0.18215
FLUX_SF, FLUX_SHIFT = 0.3611, 0.1159


# ══════════════════════════ 0. VAE ══════════════════════════
_FLUX_NDOWN = 4


def remap_flux_key(k):
    """sgm/CompVis 老命名 -> diffusers AutoencoderKL 命名。三个必踩的坑:
     ① resnet:      encoder.down.N.block.M  -> down_blocks.N.resnets.M
     ② attention:   mid.attn_1.{q,k,v,proj_out,norm}
                    -> mid_block.attentions.0.{to_q,to_k,to_v,to_out.0,group_norm}
     ③ decoder.up.N 的**顺序与 up_blocks 相反**(原始实现 reversed(up) 调用)
                    -> up_blocks.{NDOWN-1-N}
    """
    import re
    k = (k.replace("mid.attn_1.q.", "mid_block.attentions.0.to_q.")
          .replace("mid.attn_1.k.", "mid_block.attentions.0.to_k.")
          .replace("mid.attn_1.v.", "mid_block.attentions.0.to_v.")
          .replace("mid.attn_1.proj_out.", "mid_block.attentions.0.to_out.0.")
          .replace("mid.attn_1.norm.", "mid_block.attentions.0.group_norm.")
          .replace(".mid.block_1.", ".mid_block.resnets.0.")
          .replace(".mid.block_2.", ".mid_block.resnets.1."))
    m = re.match(r"(encoder|decoder)\.down\.(\d+)\.block\.(\d+)\.(.*)", k)
    if m:
        return f"{m.group(1)}.down_blocks.{m.group(2)}.resnets.{m.group(3)}.{m.group(4)}"
    m = re.match(r"(encoder|decoder)\.down\.(\d+)\.downsample\.(.*)", k)
    if m:
        return f"{m.group(1)}.down_blocks.{m.group(2)}.downsamplers.0.{m.group(3)}"
    m = re.match(r"decoder\.up\.(\d+)\.block\.(\d+)\.(.*)", k)
    if m:
        return (f"decoder.up_blocks.{_FLUX_NDOWN - 1 - int(m.group(1))}"
                f".resnets.{m.group(2)}.{m.group(3)}")
    m = re.match(r"decoder\.up\.(\d+)\.upsample\.(.*)", k)
    if m:
        return f"decoder.up_blocks.{_FLUX_NDOWN - 1 - int(m.group(1))}.upsamplers.0.{m.group(2)}"
    return k


def load_flux_vae(ckpt, diff_dir=""):
    """优先用 **diffusers 格式**目录(官方转换件, 键名直接对得上, 零映射风险);
    没有才回退到 sgm 原始件 + 手工重映射 + 恒等 quant_conv。"""
    global FLUX_SF, FLUX_SHIFT
    if diff_dir and os.path.isdir(diff_dir):
        import json as _json
        cfgp = os.path.join(diff_dir, "config.json")
        c = _json.load(open(cfgp, encoding="utf-8")) if os.path.exists(cfgp) else {}
        FLUX_SF = float(c.get("scaling_factor", FLUX_SF))
        FLUX_SHIFT = float(c.get("shift_factor", FLUX_SHIFT))
        print(f"[flux-ae] diffusers 目录 {diff_dir}\n"
              f"          latent_channels={c.get('latent_channels')} "
              f"scaling={FLUX_SF} shift={FLUX_SHIFT} "
              f"use_quant_conv={c.get('use_quant_conv')} "
              f"use_post_quant_conv={c.get('use_post_quant_conv')}")
        # ★ diffusers 0.27.2 的 AutoencoderKL 不认 use_quant_conv/use_post_quant_conv
        #   (Flux AE 是 false) -> from_pretrained 会报缺 key。手工建 + load(strict=False)
        #   + 把这两个 conv 灌成**恒等**(= 真实 Flux AE 的行为: 没有这两个 conv)。
        _ok = ("in_channels", "out_channels", "latent_channels", "down_block_types",
               "up_block_types", "block_out_channels", "layers_per_block",
               "norm_num_groups", "sample_size", "scaling_factor")
        kw = {k: v for k, v in c.items() if k in _ok}
        kw["down_block_types"] = tuple(kw.get("down_block_types", ("DownEncoderBlock2D",) * 4))
        kw["up_block_types"] = tuple(kw.get("up_block_types", ("UpDecoderBlock2D",) * 4))
        kw["block_out_channels"] = tuple(kw.get("block_out_channels", (128, 256, 512, 512)))
        vae = AutoencoderKL(**kw)
        sdf = {}
        with safe_open(os.path.join(diff_dir, "diffusion_pytorch_model.safetensors"),
                       framework="pt") as f:
            for k in f.keys():
                sdf[k] = f.get_tensor(k)
        miss, unexp = vae.load_state_dict(sdf, strict=False)
        _skip = ("quant_conv.weight", "quant_conv.bias",
                 "post_quant_conv.weight", "post_quant_conv.bias")
        print(f"          load: missing(除两 conv)={[k for k in miss if k not in _skip][:4]} "
              f"| unexpected={unexp[:4]}")
        with th.no_grad():
            for mod in (vae.quant_conv, vae.post_quant_conv):
                w = mod.weight
                w.zero_()
                for i in range(w.shape[0]):
                    w[i, i, 0, 0] = 1.0
                mod.bias.zero_()
        return vae.to(dev).eval()
    cfg = dict(in_channels=3, out_channels=3, latent_channels=16,
               down_block_types=("DownEncoderBlock2D",) * 4,
               up_block_types=("UpDecoderBlock2D",) * 4,
               block_out_channels=(128, 256, 512, 512), layers_per_block=2,
               norm_num_groups=32, sample_size=1024)
    vae = AutoencoderKL(**cfg)
    sd = {}
    with safe_open(ckpt, framework="pt") as f:
        for k in f.keys():
            sd[remap_flux_key(k)] = f.get_tensor(k)
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    _skip = ("quant_conv.weight", "quant_conv.bias",
             "post_quant_conv.weight", "post_quant_conv.bias")
    _mw = [k for k in missing if k not in _skip]
    print(f"[flux-ae] 重映射后: missing(除两个 conv)={len(_mw)} {_mw[:4]} | "
          f"unexpected={len(unexpected)} {unexpected[:4]}")
    if _mw or unexpected:
        print("  ⚠ 映射未完全对上 -> 下面的往返自检会暴露; 若 PSNR 崩, 优先怀疑 "
              "decoder.up 索引方向 / attention 子名")
    with th.no_grad():
        for mod in (vae.quant_conv, vae.post_quant_conv):
            w = mod.weight
            w.zero_()
            for i in range(w.shape[0]):
                w[i, i, 0, 0] = 1.0
            mod.bias.zero_()
    return vae.to(dev).eval()


class SDVAE:
    name, C, dec_tag = "SD-4ch", 4, "sd"

    def __init__(self):
        self.vae = _get_vae(dev, "data/pretrained/pretrained_models/sd-vae-ft-ema").eval()

    @th.no_grad()
    def encode(self, x):
        return self.vae.encode(x.to(dev) * 2 - 1).latent_dist.mode() * SD_SF

    @th.no_grad()
    def decode(self, z):
        with th.autocast("cuda", dtype=th.bfloat16):
            d = self.vae.decode((z / SD_SF).to(th.bfloat16)).sample
        return ((d.clamp(-1, 1) + 1) / 2).float()


class FluxVAE:
    name, C, dec_tag = "FLUX-16ch", 16, "flux"

    def __init__(self):
        self.vae = load_flux_vae(a.flux_ckpt, a.flux_diff_dir)

    @th.no_grad()
    def encode(self, x):
        return (self.vae.encode(x.to(dev) * 2 - 1).latent_dist.mode() - FLUX_SHIFT) * FLUX_SF

    @th.no_grad()
    def decode(self, z):
        with th.autocast("cuda", dtype=th.bfloat16):
            d = self.vae.decode((z / FLUX_SF + FLUX_SHIFT).to(th.bfloat16)).sample
        return ((d.clamp(-1, 1) + 1) / 2).float()


# ══════════════════════════ 1. 信号 ══════════════════════════
_K_LAP = th.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
_K_SX = th.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
_K_SY = th.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])


def _depthwise(x, k):
    c = x.shape[1]
    w = k.view(1, 1, 3, 3).repeat(c, 1, 1, 1).to(x.device, x.dtype)
    return F.conv2d(x, w, padding=1, groups=c)


def s_raw_l1(x, y):
    return (x - y).abs().mean()


def s_raw_l2(x, y):
    return ((x - y) ** 2).mean()


def s_lap_l1(x, y):
    return (_depthwise(x, _K_LAP) - _depthwise(y, _K_LAP)).abs().mean()


def s_sobel_l1(x, y):
    gx = (_depthwise(x, _K_SX) ** 2 + _depthwise(x, _K_SY) ** 2).sqrt()
    gy = (_depthwise(y, _K_SX) ** 2 + _depthwise(y, _K_SY) ** 2).sqrt()
    return (gx - gy).abs().mean()


def s_gram(x, y):
    def g(t):
        c = t.shape[1]
        f = t.reshape(t.shape[0], c, -1)
        G = f @ f.transpose(1, 2) / f.shape[2]
        return G / (G.norm(dim=(1, 2), keepdim=True) + 1e-8)
    return (g(x) - g(y)).norm(dim=(1, 2)).mean()


_DIRS = {}


def s_swd(x, y, p=None):
    """把每个空间位置当一个 C 维样本, 投影到 p 个随机方向后**排序**再比 L1 (切片 Wasserstein)。
    注意: 排序丢掉了空间位置对应关系 -> 这正是"空间失明"的来源, 本实验会量化它。"""
    p = p or a.swd_proj
    key = (x.shape[1], str(x.device))
    if key not in _DIRS:
        v = th.randn(p, x.shape[1], generator=th.Generator().manual_seed(7))
        _DIRS[key] = (v / v.norm(dim=1, keepdim=True)).to(x.device)
    D = _DIRS[key]                                            # (p, C)
    out = 0.0
    for i in range(x.shape[0]):
        pa = (D @ x[i].reshape(x.shape[1], -1)).sort(dim=1).values
        pb = (D @ y[i].reshape(y.shape[1], -1)).sort(dim=1).values
        out = out + (pa - pb).abs().mean()
    return out / x.shape[0]


SIGNALS = [("raw_l1", s_raw_l1), ("raw_l2", s_raw_l2), ("lap_l1", s_lap_l1),
           ("sobel_l1", s_sobel_l1), ("gram_fro", s_gram), ("swd", s_swd)]


# ══════════════════════════ 1b. 图像指标 (供自检与判据2) ══════════════════════════
def nyquist_ratio(img):
    """8px 周期块状伪影能量占比 (VAE 8x 上采样典型棋盘伪影)。img: (3,H,W) [0,1]"""
    g = img.mean(0).cpu().numpy()
    Fm = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))) ** 2
    H, W = g.shape
    yy, xx = np.mgrid[0:H, 0:W]
    fx = (xx - W / 2) / W
    fy = (yy - H / 2) / H
    band = ((np.abs(np.abs(fx) - 1 / 8) < 0.02) & (np.abs(np.abs(fy) - 1 / 8) < 0.02))
    return float(Fm[band].sum() / (Fm.sum() + 1e-12))


def ssim_g(p, q):
    p = p.mean(0).cpu().numpy(); q = q.mean(0).cpu().numpy()
    mp, mq = p.mean(), q.mean()
    vp, vq = p.var(), q.var()
    cov = ((p - mp) * (q - mq)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float(((2 * mp * mq + c1) * (2 * cov + c2)) /
                 ((mp ** 2 + mq ** 2 + c1) * (vp + vq + c2)))


def psnr_g(p, q):
    m = float(((p - q) ** 2).mean())
    return 99.0 if m <= 1e-9 else float(10 * np.log10(1.0 / m))


def ink_iou(p, q, thr=0.5):
    ip = p.mean(0).cpu().numpy() < thr
    iq = q.mean(0).cpu().numpy() < thr
    u = (ip | iq).sum()
    return float((ip & iq).sum() / u) if u else 1.0


# ══════════════════════════ 2. 数据 -> 三元组 ══════════════════════════
rows = list(csv.DictReader(open(a.train_csv, encoding="utf-8")))
grp = defaultdict(lambda: defaultdict(list))
for r in rows:
    grp[r["character"]][r["calligrapher"]].append(r)
usable = {c: d for c, d in grp.items() if len(d) >= 2}
print(f"[data] {len(rows)} 行; 有两个以上书家的字 = {len(usable)}/{len(grp)}")

tri = []
guard = 0
while len(tri) < a.n_triplets and guard < a.n_triplets * 20:
    guard += 1
    c = rng.choice(list(usable))
    if len(usable[c]) < 2:
        continue
    A, B = rng.sample(list(usable[c]), 2)
    ra = rng.choice(usable[c][A])
    rb = rng.choice(usable[c][B])
    pool = [r for r in usable[c][A] if r["img_id"] != ra["img_id"]]
    rp = rng.choice(pool) if pool else ra
    tri.append((c, A, B, ra, rp, rb, bool(pool)))
n_real = sum(t[6] for t in tri)
print(f"[tri] {len(tri)} 个三元组; pos 用真实第二张 = {n_real} ({n_real / max(len(tri), 1):.0%}), "
      f"其余用微扰")

need = {}
for _, _, _, ra, rp, rb, _ in tri:
    for r in (ra, rp, rb):
        need[r["img_id"]] = r["image_path"]
ids = list(need)[:a.n_unique]
print(f"[enc] 唯一图 {len(need)} 张 (编码前 {len(ids)})")


def load_pngs(id_list):
    out = []
    for i in id_list:
        p = need[i]
        im = Image.open(p if os.path.isabs(p) else os.path.join(ROOT, p)).convert("RGB")
        ar = np.asarray(im.resize((256, 256)), np.float32) / 255.0
        out.append(th.from_numpy(ar).permute(2, 0, 1))
    return out


def load_sd_latents(id_list, sdir=None):
    """从 shard 目录按 img_id 取 latent (4ch SD 与 16ch FLUX 通用)。
    ★ CSV 里的 img_id 是**字符串**, shard 里有的存 int 有的存 str('<U5') -> 统一转 int。
    (踩过: 直接 `i in idx` 会 0 命中, 而且不报错, 只是 LAT 空 -> 后面 KeyError。)"""
    import glob
    sdir = sdir or a.sd_shards
    idx, cache = {}, {}
    for sp in sorted(glob.glob(os.path.join(sdir, "shard_*.npz"))):
        d = np.load(sp)
        for j, x in enumerate(d["img_ids"].tolist()):
            idx[int(x)] = (sp, j)
    out, miss = {}, []
    for i in id_list:
        if int(i) in idx:
            sp, j = idx[int(i)]
            if sp not in cache:
                cache[sp] = np.load(sp)["latents"]
            # ★ key 统一成 int (今天已因 str/int 混用踩了 3 次: 0 命中且不报错)
            out[int(i)] = th.from_numpy(cache[sp][j].astype(np.float32))
        else:
            miss.append(i)
    if miss:
        print(f"[shard] ⚠ {sdir}: {len(miss)} 个 img_id 缺: {miss[:3]}")
    return out


t0 = time.time()
sd = SDVAE()
flux = FluxVAE()
LAT = {"SD-4ch": load_sd_latents(ids)}
pngs = None
if a.flux_shards and os.path.isdir(a.flux_shards):
    # ★ 有预先编好的 16ch shard(全数据集编码产物) 就直接读 -> 跳过 PNG 解码与编码, 快得多
    LAT["FLUX-16ch"] = load_sd_latents(ids, a.flux_shards)
    print(f"[lat] FLUX-16ch 直接来自 shards {a.flux_shards}")
else:
    pngs = load_pngs(ids)
    CACHE = os.path.join(a.out_dir, f"flux_latents_{len(ids)}.npz")
    fl = {}
    if os.path.exists(CACHE):
        _d = np.load(CACHE)
        fl = {int(k): th.from_numpy(_d[k].astype(np.float32)) for k in _d.files}
        print(f"[cache] 命中 {CACHE} ({len(fl)} 张)")
    else:
        for k in range(0, len(pngs), a.enc_batch):
            z = flux.encode(th.stack(pngs[k:k + a.enc_batch]))
            for t, i in enumerate(ids[k:k + a.enc_batch]):
                fl[int(i)] = z[t].cpu()
        np.savez_compressed(CACHE, **{str(k): v.numpy().astype(np.float16)
                                      for k, v in fl.items()})
        print(f"[cache] 写入 {CACHE}")
    LAT["FLUX-16ch"] = fl
for v in ("SD-4ch", "FLUX-16ch"):
    if LAT[v]:
        k0 = next(iter(LAT[v]))
        print(f"[lat] {v}: {len(LAT[v])} 张, C={LAT[v][k0].shape[0]}, "
              f"总耗时 {time.time() - t0:.0f}s")
# ★ 两 VAE 共同覆盖的 id (int key) —— 后面三元组/AUROC/护栏全用它
_both = {int(k) for k in set(LAT["SD-4ch"]) & set(LAT["FLUX-16ch"])}
print(f"[guard] 两 VAE 共同覆盖 {len(_both)} 张")

# ══════════════════════════ 3a. 打包上卡 + 向量化信号 ══════════════════════════
# 为什么: 逐三元组在 Python 里调 CPU 张量 = 80 万次往返 -> 十分钟起步(实测)。
#         把 latent 整体搬上 GPU, 一次算完所有三元组 -> 秒级。
PACK = {}
for _vn in ("SD-4ch", "FLUX-16ch"):
    _L = LAT[_vn]
    _ks = sorted(_L)
    PACK[_vn] = {"Z": th.stack([_L[k] for k in _ks]).to(dev).float(),   # (N,C,32,32) fp32
                 "i2r": {k: i for i, k in enumerate(_ks)}}
    _gz = PACK[_vn]["Z"].numel() * 4 / 1e9
    print(f"[pack] {_vn}: {tuple(PACK[_vn]['Z'].shape)}  {_gz:.2f} GB on {dev}")
_CHUNK = 2048


def _rows(vname, ids_):
    m = PACK[vname]["i2r"]
    return th.tensor([m[int(i)] for i in ids_], device=dev, dtype=th.long)


def _take(vname, ix):
    return PACK[vname]["Z"].index_select(0, ix)


def v_raw_l1(x, y):
    return (x - y).flatten(1).abs().mean(1)


def v_raw_l2(x, y):
    return (x - y).flatten(1).pow(2).mean(1)


def v_lap_l1(x, y):
    return (_depthwise(x, _K_LAP) - _depthwise(y, _K_LAP)).flatten(1).abs().mean(1)


def v_sobel_l1(x, y):
    gx = (_depthwise(x, _K_SX) ** 2 + _depthwise(x, _K_SY) ** 2).sqrt()
    gy = (_depthwise(y, _K_SX) ** 2 + _depthwise(y, _K_SY) ** 2).sqrt()
    return (gx - gy).flatten(1).abs().mean(1)


def _gram_n(x):
    f = x.flatten(2)                                     # (B,C,HW)
    G = f @ f.transpose(1, 2) / x.shape[2] / x.shape[3]  # (B,C,C)
    return G / (G.flatten(1).norm(dim=1).view(-1, 1, 1) + 1e-8)


def v_gram(x, y):
    return (_gram_n(x) - _gram_n(y)).flatten(1).norm(dim=1)


_SWD_D = {}


def v_swd(x, y):
    key = (x.shape[1], str(x.device))
    if key not in _SWD_D:
        v = th.randn(a.swd_proj, x.shape[1], generator=th.Generator().manual_seed(7))
        _SWD_D[key] = (v / v.norm(dim=1, keepdim=True)).to(x.device)
    D = _SWD_D[key]                                      # (p,C)
    px = th.einsum("pc,bchw->bphw", D, x).flatten(2).sort(dim=2).values
    py = th.einsum("pc,bchw->bphw", D, y).flatten(2).sort(dim=2).values
    return (px - py).abs().mean(dim=(1, 2))


VEC = [("raw_l1", v_raw_l1), ("raw_l2", v_raw_l2), ("lap_l1", v_lap_l1),
       ("sobel_l1", v_sobel_l1), ("gram_fro", v_gram), ("swd", v_swd)]


def batch_dist(vname, fn, i_ids, j_ids, chunk=None):
    """一次算完所有 (i,j) 对的距离向量 (GPU, 分块防爆显存)。"""
    ix, jx = _rows(vname, i_ids), _rows(vname, j_ids)
    ch = chunk or _CHUNK
    out = []
    with th.no_grad():
        for s in range(0, len(i_ids), ch):
            e = min(s + ch, len(i_ids))
            out.append(fn(_take(vname, ix[s:e]), _take(vname, jx[s:e])))
    return th.cat(out)

# ── 往返自检: 映射/装载错了立刻暴露 (PSNR 会崩到十几 dB) ──────────────
print("\n[自检] 往返保真度 (encode->decode 与原因比) —— ★ 只是旁观检查, 绝不允许带走整轮")
for vobj in (sd, flux):
    try:
        ps, ss = [], []
        chk = [i for i in ids[:6] if int(i) in LAT[vobj.name]]
        pngs_map = dict(zip(chk, load_pngs(chk)))     # 只读这 6 张
        for i in chk:
            x = pngs_map[i][None]
            rt = vobj.decode(vobj.encode(x))[0].cpu()    # ★ 解码在 CUDA, 原因在 CPU -> 必须对齐
            xg = x[0].cpu()
            ps.append(psnr_g(rt, xg)); ss.append(ssim_g(rt, xg))
        if ps:
            print(f"  {vobj.name:<10} PSNR={np.mean(ps):.2f} dB  SSIM={np.mean(ss):.4f} "
                  f"(n={len(ps)})")
    except Exception as _e:                                     # noqa: BLE001
        print(f"  [自检] {vobj.name} 失败(不影响主结果): {type(_e).__name__}: {_e}")

# ── 历史可比协议: pairwise AUROC (同书家 vs 异书家), 直接对得上文档里的 0.606 / 0.755 ──
id2callig = {int(r["img_id"]): r["calligrapher"] for r in rows}     # ★ int key
id2char = {int(r["img_id"]): r["character"] for r in rows}
# ★ 按字分组直接构造对, 不用拒绝采样(实测: 拒绝采样 160k 次只凑出 63 对 ✗ 白等)
by_char = defaultdict(lambda: defaultdict(list))
for _i in sorted(_both):
    if _i in id2char:
        by_char[id2char[_i]][id2callig[_i]].append(_i)
same_pairs, diff_pairs = [], []
_chars = [c for c, g in by_char.items() if len(g) >= 2]
rng.shuffle(_chars)
for _rd in range(500):
    if len(same_pairs) >= a.n_pairs and len(diff_pairs) >= a.n_pairs:
        break
    for c in _chars:
        g = by_char[c]
        ks = list(g)
        if len(diff_pairs) < a.n_pairs:                      # 同字异书家
            A, B = rng.sample(ks, 2)
            diff_pairs.append((rng.choice(g[A]), rng.choice(g[B])))
        if len(same_pairs) < a.n_pairs:                      # 同字同书家(不同图)
            multi = [k for k in ks if len(g[k]) >= 2]
            if multi:
                A = rng.choice(multi)
                _x, _y = rng.sample(g[A], 2)
                same_pairs.append((_x, _y))
        if len(same_pairs) >= a.n_pairs and len(diff_pairs) >= a.n_pairs:
            break
same_pairs, diff_pairs = same_pairs[:a.n_pairs], diff_pairs[:a.n_pairs]
print(f"[auroc] 同书家对 {len(same_pairs)} / 异书家对 {len(diff_pairs)} "
      f"(同字限定, 按字分组构造; 可用字 {len(_chars)})")

# ★ 护栏: 只保留**两个 VAE 都有 latent** 的 id, 否则中途 KeyError(且不报错地白跑)
_n0 = len(tri)
tri = [t for t in tri if all(int(t[i]["img_id"]) in _both for i in (3, 5))]
same_pairs = [(i, j) for i, j in same_pairs if i in _both and j in _both]
diff_pairs = [(i, j) for i, j in diff_pairs if i in _both and j in _both]
print(f"[guard] 三元组 {_n0} -> {len(tri)} | AUROC 对: 同 {len(same_pairs)} 异 {len(diff_pairs)}")
if len(tri) < 50:
    raise SystemExit("[guard] 可用三元组太少 -> 先确认 flux shards 是否编全")

_pg = th.Generator().manual_seed(99)


def perturb(z, shift=2, sigma=0.02):
    return th.roll(z, shifts=shift, dims=2) + sigma * th.randn(z.shape, generator=_pg)


# ══════════════════════════ 3. 判据1: margin ══════════════════════════
print("\n" + "=" * 84)
print(f"判据1 排他性 margin (n={len(tri)})   Margin = d(anchor,neg) - d(anchor,pos) > 0 才好")
print(f"{'vae':<10} {'signal':<10} {'frac>0(真)':>11} {'frac>0(微扰)':>12} {'z(真)':>8} "
      f"{'AUROC(同字同书家)':>17}")
print("-" * 84)
margin_rows = []
import scipy.stats as _st                                                      # noqa: E402
_t_a = [t[3]["img_id"] for t in tri]
_t_p = [t[4]["img_id"] for t in tri]
_t_n = [t[5]["img_id"] for t in tri]
_ok = th.tensor([bool(t[6]) for t in tri], device=dev)
print(f"[vec] 全部 {len(tri)} 个三元组 × {len(VEC)} 信号 一次性在 {dev} 上算 "
      f"(分块 {_CHUNK})")
_t_start = time.time()
for vname in ("SD-4ch", "FLUX-16ch"):
    for sname, fn in VEC:
        d_pos, d_neg = [], []
        for s in range(0, len(tri), _CHUNK):
            e = min(s + _CHUNK, len(tri))
            A = _take(vname, _rows(vname, _t_a[s:e]))
            Pp = _take(vname, _rows(vname, _t_p[s:e]))
            ok_c = _ok[s:e]
            if (~ok_c).any():                     # 无真实第二张 -> 用 anchor 平移+噪声
                Pp = Pp.clone()
                Ap = A[~ok_c]
                Pp[~ok_c] = th.roll(Ap, shifts=2, dims=2) + 0.02 * th.randn_like(Ap)
            d_pos.append(fn(A, Pp))
            d_neg.append(fn(A, _take(vname, _rows(vname, _t_n[s:e]))))
        d_pos, d_neg = th.cat(d_pos), th.cat(d_neg)
        m = d_neg - d_pos
        m_r, m_p = m[_ok], m[~_ok]
        fr_r = float((m_r > 0).float().mean()) if m_r.numel() else float("nan")
        fr_p = float((m_p > 0).float().mean()) if m_p.numel() else float("nan")
        z_r = float(m_r.mean() / (m_r.std() + 1e-12)) if m_r.numel() else float("nan")
        ds = batch_dist(vname, fn, [p[0] for p in same_pairs], [p[1] for p in same_pairs])
        dd = batch_dist(vname, fn, [p[0] for p in diff_pairs], [p[1] for p in diff_pairs])
        u = _st.mannwhitneyu(dd.cpu().numpy(), ds.cpu().numpy(),
                             alternative="greater").statistic
        auroc = float(u / (len(dd) * len(ds)))
        margin_rows.append((vname, sname, float(m_r.mean()), float(m_r.std()),
                            fr_r, fr_p, z_r, auroc))
        print(f"{vname:<10} {sname:<10} {fr_r:>11.3f} {fr_p:>12.3f} {z_r:>8.3f} "
              f"{auroc:>17.4f}", flush=True)
print(f"[vec] 判据1 完成, 用时 {time.time() - _t_start:.1f}s")
with open(os.path.join(a.out_dir, "margin.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["vae", "signal", "margin_mean_real", "margin_std_real",
                "frac_pos_real", "frac_pos_perturbed", "z_real", "auroc_same_callig"])
    w.writerows(margin_rows)
print("注: frac>0(真)=pos 用真实第二张(硬); (微扰) 那一列必然接近 1.0, 只作参考;")
print("    AUROC=同字前提下'同书家对更近'的概率, 直接对齐文档里的 0.606(latent 2.9M)/0.755(DINO 头)")


# ══════════════════════════ 4. 判据2: 梯度反演 ══════════════════════════
def nyquist_ratio(img):
    """8px 周期块状伪影能量占比 (VAE 8x 上采样典型棋盘伪影)。img: (3,H,W) [0,1]"""
    g = img.mean(0).cpu().numpy()
    Fm = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))) ** 2
    H, W = g.shape
    yy, xx = np.mgrid[0:H, 0:W]
    fx = (xx - W / 2) / W
    fy = (yy - H / 2) / H
    band = ((np.abs(np.abs(fx) - 1 / 8) < 0.02) & (np.abs(np.abs(fy) - 1 / 8) < 0.02))
    return float(Fm[band].sum() / (Fm.sum() + 1e-12))


def ssim_g(p, q):
    p = p.mean(0).cpu().numpy(); q = q.mean(0).cpu().numpy()
    mp, mq = p.mean(), q.mean()
    vp, vq = p.var(), q.var()
    cov = ((p - mp) * (q - mq)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float(((2 * mp * mq + c1) * (2 * cov + c2)) /
                 ((mp ** 2 + mq ** 2 + c1) * (vp + vq + c2)))


def psnr_g(p, q):
    m = float(((p - q) ** 2).mean())
    return 99.0 if m <= 1e-9 else float(10 * np.log10(1.0 / m))


def ink_iou(p, q, thr=0.5):
    ip = p.mean(0).cpu().numpy() < thr
    iq = q.mean(0).cpu().numpy() < thr
    u = (ip | iq).sum()
    return float((ip & iq).sum() / u) if u else 1.0


print("\n" + "=" * 84)
print(f"判据2 梯度反演 (Adam {a.steps} 步, 从纯噪声恢复 target latent)")
_TH_ = None  # (指标函数已提前定义, 见 1b)
hdr = (f"{'vae':<10} {'signal':<10} {'PSNR':>7} {'SSIM':>7} {'inkIoU':>7} "
       f"{'伪影比':>8} {'收敛比':>8}")
print(hdr); print("-" * len(hdr))
inv_rows, panels = [], {}
tids = []
for _, _, _, ra, rp, rb, _ in tri:
    for cand in (int(ra["img_id"]), int(rp["img_id"]), int(rb["img_id"])):
        if cand in LAT["SD-4ch"] and cand in LAT["FLUX-16ch"] and cand not in tids:
            tids.append(cand)
    if len(tids) >= a.targets:
        break
tids = tids[:a.targets]
print(f"[inv] 目标 {len(tids)} 个 × {a.seeds} seed × {len(SIGNALS)} 信号 × 2 VAE "
      f"= {len(tids) * a.seeds * len(SIGNALS) * 2} 次反演")
for vobj in (sd, flux):
    L = LAT[vobj.name]
    rows_p = []
    for t in tids:
        tgt = L[t][None].to(dev)          # ★ shard 读出来在 CPU, VAE 在 CUDA -> 必须先上卡
        tgt_img = vobj.decode(tgt)[0].cpu()
        row = [("target", tgt_img)]
        for sname, fn in SIGNALS:
            best = None
            for s in range(a.seeds):
                g = th.Generator().manual_seed(1000 + s)
                x = (th.randn(tgt.shape, generator=g).to(tgt.device)
                     * (tgt.std() + 1e-6)).requires_grad_(True)
                opt = th.optim.Adam([x], lr=a.lr)
                d0 = float(fn(x.detach(), tgt))
                for _ in range(a.steps):
                    opt.zero_grad()
                    (fn(x, tgt) / (d0 + 1e-12)).backward()   # ★ 按初值归一 -> 同一 lr 可比
                    opt.step()
                with th.no_grad():
                    d1 = float(fn(x.detach(), tgt))
                    img = vobj.decode(x.detach())[0].cpu()   # ★ 与 target 同设备(CPU), 便于算指标
                rec = (vobj.name, sname, t, psnr_g(img, tgt_img), ssim_g(img, tgt_img),
                       ink_iou(img, tgt_img), nyquist_ratio(img), d1 / (d0 + 1e-12))
                inv_rows.append(rec)
                if best is None or rec[4] > best[1][4]:      # 海报留 SSIM 最好的那张
                    best = (img, rec)
            row.append((sname, best[0]))
        rows_p.append(row)
    panels[vobj.name] = rows_p

agg = {}
for r in inv_rows:
    agg.setdefault((r[0], r[1]), []).append(r)
print(f"\n{'vae':<10} {'signal':<10} {'PSNR':>7} {'SSIM':>7} {'inkIoU':>7} "
      f"{'伪影比':>8} {'收敛比':>8} {'n':>4}")
print("-" * 66)
agg_rows = []
for (vn, sn), rs in sorted(agg.items()):
    m = [float(np.mean([x[i] for x in rs])) for i in range(3, 8)]
    agg_rows.append((vn, sn, *m, len(rs)))
    print(f"{vn:<10} {sn:<10} {m[0]:>7.2f} {m[1]:>7.4f} {m[2]:>7.3f} {m[3]:>8.4f} "
          f"{m[4]:>8.3f} {len(rs):>4}")
with open(os.path.join(a.out_dir, "inversion.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["vae", "signal", "target_id", "psnr_vs_target", "ssim_vs_target",
                "ink_iou", "nyquist_artifact", "final_over_init"])
    w.writerows(inv_rows)

# ★ summary.txt: 两张榜 + 名次 (结果不依赖 stdout, 后台跑也拿得到)
with open(os.path.join(a.out_dir, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("=== 判据1 margin (AUROC=同字下同书家更近的概率, 越大越好) ===\n")
    f.write("vae,signal,frac_pos_real,z_real,auroc_same_callig\n")
    for r in margin_rows:
        f.write(f"{r[0]},{r[1]},{r[4]:.4f},{r[6]:.4f},{r[7]:.4f}\n")
    f.write("\n=== 判据2 反演 (PSNR/SSIM 越大越好; 伪影比/收敛比越小越好) ===\n")
    f.write("vae,signal,psnr,ssim,ink_iou,nyquist,final_over_init,n\n")
    for r in agg_rows:
        f.write(f"{r[0]},{r[1]},{r[2]:.2f},{r[3]:.4f},{r[4]:.3f},{r[5]:.4f},{r[6]:.3f},{r[7]}\n")
    f.write("\n=== 名次 ===\n")
    f.write("判据1 by AUROC: " + ", ".join(
        f"{r[1]}@{r[0]}={r[7]:.3f}" for r in sorted(margin_rows, key=lambda x: -x[7])) + "\n")
    f.write("判据2 by SSIM : " + ", ".join(
        f"{r[1]}@{r[0]}={r[3]:.3f}" for r in sorted(agg_rows, key=lambda x: -x[3])) + "\n")
print(f"[summary] {os.path.join(a.out_dir, 'summary.txt')}")

S = 180
for vname, rows_p in panels.items():
    nrow, ncol = len(rows_p), len(rows_p[0])
    canvas = Image.new("RGB", (S * ncol, S * nrow + 24), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("_fonts/msyh.ttc", 16)
    sig_lab = ", ".join(s[0] for s in SIGNALS)
    d.text((4, 2), f"{vname} 反演: 行=不同目标, 列=[target, {sig_lab}]",
           font=font, fill=(0, 0, 0))
    for ri, row in enumerate(rows_p):
        for ci, (lab, img) in enumerate(row):
            ar = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            canvas.paste(Image.fromarray(ar).resize((S, S)), (ci * S, 24 + ri * S))
            d.text((ci * S + 4, 26 + ri * S), lab[:12], font=font, fill=(200, 0, 0))
    p = os.path.join(a.out_dir, f"poster_{vname.replace('.', '_')}.png")
    canvas.save(p)
    print(f"[poster] {p}")

print("\n" + "=" * 84)
print("判读: 合格 = 判据1 AUROC 明显>0.5(最好>0.6) 且 frac>0(真)>0.6; "
      "判据2 PSNR/SSIM 高、伪影比低、收敛比<0.1。")
