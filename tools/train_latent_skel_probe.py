"""训练 LatentSkelProbe: 4ch 图像 latent -> 4ch 实例骨架 latent (冻结后用作结构 loss)。

用法:
    python tools/train_latent_skel_probe.py \
        --latent-shards-dir data/fame-kxl-tj-px60/shards_std_sym \
        --skel-shards-dir   data/aux/inst_skel_latents_px60 \
        --out assets/structure_probes/latent_skel_probe_v1/best.pt \
        --width 64 --depth 3 --epochs 6 --batch 256 --lr 3e-4

数据: 两侧都是预编码的 shard_*.npz, 含 'latents' (N,C,H,W) 与 'img_ids' (N,)。
按 img_id 取交集, 一对一回归。极小网络 + 纯 MSE, 单卡 4090 上几分钟。

⚠ 关键前提: **target 必须是实例骨架, 不能是标准字形骨架**。
   doc 54 实测: 喂 GT 实例骨架 strict 0.7326 vs 标准骨架 0.5680(+0.16)。
   我们要补的正是"这个书家写的这个字的具体形态" —— 用标准字形就是纯重复 g, 必败。

产物: {'model': state_dict, 'args': {...}, 'metrics': {...}}
由 --latent-skel-probe 加载; train.py 会按 args 重建结构并 strict=False 校验。
"""
import argparse, glob, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

from src.train.latent_structure import LatentSkelProbe


def index_shards(d):
    """-> {img_id: (shard_path, row_idx)}"""
    out = {}
    shards = sorted(glob.glob(os.path.join(d, "shard_*.npz")))
    if not shards:
        raise FileNotFoundError(f"No shard_*.npz in {d}")
    for sp in shards:
        z = np.load(sp)
        for j, iid in enumerate(z["img_ids"]):
            out[int(iid)] = (sp, j)
        z.close()
    return out


class PairSet(torch.utils.data.Dataset):
    """惰性按 (shard,row) 取; shard 常驻内存(总量小)。"""

    def __init__(self, img_idx, skel_idx, img_shape, skel_shape):
        self.pairs = sorted(set(img_idx) & set(skel_idx))
        if not self.pairs:
            raise RuntimeError("img shards 与 skel shards 的 img_id 交集为空 —— 数据源不匹配")
        self.img_idx, self.skel_idx = img_idx, skel_idx
        self.img_shape, self.skel_shape = img_shape, skel_shape
        self._cache = {}

    def _load(self, sp):
        if sp not in self._cache:
            z = np.load(sp)
            self._cache[sp] = z["latents"]
        return self._cache[sp]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        iid = self.pairs[i]
        sp, j = self.img_idx[iid]
        x = torch.from_numpy(np.asarray(self._load(sp)[j])).float()
        sp2, j2 = self.skel_idx[iid]
        y = torch.from_numpy(np.asarray(self._load(sp2)[j2])).float()
        return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-shards-dir", required=True, help="GT 图像 latent shards")
    ap.add_argument("--skel-shards-dir", required=True, help="**实例**骨架 latent shards")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", default="mse", choices=["mse", "l1"],
                    help="回归损失。**实测 l1 明显更好**: latent MSE 被背景主导, "
                         "对细笔画塌陷的梯度太弱 (见 tools/diag_skel_latent_mapping.py)。")
    ap.add_argument("--diag-vae", default="", help="给则训完用 VAE 解码对比墨量/IoU")
    args = ap.parse_args()

    _lossfn = F.l1_loss if args.loss == "l1" else F.mse_loss

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    img_idx, skel_idx = index_shards(args.latent_shards_dir), index_shards(args.skel_shards_dir)
    z0 = np.load(sorted(glob.glob(os.path.join(args.latent_shards_dir, "shard_*.npz")))[0])
    z1 = np.load(sorted(glob.glob(os.path.join(args.skel_shards_dir, "shard_*.npz")))[0])
    img_shape, skel_shape = tuple(z0["latents"].shape[1:]), tuple(z1["latents"].shape[1:])
    print(f"img latent {img_shape}  skel latent {skel_shape}")
    z0.close(); z1.close()
    if img_shape[0] != skel_shape[0]:
        raise ValueError(f"通道数不一致: img={img_shape[0]} skel={skel_shape[0]}")

    ds = PairSet(img_idx, skel_idx, img_shape, skel_shape)
    n = len(ds)
    n_val = max(1, int(n * args.val_frac))
    tr, va = torch.utils.data.random_split(ds, [n - n_val, n_val])
    print(f"pairs={n}  train={len(tr)} val={len(va)}")

    dl_tr = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=True,
                                        num_workers=args.num_workers, drop_last=True)
    dl_va = torch.utils.data.DataLoader(va, batch_size=args.batch, shuffle=False,
                                        num_workers=args.num_workers)

    m = LatentSkelProbe(in_channels=img_shape[0], out_channels=skel_shape[0],
                        width=args.width, depth=args.depth).to(dev)
    print(f"probe params: {sum(p.numel() for p in m.parameters()):,}")
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)

    # 参考基线: 直接预测 skel latent 的**训练集均值** (结构 loss 至少要显著优于它,
    # 否则 probe 只是学到了一个常数, 梯度信号退化)
    best = float("inf")
    for ep in range(args.epochs):
        m.train()
        t0, se = time.time(), 0.0
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = _lossfn(m(x), y)
            loss.backward(); opt.step()
            se += loss.item() * x.shape[0]
        m.eval()
        ve = 0.0
        with torch.no_grad():
            for x, y in dl_va:
                ve += _lossfn(m(x.to(dev)), y.to(dev)).item() * x.shape[0]
        tr_l, va_l = se / len(tr), ve / len(va)
        print(f"  ep{ep}  train={tr_l:.5f}  val={va_l:.5f}  ({time.time()-t0:.0f}s)")
        if va_l < best:
            best = va_l
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            torch.save({
                "model": m.state_dict(),
                "args": dict(in_channels=img_shape[0], out_channels=skel_shape[0],
                             width=args.width, depth=args.depth),
                "metrics": dict(val_mse=best, train_mse=tr_l, n_pairs=n, epochs=ep + 1),
            }, args.out)
            print(f"    -> saved {args.out} (val_mse={best:.5f})")

    # 常数基线
    m.eval()
    with torch.no_grad():
        ys = torch.cat([y for _, y in dl_va]).to(dev)
        const = _lossfn(ys.mean(dim=0, keepdim=True).expand_as(ys), ys).item()
    print(f"\n常数基线 val_{args.loss} = {const:.5f}   probe best = {best:.5f}   "
          f"比值 = {best/const:.3f}")
    if best > const * 0.5:
        print("⚠ probe 相比常数基线改进有限 —— 检查 skel_shards_dir 是否真指向实例骨架, "
              "或 img/skel 的 img_id 是否错位。")

    # ── 图像空间验收: probe(GT骨架latent) 解码出来还像不像骨架 ──
    if args.diag_vae:
        _vae_diag(m, dl_va, args.diag_vae, dev)


@torch.no_grad()
def _vae_diag(m, dl_va, vae_path, dev):
    """把 probe 输出解码成灰度图, 与 GT 骨架图比 IoU/墨量/连通分量。

    这是**唯一有意义**的验收口径: latent MSE 再低, 解码出来糊了也没用。
    """
    import numpy as np
    from diffusers.models import AutoencoderKL
    from scipy.ndimage import label, generate_binary_structure

    vae = AutoencoderKL.from_pretrained(vae_path, local_files_only=True).eval().to(dev)
    sc = float(getattr(vae.config, "scaling_factor", 0.18215))

    def dec(lat):
        im = vae.decode(lat.float() / sc).sample
        return ((im.clamp(-1, 1) + 1) / 2).mean(1)

    xb = next(iter(dl_va))
    x, y = xb[0].to(dev)[:32], xb[1].to(dev)[:32]
    gy, gp = dec(y), dec(m(x))
    iy, ip = (gy < 0.5), (gp < 0.5)
    st = generate_binary_structure(2, 2)
    iou = float((iy & ip).sum().float() / (iy | ip).sum().clamp_min(1).float())
    ncy = float(np.mean([label(a.cpu().numpy(), st)[1] for a in iy]))
    ncp = float(np.mean([label(a.cpu().numpy(), st)[1] for a in ip]))
    print("\n[diag-vae] probe 输出解码 vs GT 骨架图 (32 样本)")
    print(f"  墨量    probe {float(ip.float().mean()):.4f} / GT {float(iy.float().mean()):.4f} "
          f"(比 {float(ip.float().mean()) / max(float(iy.float().mean()), 1e-9):.3f})")
    print(f"  IoU(probe, GT) = {iou:.4f}")
    print(f"  连通分量 probe {ncp:.1f} / GT {ncy:.1f}")
    # 上限参考: VAE 自身往返保真度 (对真骨架 PNG 实测 IoU=0.956, 见 diag_skel_latent_mapping.py)
    # ⚠ 不要拿 dec(y) 跟 iy 比 —— 两者同源, 恒等于 1.0, 是个假上限。
    print("  [上限参考] VAE 自身 latent->img 往返 (对骨架 PNG) IoU = 0.956 "
          "(tools/diag_skel_latent_mapping.py)")




if __name__ == "__main__":
    main()
