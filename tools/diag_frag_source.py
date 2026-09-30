#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/diag_frag_source.py — 碎片化到底从哪来? 单变量定位。

核心手法: 拿**已经训好的好网络 v1**, 只翻一个开关, 看碎片率怎么变。
  - v1 @ max_off=3 (它训练时的值)  <- 基准
  - v1 @ max_off=6 (DiT 侧的值)    <- 隔离 max_off 的影响
  - v1 @ preserve_amp=0/1          <- 隔离保幅校准的影响
  - v4 @ max_off=6                 <- 我训的
  - v3 @ max_off=6                 <- 我训的(probe, 错位)
指标全部在**图空间**算, 且直接看 SkelNet 自己的输出 g' (不经过 DiT), 隔离变量。
"""
import os
import sys
import glob
import csv
import numpy as np
import torch

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8")
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_all(d):
    L, I = [], []
    for sp in sorted(glob.glob(os.path.join(d, 'shard_*.npz'))):
        with np.load(sp) as z:
            L.append(z['latents'].astype(np.float32))
            I.append(z['img_ids'])
    return np.concatenate(L), np.concatenate(I)


def main():
    from src.model.deform_skel import DeformSkel
    from diffusers.models import AutoencoderKL
    from scipy.ndimage import label, generate_binary_structure

    vae = AutoencoderKL.from_pretrained('data/pretrained/pretrained_models/sd-vae-ft-ema',
                                        local_files_only=True).eval().to(DEV)
    sc = 0.18215

    @torch.no_grad()
    def dec(z):
        # ⚠ 分块: 256 条一次性过 VAE 解码器 = 256x3x256x256 激活, 会吃 8G
        outs = []
        for i in range(0, z.shape[0], 32):
            im = vae.decode(z[i:i + 32].to(DEV).float() / sc).sample
            outs.append((((im.clamp(-1, 1) + 1) / 2).mean(1).float().cpu()))
        return torch.cat(outs, 0)

    G, gid = load_all('data/top10_style23/shards_std')
    T, _ = load_all('data/top10_style23/shards_aux_skel3')
    Y = torch.load('assets/callig_script_emb_top10.pt', map_location='cpu', weights_only=False)
    if isinstance(Y, dict):
        Y = Y.get('emb', Y.get('weight', list(Y.values())[0]))
    Y = Y.float()

    rows = list(csv.DictReader(open('assets/train_top10_style23.csv', encoding='utf-8')))
    slots = sorted({r['slot_name'] for r in rows})
    s2i = {s: i for i, s in enumerate(slots)}
    id2s = {}
    for r in rows:
        try:
            id2s[int(r['img_id'])] = s2i[r['slot_name']]
        except Exception:
            pass
    sid_all = np.array([id2s.get(int(i), 0) for i in gid])

    rng = np.random.RandomState(0)
    sel = rng.choice(len(G), 256, replace=False)
    Gs = torch.from_numpy(G[sel]).to(DEV)
    Ts = torch.from_numpy(T[sel]).to(DEV)
    Ys = Y.to(DEV)
    sid = torch.from_numpy(sid_all[sel]).to(DEV)

    ST = generate_binary_structure(2, 2)

    def skel3(gray):
        from skimage.morphology import skeletonize
        out = []
        for a in gray.cpu().numpy():
            k = skeletonize(a < 0.5)
            from scipy.ndimage import binary_dilation
            out.append(torch.from_numpy(binary_dilation(k, ST, iterations=1)))
        return torch.stack(out)

    with torch.no_grad():
        gt_gray = dec(Ts)
        gt_ink = (gt_gray < 0.5)
        gt_sk3 = skel3(gt_gray)

    def stats(g2):
        with torch.no_grad():
            g = dec(g2)
            ink = (g < 0.5)
            sk = skel3(g)
            # 连通分量 (8-连通) 与碎片率 = 分量数 / GT 分量数
            nc = np.mean([label(a.numpy(), ST)[1] for a in sk])
            nc_gt = np.mean([label(a.numpy(), ST)[1] for a in gt_sk3])
            iou = float((ink & gt_ink).sum().float() / (ink | gt_ink).sum().clamp_min(1).float())
            iou_sk = float((sk & gt_sk3).sum().float() / (sk | gt_sk3).sum().clamp_min(1).float())
        return dict(ink=float(ink.float().mean()), nc=float(nc), nc_gt=float(nc_gt),
                    frag=float(nc / max(nc_gt, 1e-9)), iou=iou, iou_sk=iou_sk)

    def build(ck_path, width, topo, style_tok):
        ck = torch.load(ck_path, map_location='cpu', weights_only=False)
        sd = ck.get('deform', ck) if isinstance(ck, dict) else ck
        return sd

    print(f"{'配置':<46}{'墨量':>8}{'分量':>7}{'frag':>7}{'IoU':>8}{'骨架IoU':>9}")
    print("  " + "-" * 82)

    def run(tag, sd, width, topo, style_tok, max_off, preserve_amp):
        m = DeformSkel(cond_dim=128, ch=4, grid=32, style_ch=32, width=width, max_off=max_off,
                       coarse=8, residual=0, res_cap=1.0, blur=0, dt_ch=1, affine=1, ckpt=0,
                       stroke_mod=1, stroke_cap=1.0, gate_radius=0.25, topo_mode=topo,
                       warp_iters=1, film_mode='film', style_tokens=style_tok, attn_heads=4,
                       preserve_amp=preserve_amp).to(DEV)
        cur = m.state_dict()
        keep = {k: v for k, v in sd.items()
                if k in cur and tuple(cur[k].shape) == tuple(v.shape)}
        miss = [k for k in cur if k not in keep]
        m.load_state_dict(keep, strict=False)
        m.eval()
        with torch.no_grad():
            g2 = m(Gs, Ys[sid])
        s = stats(g2)
        print(f"  {tag:<44}{s['ink']:>8.4f}{s['nc']:>7.1f}{s['frag']:>7.2f}"
              f"{s['iou']:>8.4f}{s['iou_sk']:>9.4f}"
              + (f"   [missing {len(miss)}]" if miss else ""))
        del m
        torch.cuda.empty_cache()

    # GT 参考
    with torch.no_grad():
        s = stats(Ts)
    print(f"  {'[参考] GT 自己 (上界)':<44}{s['ink']:>8.4f}{s['nc']:>7.1f}{s['frag']:>7.2f}"
          f"{s['iou']:>8.4f}{s['iou_sk']:>9.4f}")

    v1 = build('assets/deform_skel_top10_v1.pt', 96, 0, 0)
    v3 = build('assets/deform_skel_top10_v3_probe.pt', 128, 1, 0)
    v4 = build('assets/deform_skel_top10_v4_styleattn.pt', 128, 1, 16)

    # ★ 单变量: v1 只翻 max_off / preserve_amp
    run('v1 @ max_off=3.0 (它训练时的值)', v1, 96, 0, 0, 3.0, 0)
    run('v1 @ max_off=6.0 (DiT 侧的值)  ★只翻 max_off', v1, 96, 0, 0, 6.0, 0)
    run('v1 @ max_off=3.0 + preserve_amp=1', v1, 96, 0, 0, 3.0, 1)
    run('v1 @ max_off=6.0 + preserve_amp=1', v1, 96, 0, 0, 6.0, 1)
    print()
    run('v3 (probe, 训max_off=3, 无style_tok)', v3, 128, 1, 0, 6.0, 1)
    run('v4 (probe, 训max_off=6, 有style_tok)', v4, 128, 1, 16, 6.0, 1)


if __name__ == "__main__":
    main()
