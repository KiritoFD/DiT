"""公平比较两个 latent 空间: 固定距离 vs **学出来的尺子**(白化/线性探针) + 像素基线。

为什么要有这个: 之前的竞技场用的是固定无监督距离(raw L1/L2), 它被高方差通道主导,
测的是"这把固定尺子能不能分开", 而不是"空间里有没有信息"。FLUX-16ch 各通道方差差异大,
raw L2 基本听不见信息所在的方向 -> 会系统性低估它。

三个空间, 同一批 id:
  SD-4ch   : exp-std/data/shards_img            (4x32x32 = 1024 维)
  FLUX-16ch: exp-std/data/shards_img_flux16     (16x32x32 = 4096 维)
  pixel-64 : 真迹图灰度 64x64 (降采样)          (4096 维) —— 无 VAE 的参照

三把尺子:
  A. raw L2                (原样欧氏距离)
  B. z-L2                  (每维 z-score 后欧氏 —— 去掉通道幅度差)
  C. PCA 白化 L2           (降到 128 维再白化 —— 去掉方向冗余/相关)
  每把尺子都报: 同字异书家的 pairwise AUROC (4000 对/类)
  D. 学出来的线性探针      (logistic regression, **按字划分** 8:2 -> 只在训练字上拟合,
     在"没见过的字"上测 23 类槽位准确率; 随机 = 1/23 = 4.3%) —— 直接量"信息量", 绕过尺子选择
"""
import argparse
import csv
import glob
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch as th

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="exp-std/csv/train.csv")
ap.add_argument("--sd-shards", default="exp-std/data/shards_img")
ap.add_argument("--flux-shards", default="exp-std/data/shards_img_flux16")
ap.add_argument("--max-ids", type=int, default=26002)
ap.add_argument("--pixel-cap", type=int, default=8000)
ap.add_argument("--n-pairs", type=int, default=4000)
ap.add_argument("--pca-dim", type=int, default=128)
ap.add_argument("--probe-steps", type=int, default=400)
ap.add_argument("--out", default="exp-std/signal_arena_full/vae_space_probe.csv")
a = ap.parse_args()
dev = th.device("cuda" if th.cuda.is_available() else "cpu")
th.manual_seed(0)
np.random.seed(0)


def load_shard_map(sdir):
    idx, cache = {}, {}
    for sp in sorted(glob.glob(os.path.join(sdir, "shard_*.npz"))):
        d = np.load(sp)
        for j, x in enumerate(d["img_ids"].tolist()):
            idx[int(x)] = (sp, j)
    return idx, cache


def gather(sdir, ids):
    idx, cache = load_shard_map(sdir)
    out, miss = [], []
    for i in ids:
        if int(i) in idx:
            sp, j = idx[int(i)]
            if sp not in cache:
                cache[sp] = np.load(sp)["latents"]
            out.append(th.from_numpy(cache[sp][j].astype(np.float32)))
        else:
            miss.append(i)
    return out, miss


rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
id2pair = {int(r["img_id"]): int(r["pair_id"] or 0) for r in rows}
id2char = {int(r["img_id"]): r["character"] for r in rows}
ids = [int(r["img_id"]) for r in rows][:a.max_ids]
print(f"[in] csv 行 {len(rows)}, 取 {len(ids)} 个 id")

t0 = time.time()
sd_l, miss_sd = gather(a.sd_shards, ids)
fl_l, miss_fl = gather(a.flux_shards, ids)
keep = [i for i, (x, y) in enumerate(zip(sd_l, fl_l))]
print(f"[load] SD {len(sd_l)}(缺 {len(miss_sd)}) / FLUX {len(fl_l)}(缺 {len(miss_fl)})  {time.time() - t0:.0f}s")
if miss_sd or miss_fl:
    raise SystemExit("两个 shard 覆盖不一致, 先补齐")

X = {
    "SD-4ch": th.stack(sd_l).flatten(1),
    "FLUX-16ch": th.stack(fl_l).flatten(1),
}
del sd_l, fl_l
th.cuda.empty_cache()


def load_pixels(id_list):
    from PIL import Image
    out = []
    for i in id_list[:a.pixel_cap]:
        p = f"data/top10_style23/imgs/{i:06d}.png"
        if not os.path.exists(p):
            p = f"data/top10_style23/imgs/{i}.png"
        im = Image.open(p).convert("L").resize((64, 64))
        arr = np.asarray(im, np.float32) / 255.0
        out.append(th.from_numpy(1.0 - arr).flatten())      # 用"墨量"而不是亮度
    return th.stack(out)


t0 = time.time()
Xp = load_pixels(ids)
X["pixel-64"] = Xp
ids_p = ids[:Xp.shape[0]]
print(f"[pixel] {tuple(Xp.shape)}  {time.time() - t0:.0f}s")

# ── 成对指标 ─────────────────────────────────────────────────────────
tri_chars = defaultdict(lambda: defaultdict(list))         # char -> pair -> [id]
for i in ids:
    if i in id2char:
        tri_chars[id2char[i]][id2pair[i]].append(i)
chars_pairs = [c for c, g in tri_chars.items() if len(g) >= 2]
rng = np.random.RandomState(0)
rng.shuffle(chars_pairs)
same, diff = [], []
for _ in range(500):
    if len(same) >= a.n_pairs and len(diff) >= a.n_pairs:
        break
    for c in chars_pairs:
        g = tri_chars[c]
        ks = list(g)
        if len(diff) < a.n_pairs:
            u, v = rng.choice(ks, 2, replace=False)
            diff.append((rng.choice(g[u]), rng.choice(g[v])))
        if len(same) < a.n_pairs:
            multi = [k for k in ks if len(g[k]) >= 2]
            if multi:
                k = multi[rng.randint(len(multi))]
                x, y = rng.choice(g[k], 2, replace=False)
                same.append((x, y))
        if len(same) >= a.n_pairs and len(diff) >= a.n_pairs:
            break
same, diff = same[:a.n_pairs], diff[:a.n_pairs]
print(f"[pairs] 同 {len(same)} 异 {len(diff)} (同字限定)")

import scipy.stats as st


def auroc_from(space, d_same, d_diff):
    u = st.mannwhitneyu(d_diff.cpu().numpy(), d_same.cpu().numpy(),
                        alternative="greater").statistic
    return float(u / (len(d_diff) * len(d_same)))


def pair_dists(Z, sel):
    ix = th.tensor([sel[k][0] for k in range(len(sel))], device=Z.device)
    jx = th.tensor([sel[k][1] for k in range(len(sel))], device=Z.device)
    out = []
    for s in range(0, len(sel), 4096):
        p = Z.index_select(0, ix[s:s + 4096])
        q = Z.index_select(0, jx[s:s + 4096])
        out.append((p - q).pow(2).sum(1))
    return th.cat(out)


print("\n" + "=" * 78)
print("A/B/C. 尺子对比: 同字异书家 pairwise AUROC (越大越好, 0.5=瞎)")
print(f"{'空间':<12} {'raw L2':>9} {'z-L2':>9} {'PCA白化L2':>11} {'维数':>7}")
print("-" * 78)
pos_of = {i: k for k, i in enumerate(ids)}          # ★ 预先建反查, 别用 list.index(O(N))
pos_p = {i: k for k, i in enumerate(ids_p)}
res = []
for name, Xs in X.items():
    n = Xs.shape[0]
    posmap = pos_p if name == "pixel-64" else pos_of
    sel_s = [(posmap[x], posmap[y]) for x, y in same if x in posmap and y in posmap]
    sel_d = [(posmap[x], posmap[y]) for x, y in diff if x in posmap and y in posmap]
    Zs = Xs.to(dev)
    mu, sd = Zs.mean(0, keepdim=True), Zs.std(0, keepdim=True) + 1e-6
    Zz = (Zs - mu) / sd
    Zr = (Zs - Zs.mean(0, keepdim=True))
    # PCA -> 白化 (在 GPU 上做低秩 SVD)
    q = min(a.pca_dim, min(Zr.shape) - 1)
    U, S, V = th.pca_lowrank(Zr, q=q, center=False)
    Zw = (Zr @ V) / (S.view(1, -1) / np.sqrt(n - 1) + 1e-8)
    row = [name]
    for tag, Z in (("raw", Zs), ("z", Zz), ("w", Zw)):
        ds, dd = pair_dists(Z, sel_s), pair_dists(Z, sel_d)
        row.append(auroc_from(tag, ds, dd))
    print(f"{name:<12} {row[1]:>9.4f} {row[2]:>9.4f} {row[3]:>11.4f} {Zs.shape[1]:>7}")
    res.append(row)
    del Zs, Zz, Zw, Zr
    th.cuda.empty_cache()

# ── D. 线性探针 (按字划分) ───────────────────────────────────────────
print("\n" + "=" * 78)
print("D. 线性探针 (logistic regression, **按字划分** 8:2; 留出集是没见过的字)")
print(f"{'空间':<12} {'raw 准确率':>12} {'z 准确率':>11} {'白化 准确率':>13} {'随机':>7} {'类别数':>7}")
print("-" * 78)
labels = [id2pair[i] for i in ids]
uniq = sorted(set(labels))
l2i = {c: k for k, c in enumerate(uniq)}
y_all = th.tensor([l2i[c] for c in labels], device=dev)
# 按字划分
allchars = sorted(set(id2char[i] for i in ids))
rng.shuffle(allchars)
tr_chars = set(allchars[: int(len(allchars) * 0.8)])
is_tr = th.tensor([id2char[i] in tr_chars for i in ids], device=dev)
print(f"[split] 训练字 {len(tr_chars)} / 留出字 {len(allchars) - len(tr_chars)} | "
      f"训练样本 {int(is_tr.sum())} / 留出 {int((~is_tr).sum())} | 类别 {len(uniq)}")
for name, Xs in X.items():
    row = [name]
    # pixel 空间只覆盖 ids 的前 pixel_cap 个 -> 标签/划分要跟着切
    _m = Xs.shape[0]
    y_use, tr_use = y_all[:_m], is_tr[:_m]
    for tag in ("raw", "z", "w"):
        Zs = Xs.to(dev)
        if tag == "raw":
            F = Zs
        elif tag == "z":
            F = (Zs - Zs.mean(0, keepdim=True)) / (Zs.std(0, keepdim=True) + 1e-6)
        else:
            Zr = Zs - Zs.mean(0, keepdim=True)
            q = min(a.pca_dim, min(Zr.shape) - 1)
            U, S, V = th.pca_lowrank(Zr, q=q, center=False)
            F = (Zr @ V) / (S.view(1, -1) / np.sqrt(Zr.shape[0] - 1) + 1e-8)
        W = th.zeros(F.shape[1], len(uniq), device=dev, requires_grad=True)
        b = th.zeros(len(uniq), device=dev, requires_grad=True)
        opt = th.optim.Adam([W, b], lr=0.05)
        Fn = F / (F.norm(dim=1, keepdim=True) + 1e-6)
        for _ in range(a.probe_steps):
            opt.zero_grad()
            logit = Fn @ W + b
            loss = th.nn.functional.cross_entropy(logit[tr_use], y_use[tr_use])
            loss.backward()
            opt.step()
        with th.no_grad():
            acc = (logit[~tr_use].argmax(1) == y_use[~tr_use]).float().mean().item()
        row.append(acc)
        del F, W, b
        th.cuda.empty_cache()
    print(f"{name:<12} {row[1]:>12.4f} {row[2]:>11.4f} {row[3]:>13.4f} "
          f"{1 / len(uniq):>7.4f} {len(uniq):>7}")
    res[list(X).index(name)].extend([row[1], row[2], row[3]])

with open(a.out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["space", "auroc_raw_l2", "auroc_z_l2", "auroc_pca_whiten",
                "probe_acc_raw", "probe_acc_z", "probe_acc_whiten"])
    w.writerows(res)
print(f"\n[out] {a.out}")
print("判读: 若 FLUX 在 z-L2 / 白化 L2 / 探针准确率上明显高于 SD -> 信息更富(之前 raw L2 掩盖了);")
print("      若三项都持平 -> 16ch 空间对'风格'确实没有更多可判别信息, 不是尺子的问题。")
