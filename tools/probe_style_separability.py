# -*- coding: utf-8 -*-
"""probe_style_separability.py — T2：生成图的"风格可分性"探针。

## 为什么需要它（替代 ratio_style 当主判据）

`ratio_style = inter_callig / intra` 有两个毛病：
  1. 依赖 cfg / k / 对数（16 对得 0.19、180 对得 0.32 —— 小样本严重低估）
  2. 定义上把"换书家"和"换噪声"放一起比，读不出"能不能认出是谁写的"

T2 直接问一个更干净的问题：

> **给一堆生成图，能不能把书家认出来？**

随机基线 = 1/n_callig。能到 20%+ 说明风格真的进了像素。

## 两个必须做对的地方（否则指标会骗人）

### ① 必须"留字"交叉验证，不能随机 CV

如果同一批字在不同折里都出现，分类器完全可以靠"认出这是哪个字 → 反查书家"
拿高分 —— 那测的是**字**，不是**风格**。

所以默认 `--group-by char`：**按字分组做 CV**，测试折里的字在训练折里从没出现过。
分类器必须在**没见过的字**上认出书家 —— 这才是"风格"。

### ② 必须同时报 **GT 天花板**

已知事实：DINO CLS 的 pair 质心间余弦 **0.956** —— DINO 对书家**本来就不敏感**。
所以"生成图 DINO 探针只有 8%"可能不是模型的问题，而是**特征空间的天花板低**。

本脚本对**同一批 GT 图**跑完全相同的流程，给出天花板。
**只有"生成 / GT"的比值才有意义。**

## 两类特征（都跑，互补）

| 特征 | 维度 | 说明 |
|---|---|---|
| `dino` | 384 | DINOv2 ViT-S/14 CLS。通用深度特征，但天花板可能低（见上） |
| `style` | ~28 | **手工笔画统计**：墨占比 / 墨色分位 / 笔画宽度(距离变换) / 边缘锐度 / 游程 / 3×3 密度分布 / 包围盒比例 … 可解释、无天花板问题 |

## 分类器（两种，都不依赖 sklearn）

* `centroid`：最近质心（零训练，最稳，先看它）
* `logreg`：torch 手写逻辑回归 + Adam（看线性可分性上限）

## 用法

```bash
# A) 从已保存的样本目录（batch_eval --save-samples 的产物）
#    目录里是 g*.png（生成）与 gt*.png（GT），按数字序对应 eval csv 的行序
python tools/probe_style_separability.py \
    --from-samples assets/ink_eval/v13_base_50k/strict \
    --eval-csv assets/eval_v13_strict.csv \
    --out assets/t2_v13base_strict.json

# B) 只用预提特征
python tools/probe_style_separability.py --features assets/t2_feats.npz

# C) 本地自检（合成图，验证代码路径）
python tools/probe_style_separability.py --self-test
```
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402


# ── 文件收集（锚定结尾，避免 gt0.png 被 g(\d+) 误匹配）───────────────────────
def _collect(d, prefix):
    out = {}
    for p in glob.glob(os.path.join(d, f"{prefix}*.png")):
        m = re.search(rf"{prefix}(\d+)\.png$", os.path.basename(p))
        if m:
            out[int(m.group(1))] = p
    return out


# ── 特征 1：DINOv2 CLS ──────────────────────────────────────────────────────
def _find_dino_ckpt():
    """按优先级找本地 DINOv2 权重。**绝不回退到 torch.hub**（需联网，受限机器会卡死）。

    ⚠ 实测踩到: `src.loss.losses._default_dino_ckpt()` 只找 `data/pretrained/`，
      而本地副本在 `pretrained_models/`。找不到时旧代码会回退 torch.hub ->
      联网下载 -> **静默卡住 6 分钟以上**（自检就是这样卡住的）。
    """
    cands = []
    try:
        from src.loss.losses import _default_dino_ckpt
        d = _default_dino_ckpt()
        if d:
            cands.append(d)
    except Exception:  # noqa: BLE001
        pass
    cands += [
        os.path.join(ROOT, "data", "pretrained", "dinov2_vits14_pretrain.safetensors"),
        # ⚠ 远端实际路径多一层 pretrained_models/（实测）
        os.path.join(ROOT, "data", "pretrained", "pretrained_models",
                     "dinov2_vits14_pretrain.safetensors"),
        os.path.join(ROOT, "pretrained_models", "dinov2_vits14_pretrain.safetensors"),
        os.path.join(ROOT, "pretrained_models", "dinov2_vits14_pretrain.pth"),
    ]
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


def feat_dino(paths, device="cpu", ckpt=None, size=256):
    from src.loss.losses import _load_local_dinov2
    from PIL import Image
    ck = ckpt or _find_dino_ckpt()
    if not ck:
        raise RuntimeError(
            "找不到本地 DINOv2 权重（已查 data/pretrained/ 与 pretrained_models/）。"
            "用 --dino-ckpt 指定；**不会回退 torch.hub**（联网会卡死）。")
    model = _load_local_dinov2(ck).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), 64):
            chunk = paths[i:i + 64]
            xs = []
            for p in chunk:
                im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
                xs.append(torch.from_numpy(
                    np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0))
            x = torch.stack(xs).to(device)
            x = (x - mean) / std
            o = model(x)
            f = o.last_hidden_state[:, 0] if hasattr(o, "last_hidden_state") else o[0][:, 0]
            feats.append(f.float().cpu().numpy())
    return np.concatenate(feats, 0).astype(np.float32)


# ── 特征 2：手工笔画统计（可解释）───────────────────────────────────────────
def _run_mean(mask):
    """每行连续 True 的长度均值（向量化）。"""
    if not mask.any():
        return 0.0
    pad = np.zeros((mask.shape[0], mask.shape[1] + 2), dtype=np.int8)
    pad[:, 1:-1] = mask.astype(np.int8)
    d = np.diff(pad, axis=1)
    st = np.where(d == 1)[1]
    en = np.where(d == -1)[1]
    if st.size == 0 or st.size != en.size:
        return 0.0
    return float((en - st).mean())


N_STYLE_FEAT = 29   # 见下方分项；必须与"空墨图"的补齐长度一致


def feat_style(paths):
    from PIL import Image
    from scipy import ndimage
    rows = []
    for p in paths:
        a = np.asarray(Image.open(p).convert("L").resize((256, 256), Image.BILINEAR),
                       dtype=np.float32)
        ink = a < 128
        n = ink.sum()
        r = []
        r.append(float(ink.mean()))                                   # 墨占比
        if n < 10:
            # ⚠ 补齐长度必须 == N_STYLE_FEAT，否则 np.stack 会因形状不一致报错
            #   （实测踩到：原来写 27，实际是 29 维）
            rows.append(np.zeros(N_STYLE_FEAT, dtype=np.float32))
            rows[-1][0] = float(ink.mean())
            continue
        v = a[ink] / 255.0
        r += [float(np.percentile(v, q)) for q in (10, 25, 50, 75, 90)]   # 墨色分位
        r.append(float(v.std()))
        # 笔画宽度：距离变换 ×2
        dt = ndimage.distance_transform_edt(ink)
        w = dt[ink] * 2.0
        r += [float(w.mean()), float(np.median(w)), float(w.std()),
              float(np.percentile(w, 90))]
        # 包围盒
        ys, xs = np.where(ink)
        # ⚠ 用 np.ptp() 而不是 arr.ptp()：后者在 numpy>=2.0 已被移除
        bh, bw = int(np.ptp(ys)) + 1, int(np.ptp(xs)) + 1
        r += [float(bh) / 256, float(bw) / 256, float(bw) / max(bh, 1)]
        # 墨心偏移（相对包围盒中心）
        r += [float(xs.mean() - (xs.min() + bw / 2)) / 256,
              float(ys.mean() - (ys.min() + bh / 2)) / 256]
        # 边缘锐度：墨边界处的梯度幅值
        gx = ndimage.sobel(a, 1)
        gy = ndimage.sobel(a, 0)
        g = np.hypot(gx, gy)
        edge = ndimage.binary_dilation(ink, iterations=1) & ~ndimage.binary_erosion(
            ink, iterations=1)
        r.append(float(g[edge].mean() / 255.0) if edge.any() else 0.0)
        # 游程（横/纵连墨长度）—— 反映笔画取向与连贯性
        # ⚠ 必须向量化：纯 Python 双层循环是 256×256 = 65k 次迭代/图，
        #   50 张就要 6 分钟（实测），249 张 × 8 ckpt 根本跑不动。
        r += [_run_mean(ink), _run_mean(ink.T)]
        # 3×3 密度分布（粗空间分布）
        h, w_ = 256 // 3, 256 // 3
        for i in range(3):
            for j in range(3):
                r.append(float(ink[i * h:(i + 1) * h, j * w_:(j + 1) * w_].mean()))
        r.append(float(ndimage.binary_erosion(ink, iterations=1).sum() / max(n, 1)))
        rows.append(np.asarray(r, dtype=np.float32))
    return np.stack(rows)


# ── 分类器 ──────────────────────────────────────────────────────────────────
def _norm(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)


def acc_centroid(Xtr, ytr, Xte, yte):
    C, lab = [], []
    for c in sorted(set(ytr.tolist())):
        m = ytr == c
        if m.sum():
            C.append(_norm(Xtr[m]).mean(0))
            lab.append(c)
    C = _norm(np.stack(C))
    pred = np.asarray(lab)[np.argmax(_norm(Xte) @ C.T, axis=1)]
    return float((pred == yte).mean())


def acc_ridge(Xtr, ytr, Xte, yte, lam=1e-2):
    """闭式线性分类器（岭回归 on one-hot）= 线性可分性上限。

    ⚠ 原来这里用 torch 手写逻辑回归 + Adam 600 步，实测 **单次调用 ~10 秒**
      （29×5 的矩阵，纯粹是 torch 小算子的线程/调度开销），
      一个 k=5 + 5 次置换检验的 run() 要 **79 秒**。改成闭式解后是毫秒级，
      而且没有随机性（不需要 seed，结果可复现）。
    """
    classes = sorted(set(ytr.tolist()))
    idx = {c: i for i, c in enumerate(classes)}
    if len(classes) < 2:
        return float("nan")
    Xtr = _norm(Xtr)
    Xte = _norm(Xte)
    Y = np.zeros((len(ytr), len(classes)), dtype=np.float64)
    for i, v in enumerate(ytr.tolist()):
        Y[i, idx[v]] = 1.0
    A = Xtr.astype(np.float64)
    d = A.shape[1]
    W = np.linalg.solve(A.T @ A + lam * np.eye(d), A.T @ Y)
    pred = (Xte.astype(np.float64) @ W).argmax(1)
    yte_t = np.asarray([idx.get(v, -1) for v in yte.tolist()])
    ok = yte_t >= 0
    return float((pred[ok] == yte_t[ok]).mean()) if ok.any() else float("nan")


def grouped_cv(X, y, groups, k=5, seed=0, group_by="char"):
    """按 groups 分组做 k 折（组内不跨折）。返回 (acc_centroid, acc_logreg, n_te)。"""
    rng = np.random.RandomState(seed)
    if group_by == "none":
        g = rng.permutation(len(y)) % k
    else:
        uq = sorted(set(groups.tolist()))
        rng.shuffle(uq)
        m = {v: i % k for i, v in enumerate(uq)}
        g = np.array([m[v] for v in groups.tolist()])
    ac, al, nte = [], [], 0
    for f in range(k):
        tr, te = g != f, g == f
        if tr.sum() < 2 or te.sum() < 2:
            continue
        if len(set(y[tr].tolist())) < 2:
            continue
        ac.append(acc_centroid(X[tr], y[tr], X[te], y[te]))
        al.append(acc_ridge(X[tr], y[tr], X[te], y[te]))
        nte += int(te.sum())
    f_ = lambda v: float(np.nanmean(v)) if len(v) else float("nan")
    return f_(ac), f_(al), nte


# ── 主流程 ──────────────────────────────────────────────────────────────────
def run(X, y, groups, tag, group_by="char", k=5, n_perm=3):
    n_cls = len(set(y.tolist()))
    base = 1.0 / max(n_cls, 1)
    ac, al, nte = grouped_cv(X, y, groups, k=k, group_by=group_by)
    # 置换检验：打乱标签后重跑，得到"随机水平"的经验分布（比 1/n 更保守）
    perm = []
    rng = np.random.RandomState(0)
    for _ in range(n_perm):
        yp = y.copy()
        rng.shuffle(yp)
        perm.append(grouped_cv(X, yp, groups, k=k, group_by=group_by)[0])
    p_mean = float(np.nanmean(perm)) if perm else base
    p_std = float(np.nanstd(perm)) if perm else 0.0
    out = dict(tag=tag, n=len(y), n_classes=n_cls, n_test=nte,
               group_by=group_by, k=k,
               acc_centroid=ac, acc_logreg=al,
               baseline=base, perm_mean=p_mean, perm_std=p_std,
               lift_centroid=(ac - p_mean) if ac == ac else None)
    return out


def _print(out):
    print(f"\n── {out['tag']} ──")
    print(f"   样本 {out['n']} / 书家 {out['n_classes']} / CV={out['k']}折 "
          f"(group_by={out['group_by']}, 测试 {out['n_test']} 条)")
    print(f"   随机基线 1/{out['n_classes']} = {out['baseline']:.4f}"
          f"   置换检验均值 {out['perm_mean']:.4f} ± {out['perm_std']:.4f}")
    print(f"   最近质心 top-1 = {out['acc_centroid']:.4f}"
          f"   (相对随机 {out['lift_centroid']:+.4f})")
    print(f"   线性(岭回归) top-1 = {out['acc_logreg']:.4f}")


def _self_test(no_dino=False):
    """合成图：让「书家」真的可区分（不同笔画宽度/墨色），验证探针能测出来。"""
    from PIL import Image, ImageDraw
    import tempfile
    rng = np.random.RandomState(0)
    chars = list("一二三大小上下天地人日月")
    cals = ["A", "B", "C", "D", "E"]
    tmp = tempfile.mkdtemp(prefix="t2self_")
    paths, labels, groups = [], [], []
    for ci, cal in enumerate(cals):
        for ch in chars:
            for rep in range(2):
                im = Image.new("L", (256, 256), 255)
                d = ImageDraw.Draw(im)
                lw = 2 + ci * 4                      # 书家 -> 笔画宽度
                gray = 20 + ci * 30                  # 书家 -> 墨色
                for k in range(1, 4):
                    d.line([(40, k * 60), (216, k * 60 + (ci - 2) * 6)],
                           fill=gray, width=lw)
                p = os.path.join(tmp, f"{cal}_{ch}_{rep}.png")
                im.save(p)
                paths.append(p)
                labels.append(ci)
                groups.append(chars.index(ch))
    y = np.asarray(labels)
    g = np.asarray(groups)
    print("# self-test: 5 个「书家」，笔画宽度/墨色不同；留字 CV 应显著高于随机")
    _print(run(feat_style(paths), y, g, "style(手工)"))
    if no_dino:
        print("   [skip] dino（--no-dino）")
        return 0
    try:
        _print(run(feat_dino(paths, device="cpu"), y, g, "dino"))
    except Exception as e:  # noqa: BLE001
        print(f"   [skip] dino 不可用: {e!r}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-samples", default=None,
                    help="batch_eval --save-samples 的目录（含 g*.png / gt*.png）")
    ap.add_argument("--eval-csv", default=None, help="与样本一一对应的 eval csv")
    ap.add_argument("--features", default=None, help="预提特征 npz: {feat,labels,groups}")
    ap.add_argument("--dino-ckpt", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--group-by", choices=["char", "none"], default="char",
                    help="char(默认)=留字 CV（测风格）；none=随机 CV（偏乐观，作对照）")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=3,
                    help="置换检验次数（默认 3，够估随机水平且快）")
    ap.add_argument("--no-dino", action="store_true", help="跳过 DINO（无权重时）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return _self_test(no_dino=a.no_dino)

    if a.features:
        d = np.load(a.features)
        X, y = d["feat"], d["labels"]
        groups = d["groups"] if "groups" in d else np.arange(len(y))
        print(f"# 载入特征 {a.features}: {X.shape}")
        res = [run(X, y, groups, "precomputed", group_by=a.group_by, k=a.k, n_perm=a.n_perm)]
        for r in res:
            _print(r)
        if a.out:
            json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        return 0

    if not a.from_samples:
        raise SystemExit("需要 --from-samples / --features / --self-test")

    g_files = _collect(a.from_samples, "g")
    gt_files = _collect(a.from_samples, "gt")
    if not g_files:
        raise SystemExit(f"{a.from_samples} 里没有 g*.png")
    idx = sorted(g_files)
    print(f"# 找到生成图 {len(idx)} 张；GT {len(gt_files)} 张")

    if not a.eval_csv:
        raise SystemExit("需要 --eval-csv 来拿书家标签（按行序与 g*.png 数字序对应）")
    rows = list(csv.DictReader(open(a.eval_csv, encoding="utf-8")))
    if len(rows) < len(idx):
        raise SystemExit(f"eval csv 只有 {len(rows)} 行 < 样本 {len(idx)}")

    cals = [int(rows[i]["calligrapher_id"]) for i in idx]
    chars = [str(rows[i].get("character", i)) for i in idx]
    y = np.asarray(cals)
    # 字 -> 组 id（留字 CV）
    uqc = {c: i for i, c in enumerate(sorted(set(chars)))}
    groups = np.asarray([uqc[c] for c in chars])
    print(f"# 书家 {len(set(cals))} 个 / 字 {len(uqc)} 个 / "
          f"分布 top5 {Counter(cals).most_common(5)}")

    results = []
    gen_paths = [g_files[i] for i in idx]
    gt_paths = [gt_files[i] for i in idx if i in gt_files]

    Xs = feat_style(gen_paths)
    results.append(run(Xs, y, groups, "生成图 / style(手工)", a.group_by, a.k, a.n_perm))
    if len(gt_paths) >= 20:
        Xgt = feat_style(gt_paths)
        ygt = np.asarray([cals[idx.index(i)] for i in idx if i in gt_files])
        ggt = np.asarray([groups[idx.index(i)] for i in idx if i in gt_files])
        results.append(run(Xgt, ygt, ggt, "GT 天花板 / style(手工)", a.group_by, a.k, a.n_perm))

    if not a.no_dino:
        try:
            Xd = feat_dino(gen_paths, device=a.device, ckpt=a.dino_ckpt)
            results.append(run(Xd, y, groups, "生成图 / dino", a.group_by, a.k, a.n_perm))
            if len(gt_paths) >= 20:
                Xdgt = feat_dino(gt_paths, device=a.device, ckpt=a.dino_ckpt)
                results.append(run(Xdgt, ygt, ggt, "GT 天花板 / dino", a.group_by, a.k, a.n_perm))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] DINO 不可用，跳过: {e!r}")

    for r in results:
        _print(r)

    print("\n" + "=" * 70)
    print("怎么读：")
    print("  * 只有「生成 / GT 天花板」的**比值**有意义 —— DINO 对书家本来就不敏感")
    print("    （pair 质心余弦 0.956），GT 天花板低不是模型的问题。")
    print("  * 必须看 group_by=char（留字 CV）。group_by=none 偏乐观，只作对照。")
    print("  * 判据：lift 显著 > 0 且生成/GT ≥ 0.5 才算「风格真的进了像素」。")
    print("=" * 70)

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(results, open(a.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"saved -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
