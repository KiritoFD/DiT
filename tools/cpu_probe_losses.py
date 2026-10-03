"""CPU 测试: latent L2 该换成哪个结构损失?

方法 (不需要 VAE / 不需要训练):
  用评测缓存里已有的解码图, 对真迹墨迹施加**已知严重度的结构破坏**
  (平移/加粗/变细/挖块/换成标准字/换成别的字), 然后衡量每个候选损失:
    1) 单调性: 结构越坏, 损失是否单调变大?
    2) 与 (1 - IoU) 的相关性: 最小化它是否等价于最大化 IoU?
    3) 小误差灵敏度: 偏 1~2px 时是否还有斜率 (细结构最怕梯度为 0)?

候选:
  A. 投影剖面 (多方向 Radon 求和剖面, L1)  —— SWD 的确定性低配
  B. SWD proper (随机方向投影 + 排序, L1)
  C. 距离场加权 (GT 的距离变换场 × 预测墨)
  D. Dice / soft-IoU
"""
import numpy as np
import torch as th
from scipy import ndimage
from scipy.stats import spearmanr

CACHE = "data/top10_style23/eval_real200_cache.pt"
N = 60
SEED = 0
rng = np.random.default_rng(SEED)

c = th.load(CACHE, map_location="cpu", weights_only=False)


def to_mask(t):
    return (t[:N].mean(1).numpy() < 0.6)


gt = to_mask(c["gt_pngs"])
std = to_mask(c["std_pngs"])
print(f"[data] gt={gt.shape} std={std.shape}")


# ── 候选损失 ────────────────────────────────────────────────────────────
def iou(a, b):
    inter = (a & b).sum((-1, -2))
    union = (a | b).sum((-1, -2))
    return inter / np.maximum(union, 1)


def profiles(m, angles=(0, 45, 90, 135)):
    """Radon 式: 沿若干角度求和 -> 1D 剖面 (归一化到和=1)。"""
    out = []
    for ang in angles:
        r = ndimage.rotate(m.astype(np.float32), ang, reshape=False, order=0)
        p = r.sum(0)
        s = p.sum()
        out.append(p / max(s, 1e-6))
    return np.stack(out, 0)          # (A, W)


def loss_proj(a, b):
    return np.abs(profiles(a) - profiles(b)).sum(-1).mean(-1)


def _pts(m, k=1500):
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return np.zeros((k, 2), np.float32)
    idx = rng.choice(len(ys), size=min(k, len(ys)), replace=False)
    return np.stack([xs[idx], ys[idx]], 1).astype(np.float32)


def loss_swd(a, b, nproj=64, k=1500):
    th_dirs = rng.normal(size=(nproj, 2))
    th_dirs /= np.linalg.norm(th_dirs, axis=1, keepdims=True)
    pa, pb = _pts(a, k), _pts(b, k)
    sa = np.sort(pa @ th_dirs.T, 0)
    sb = np.sort(pb @ th_dirs.T, 0)
    return float(np.abs(sa - sb).mean())


def loss_dt(a, b):
    """GT 的距离变换场 × 预测墨: 把墨往 GT 笔画上拉, 对错位有平滑梯度。"""
    d = ndimage.distance_transform_edt(~b)          # 离 GT 墨的距离
    d = d / max(d.max(), 1e-6)
    return float((d * a).sum() / max(a.sum(), 1e-6))


def loss_dice(a, b):
    inter = (a & b).sum()
    return 1.0 - 2 * inter / max(a.sum() + b.sum(), 1e-6)


# ── 破坏: (名字, 函数) ──────────────────────────────────────────────────
def shift(m, k):
    return np.roll(np.roll(m, k, axis=0), k, axis=1)


def dil(m, k):
    return ndimage.binary_dilation(m, iterations=k)


def ero(m, k):
    return ndimage.binary_erosion(m, iterations=k)


def hole(m, size=48):
    o = m.copy()
    y = rng.integers(0, 256 - size)
    x = rng.integers(0, 256 - size)
    o[y:y + size, x:x + size] = False
    return o


perts = []
for n in range(N):
    g = gt[n]
    perts += [
        ("identity", g, g),
        ("shift1", shift(g, 1), g),
        ("shift2", shift(g, 2), g),
        ("shift4", shift(g, 4), g),
        ("shift8", shift(g, 8), g),
        ("thick1", dil(g, 1), g),
        ("thick3", dil(g, 3), g),
        ("thin1", ero(g, 1), g),
        ("hole48", hole(g), g),
        ("std(不做)", std[n], g),
        ("other_char", gt[(n + 7) % N], g),
    ]

rows = []
for name, p, g in perts:
    rows.append((name, float(iou(p, g)), loss_proj(p, g), loss_swd(p, g),
                 loss_dt(p, g), loss_dice(p, g)))

print("\n=== 各破坏下的 IoU 与候选损失 (60 样本平均) ===")
print(f"{'破坏':<12}{'IoU':>8}{'proj':>9}{'swd':>9}{'dt':>9}{'dice':>9}")
import collections
agg = collections.defaultdict(lambda: [0.0] * 5)
cnt = collections.Counter()
for name, i, lp, ls, ld, lc in rows:
    agg[name][0] += i
    agg[name][1] += lp
    agg[name][2] += ls
    agg[name][3] += ld
    agg[name][4] += lc
    cnt[name] += 1
for name in sorted(agg, key=lambda k: -agg[k][0] / cnt[k]):
    v = [x / cnt[name] for x in agg[name]]
    print(f"{name:<12}{v[0]:>8.4f}{v[1]:>9.4f}{v[2]:>9.2f}{v[3]:>9.4f}{v[4]:>9.4f}")

print("\n=== 与 (1-IoU) 的 Spearman 相关 (越接近 1 = 最小化它就等于最大化 IoU) ===")
err = np.array([1.0 - r[1] for r in rows])
for j, nm in ((2, "投影剖面 proj"), (3, "SWD"), (4, "距离场 dt"), (5, "Dice")):
    x = np.array([r[j] for r in rows])
    rho = spearmanr(x, err).statistic
    print(f"  {nm:<16} rho = {rho:+.3f}")

print("\n=== 小误差灵敏度: 平移 1px 时的损失增量 (相对 identity) ===")
base = {nm: v for nm, v in
        ((nm, [x / cnt[nm] for x in agg[nm]]) for nm in agg)}
for j, nm in ((1, "proj"), (2, "swd"), (3, "dt"), (4, "dice")):
    d = base["shift1"][j] - base["identity"][j]
    d4 = base["shift4"][j] - base["identity"][j]
    print(f"  {nm:<6} Δ(shift1) = {d:+.5f}   Δ(shift4) = {d4:+.5f}")
print("\n对比: IoU 自身在 shift1 时 drop = "
      f"{base['identity'][0] - base['shift1'][0]:.4f}")
