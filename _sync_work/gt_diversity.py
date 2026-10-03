"""数据自身的"输出多样性"参照值。

模型输出多样性该拿什么比? 拿**真迹**比: 同一个书家写同一个字的多张真迹,
它们两两之间的差异 = "人写同一个字两遍的自然差异"。模型的 intra 应该接近这个值。

太高 -> 输出乱飘; 太低 -> 模式塌缩(背答案)。
"""
import csv, os, collections, random, statistics, itertools
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

os.chdir("/root/Workspace/xy/DiT")
random.seed(0)


def ssim_np(a, b, win=7):
    """单通道 SSIM (图是灰度二值, 直接算)。"""
    x = a.astype(np.float64)
    y = b.astype(np.float64)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mx, my = uniform_filter(x, win), uniform_filter(y, win)
    mx2, my2, mxy = mx ** 2, my ** 2, mx * my
    sx2 = uniform_filter(x * x, win) - mx2
    sy2 = uniform_filter(y * y, win) - my2
    sxy = uniform_filter(x * y, win) - mxy
    return float((((2 * mxy + c1) * (2 * sxy + c2))
                  / ((mx2 + my2 + c1) * (sx2 + sy2 + c2))).mean())


def load_gray(p):
    return np.asarray(Image.open(p).convert("L").resize((256, 256), Image.LANCZOS),
                      dtype=np.float32) / 255.0


for tag, path in [("老数据", "assets/train_fame-kxl-tj-px60.csv"),
                  ("HCSU", "assets/train_hcsu_kxl.csv")]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    # ★ 参照系必须是 **(书家, 字, 书体)** 三元组 —— 只按 (书家,字) 会把
    #   "同一个字在楷书 vs 行书"的差异混进来(实测老数据 41% 的多图组合跨书体),
    #   那测的是书体差异, 不是"写两遍"的自然差异。
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["calligrapher"], r["character"], r["script"])].append(r["image_path"])
    multi = {k: v for k, v in g.items() if len(v) >= 2}
    print("=" * 70)
    print(f"{tag}: {len(g)} 个 (书家,字,书体) 三元组, 其中 >=2 张的有 {len(multi)} 个 "
          f"({100*len(multi)/len(g):.1f}%)")

    # 抽样算两两 1-SSIM
    keys = random.sample(list(multi), min(400, len(multi)))
    divs, dists = [], []
    for k in keys:
        ps = multi[k][:6]
        try:
            ims = [load_gray(p) for p in ps]
        except Exception:
            continue
        for a, b in itertools.combinations(range(len(ims)), 2):
            divs.append(1.0 - ssim_np(ims[a], ims[b]))
            dists.append(float(np.abs(ims[a] - ims[b]).mean()))
    print(f"  抽样 {len(keys)} 个组合, {len(divs)} 个真迹对")
    print(f"  **真迹 intra (1−SSIM): med={statistics.median(divs):.4f}  "
          f"mean={statistics.mean(divs):.4f}  "
          f"p10={sorted(divs)[len(divs)//10]:.4f}  p90={sorted(divs)[9*len(divs)//10]:.4f}**")
    print(f"  像素绝对差 mean={statistics.mean(dists):.4f}")

print("=" * 70)
print("对照: 模型 v12 (100k) 的 intra = 0.1878  (1−SSIM, 同条件换噪声)")
print("对照: 模型 v12_d8     的 intra = 0.2519")
