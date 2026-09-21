# -*- coding: utf-8 -*-
"""评估 Kaggle 'Chinese Calligraphy Styles by Calligraphers' 能否并入 50k。"""
import io, os, csv, zipfile, random, collections
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

D = r"G:\GitHub\DiT\Chinese Calligraphy Styles by Calligraphers_datasets(1)\Chinese Calligraphy Styles by Calligraphers_datasets\Chinese Calligraphy Styles by Calligraphers_data_datasets\data"
TRAIN = os.path.join(D, "Chinese Calligraphy Styles by Calligraphers_train_datasets.zip")
TEST = os.path.join(D, "Chinese Calligraphy Styles by Calligraphers_test_datasets.zip")

# ---- 1) Summary.csv (GBK) ----
print("===== Summary.csv (书家统计) =====")
raw = open(os.path.join(D, "Summary.csv"), "rb").read()
txt = raw.decode("gbk", errors="replace")
rows = list(csv.reader(io.StringIO(txt)))
hdr = rows[0]
print("header:", hdr[:6])
data = [r for r in rows[1:] if len(r) >= 5 and r[0].strip()]
tot_train = sum(int(r[3]) for r in data if r[3].strip().isdigit())
tot_all = sum(int(r[2]) for r in data if r[2].strip().isdigit())
print(f"书家数={len(data)}  训练图合计={tot_train}  全部合计={tot_all}")
for r in data[:12]:
    print(f"  {r[0]:<8} {r[1]:<10} total={r[2]:<6} train={r[3]:<6} test={r[4]}")

# ---- 2) zip 内部结构: 是否再分书体? 目录深度? ----
zf = zipfile.ZipFile(TRAIN)
names = [n for n in zf.namelist() if not n.endswith('/')]
print(f"\n===== train.zip 文件数={len(names)} =====")
# 顶层目录
tops = collections.Counter(n.split('/')[0] for n in names)
print("顶层目录数(=书家?):", len(tops))
print("前 10 顶层:", list(tops.items())[:10])
# 路径深度分布
depth = collections.Counter(n.count('/') for n in names)
print("路径 '/' 计数分布:", dict(depth))
# 是否出现书体关键字
script_kw = ['楷', '行', '草', '隶', '篆', '行书', '楷书', 'kai', 'xing', 'cao', 'li', 'zhuan']
sample_paths = names[:2000]
hits = {k: sum(1 for n in sample_paths if k in n) for k in script_kw}
print("书体关键字命中(前2000路径):", {k: v for k, v in hits.items() if v})
# 看一个书家目录下的二级结构
one_top = list(tops)[0]
sub = collections.Counter(n.split('/')[1] for n in names if n.startswith(one_top + '/') and n.count('/') >= 2)
print(f"\n示例书家目录 '{one_top}' 下二级目录:", list(sub.items())[:12])
# 文件名示例
ex = [n for n in names if n.startswith(one_top + '/')][:5]
print("文件名示例:", ex)

# ---- 3) 抽样图像属性: 尺寸 / 模式 / 背景极性 ----
print("\n===== 抽样图像属性 (跨书家随机 200 张) =====")
img_names = [n for n in names if n.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
random.seed(0)
samp = random.sample(img_names, min(200, len(img_names)))
sizes = collections.Counter()
modes = collections.Counter()
ink_ratios, gray_means = [], []
wb = bb = 0
for n in samp:
    try:
        im = Image.open(io.BytesIO(zf.read(n)))
        modes[im.mode] += 1
        g = im.convert('L')
        sizes[g.size] += 1
        a = np.asarray(g, dtype=np.float32)
        gm = a.mean(); ink = (a < 128).mean()
        gray_means.append(gm); ink_ratios.append(ink)
        if ink < 0.35 and gm > 140: wb += 1        # 白底黑字
        elif ink > 0.5 or gm < 100: bb += 1        # 黑底/反相
    except Exception as e:
        print("  read fail", n, e)
print(f"模式: {dict(modes)}")
print(f"尺寸 top5: {sizes.most_common(5)}")
ink_ratios = np.array(ink_ratios); gray_means = np.array(gray_means)
print(f"ink<128 占比: mean={ink_ratios.mean():.3f} p10={np.percentile(ink_ratios,10):.3f} "
      f"p50={np.percentile(ink_ratios,50):.3f} p90={np.percentile(ink_ratios,90):.3f}")
print(f"灰度均值: mean={gray_means.mean():.1f}  白底(ink<0.35&mean>140)={wb}  黑底/反相={bb}  (共{len(samp)})")
