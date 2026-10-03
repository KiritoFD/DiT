#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""g 到底是什么：同一个字的不同样本，std 骨架是同一张（字体渲染）还是各不同（逐样本）？

判据：按 md5 去重。
  · 每个字只有 1 个不同 std -> g 是**字体渲染**（与书家无关）
  · 每个字有 N 个不同 std（N≈该字样本数）-> g 是**逐样本骨架**（携带书写差异）
"""
import csv, hashlib, os, collections
os.chdir('/root/Workspace/xy/DiT')

rows = list(csv.DictReader(open('assets/train_50k_v2_fixed.csv', encoding='utf-8')))
print('总样本 %d' % len(rows))


def h(p):
    if not os.path.exists(p):
        return None
    return hashlib.md5(open(p, 'rb').read()).hexdigest()[:10]


by_char = collections.defaultdict(list)
for r in rows[:8000]:
    by_char[str(r.get('character', ''))].append(r)

n1 = 0; tot = 0; ratios = []
for ch, rs in by_char.items():
    hs = {h(x.get('std_path', '')) for x in rs}
    hs.discard(None)
    if not hs:
        continue
    tot += 1
    if len(hs) == 1:
        n1 += 1
    ratios.append(len(hs) / len(rs))
print('统计 ' + str(tot) + ' 个字')
print('  只有 1 个不同 std 的字: ' + str(n1) + ' (' + ('%.1f' % (100.0*n1/max(tot,1))) + '%)  <- 接近全部则是纯字体渲染')
print('  不同 std 数/样本数 均值: ' + ('%.3f' % (sum(ratios)/max(len(ratios),1))) + '  (~1.0 = 每个样本都不同)')



# 同一书家同一字 vs 不同书家同一字
ex = [r for r in rows[:8000] if str(r.get('character','')) == '出']
print()
print('例：字「出」有 %d 个样本，来自 %d 个书家' % (len(ex), len({r.get('calligrapher') for r in ex})))
hs = {h(r.get('std_path','')) for r in ex}
print('  不同 std 数 = %d' % len(hs))
for r in ex[:6]:
    print('   书家=%-8s std=%s md5=%s' % (r.get('calligrapher'), r.get('std_path'), h(r.get('std_path',''))))

# 源图是不是字体渲染：看 fame 目录
src = rows[0].get('src_image_path', '')
print()
print('源图目录样例:', src)
import glob
d = os.path.dirname(src)
fs = sorted(glob.glob(d + '/*.png'))[:3] if os.path.isdir(d) else []
print('  该目录存在:', os.path.isdir(d), '样例:', [os.path.basename(x) for x in fs])
