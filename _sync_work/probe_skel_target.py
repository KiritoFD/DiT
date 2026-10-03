#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""哪个骨架是"逐样本"的（可做中间监督目标）？

判据：同一个字、不同书家 -> 骨架是否不同。
  共享（字体渲染）: 同字所有样本 md5 相同
  逐样本        : 同字样本各不同
对照 std 已知是共享的（43.5% 的字只有 1 张）。
"""
import csv, hashlib, os, collections
os.chdir('/root/Workspace/xy/DiT')

rows = list(csv.DictReader(open('assets/train_50k_v2_fixed.csv', encoding='utf-8')))[:8000]
DIRS = {
    'std':        lambda r: r.get('std_path', ''),
    'inst_skel1': lambda r: 'data/50k/inst_skel1/%06d.png' % int(os.path.basename(r['image_path']).split('.')[0]),
    'inst_skel20': lambda r: 'data/50k/inst_skel20/%06d.png' % int(os.path.basename(r['image_path']).split('.')[0]),
    'aux_skel3':  lambda r: 'data/50k/aux_skel3/%06d.png' % int(os.path.basename(r['image_path']).split('.')[0]),
}


def h(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()[:10] if os.path.exists(p) else None


by_char = collections.defaultdict(list)
for r in rows:
    by_char[str(r.get('character', ''))].append(r)

print('统计 %d 个字 / %d 样本' % (len(by_char), len(rows)))
print()
print('%-14s %-16s %-16s %s' % ('目录', '只有1张的字占比', 'std数/样本数', '样本覆盖'))
for nm, fn in DIRS.items():
    n1 = tot = 0
    ratios = []
    cov = 0
    for ch, rs in by_char.items():
        hs = {h(fn(x)) for x in rs}
        hs.discard(None)
        if not hs:
            continue
        cov += len([x for x in rs if h(fn(x))])
        tot += 1
        if len(hs) == 1:
            n1 += 1
        ratios.append(len(hs) / len(rs))
    if not tot:
        print('%-14s 不可读' % nm); continue
    print('%-14s %14.1f%% %16.3f %d/%d' %
          (nm, 100.0*n1/tot, sum(ratios)/len(ratios), cov, len(rows)))
print()
print('读法: "只有1张的字占比"低 + "std数/样本数"高 => 逐样本(可做监督目标)')
print('      std 已知是共享的 -> 应表现为占比高、比值低')
