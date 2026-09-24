#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""免训练风格信号测试：内容匹配的书家检索。

问题: 不训任何东西，有没有一个表示能把"书家"从图像里读出来？
做法: 对每个 query（strict 样本的 GT），在**同字**的训练样本池里找最近邻，
      看它是不是同书家。命中率 vs 随机基线 = 富集倍数。
      这与 cal_enrich 同口径，可直接对比。

测的表示（全部免训练）:
  dino_mean / dino_std / dino_meanstd   —— DINOv2-S/14 patch 特征（已有缓存）
  pixel_ssim                            —— 当前指标的基准
  pixel_l2                              —— 原图 L2
  ink_stats                             —— 手写墨迹统计（墨量/笔宽/重心/倾斜）
"""
import csv, json, os, sys
import concurrent.futures as cf
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

os.chdir('/root/Workspace/xy/DiT'); sys.path.insert(0, '.')
from src.eval.metrics import ssim as _ssim

CACHE = 'data/dino_cache/50k_v1'
MAXC, NW = 12, 16


def rid(p):
    import re
    m = re.search(r'(\d+)\.png$', str(p))
    return int(m.group(1)) if m else None


tr = list(csv.DictReader(open('assets/train_50k_v2_fixed.csv', encoding='utf-8')))
ev = list(csv.DictReader(open('assets/eval_v13_strict_fixed.csv', encoding='utf-8')))[:249]
by_char = {}
for r in tr:
    by_char.setdefault(str(r.get('character', '')), []).append(r)

ids = np.load(f'{CACHE}/ids.npy')
pos = {int(i): k for k, i in enumerate(ids)}
feats = np.memmap(f'{CACHE}/feats.f16', dtype=np.float16, mode='r',
                  shape=(len(ids), 256, 384))


def dino(i):
    k = pos.get(int(i))
    if k is None:
        return None
    f = np.asarray(feats[k], dtype=np.float32)      # (256,384)
    return f


def gray(p):
    with Image.open(p) as f:
        return np.asarray(f.convert('L'), dtype=np.float32) / 255.0


def ink_stats(p):
    g = gray(p)
    ink = g < 0.5
    if ink.sum() < 5:
        return np.zeros(8, np.float32)
    d = distance_transform_edt(ink)
    ys, xs = np.nonzero(ink)
    # 墨量 / 笔宽(2*mean EDT@骨架近似用 2*mean(d[ink])) / 包围盒长宽比 / 重心 / 二阶矩
    w = 2 * float(d.max())
    h, ww = g.shape
    return np.array([ink.mean(), w, (ys.max()-ys.min()+1)/(xs.max()-xs.min()+1),
                     ys.mean()/h, xs.mean()/ww,
                     ys.std()/h, xs.std()/ww, float(ink.sum())**0.5/100.],
                    np.float32)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def one(job):
    i, r = job
    ch = str(r.get('character', ''))
    cal = r.get('calligrapher')
    qid = rid(r.get('image_path', ''))
    pool = [x for x in by_char.get(ch, []) if rid(x.get('image_path', '')) != qid]
    if not pool or len(pool) > MAXC * 2:
        pool = pool[:MAXC * 2] if pool else []
    if not pool:
        return None
    gp = r.get('image_path', '')
    if not os.path.exists(gp):
        return None
    qi = qid
    out = {}
    # ---- DINO ----
    fq = dino(qi)
    if fq is not None:
        q_mean, q_std = fq.mean(0), fq.std(0)
        q_ms = np.concatenate([q_mean, q_std])
        cand = []
        for x in pool:
            cid = rid(x.get('image_path', ''))
            fc = dino(cid) if cid is not None else None
            if fc is None:
                continue
            cand.append((x.get('calligrapher'), fc.mean(0), fc.std(0),
                         np.concatenate([fc.mean(0), fc.std(0)])))
        if cand:
            for nm, qv, idx in [('dino_mean', q_mean, 1), ('dino_std', q_std, 2),
                                ('dino_meanstd', q_ms, 3)]:
                sc = [(cos(qv, c[idx]), c[0]) for c in cand]
                out[nm] = max(sc)[1] == cal
    # ---- 像素 ----
    g = gray(gp)
    ss, l2, isc = [], [], []
    q_ink = ink_stats(gp)
    for x in pool:
        p = x.get('image_path', '')
        if not os.path.exists(p):
            continue
        gp2 = np.asarray(Image.open(p).convert('L'), dtype=np.float32) / 255.0
        if gp2.shape != g.shape:
            continue
        rgb = np.stack([g]*3, -1); rgb2 = np.stack([gp2]*3, -1)
        ss.append((float(_ssim(rgb, rgb2)), x.get('calligrapher')))
        l2.append((-float(np.abs(g - gp2).mean()), x.get('calligrapher')))
        isc.append((-float(np.abs(q_ink - ink_stats(p)).sum()), x.get('calligrapher')))
    if ss:
        out['pixel_ssim'] = max(ss)[1] == cal
        out['pixel_l2'] = max(l2)[1] == cal
        out['ink_stats'] = max(isc)[1] == cal
    out['_cal'] = cal
    out['_n'] = len(ss)
    out['_ncal'] = sum(1 for _, c in ss if c == cal)
    return out


jobs = [(i, r) for i, r in enumerate(ev)]
with cf.ThreadPoolExecutor(max_workers=NW) as ex:
    rs = [x for x in ex.map(one, jobs) if x]
print(f'{len(rs)} 列有效（同字池非空且有 DINO 特征）')
KEYS = ['dino_mean', 'dino_std', 'dino_meanstd', 'pixel_ssim', 'pixel_l2', 'ink_stats']
print(f'\n{"表示":>16} | {"命中率":>8} {"基线":>8} {"富集":>7} {"n":>5}')
print('-' * 54)
for k in KEYS:
    ok = [x for x in rs if k in x]
    if not ok:
        print(f'{k:>16} | 不可用'); continue
    hit = np.mean([x[k] for x in ok])
    base = np.mean([x['_ncal'] / max(x['_n'], 1) for x in ok])
    print(f'{k:>16} | {hit:7.1%} {base:7.1%} {hit/max(base,1e-9):6.2f}x {len(ok):>5}')

# 免训练信号的"检索稳定性": 同一样本不同实例能否互相找回
print(f'\n参考: 纯随机猜 = 基线列（≈ 1/池内候选数）')
print('判读: 富集 >1.3x 才算"有可用的免训练信号"; ~1.0x = 该表示读不出书家')
