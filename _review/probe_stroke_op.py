#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_stroke_op.py — 折线生成式算子原型: 抽折线 -> 只动点 -> 重新栅格化。

动机: warp 是重采样算子, 散度大的地方会把细脊摊到阈值以下 -> 断笔。
      折线栅格化按构造输出连通线, 断不了。这是"算子保证是线"的唯一做法。

本脚本只验证可行性(不训练):
  A. 能否从标准骨架 PNG 抽出折线 (骨架化 -> 追踪 -> RDP 简化)
  B. 用距离场栅格化回去, 与原始二值图的 IoU
  C. 栅格化结果的连通分量数 == 笔画条数 (按构造, 而非恰好)
  D. 对折线点施加"形变"后, 连通性是否仍然成立(应该是恒成立)
"""
import glob
import os
import sys

import numpy as np
import torch as th
from PIL import Image
from scipy.ndimage import label
from skimage.morphology import skeletonize

os.chdir('/root/Workspace/xy/DiT')
DEV = 'cpu'
SIZE = 256
N = 16
NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def trace_polylines(sk):
    """骨架二值图 -> 折线列表。每个折线是 [(y,x), ...]。"""
    pts = set(zip(*np.nonzero(sk)))
    deg = {}
    for p in pts:
        deg[p] = sum(((p[0] + dy, p[1] + dx) in pts) for dy, dx in NB8)
    nodes = {p for p in pts if deg[p] != 2}
    visited = set()
    lines = []

    def walk(start, nxt):
        line = [start, nxt]
        prev, cur = start, nxt
        while cur not in nodes:
            nbrs = [(cur[0] + dy, cur[1] + dx) for dy, dx in NB8
                    if (cur[0] + dy, cur[1] + dx) in pts]
            nxts = [q for q in nbrs if q != prev]
            if not nxts:
                break
            prev, cur = cur, nxts[0]
            line.append(cur)
            if len(line) > 100000:
                break
        return line

    for p in sorted(nodes):
        for dy, dx in NB8:
            q = (p[0] + dy, p[1] + dx)
            if q in pts and (p, q) not in visited:
                ln = walk(p, q)
                for k in range(len(ln) - 1):
                    visited.add((ln[k], ln[k + 1]))
                    visited.add((ln[k + 1], ln[k]))
                if len(ln) >= 2:
                    lines.append(ln)
    # 纯环 (没有节点): 任意起点绕一圈
    covered = {p for ln in lines for p in ln}
    rest = pts - covered
    while rest:
        start = next(iter(rest))
        ln = [start]
        prev, cur = None, start
        while True:
            nbrs = [(cur[0] + dy, cur[1] + dx) for dy, dx in NB8
                    if (cur[0] + dy, cur[1] + dx) in pts and (cur[0] + dy, cur[1] + dx) != prev]
            nxts = [q for q in nbrs if q not in ln]
            if not nxts:
                break
            prev, cur = cur, nxts[0]
            ln.append(cur)
            if len(ln) > 100000:
                break
        lines.append(ln)
        rest -= set(ln)
    return lines


def rdp(points, eps):
    if len(points) < 3:
        return points
    p0, p1 = np.array(points[0], float), np.array(points[-1], float)
    d = p1 - p0
    n = np.linalg.norm(d)
    if n < 1e-9:
        dist = [np.linalg.norm(np.array(p, float) - p0) for p in points]
    else:
        dist = [abs(np.cross(d, np.array(p, float) - p0)) / n for p in points]
    i = int(np.argmax(dist))
    if dist[i] > eps:
        return rdp(points[:i + 1], eps)[:-1] + rdp(points[i:], eps)
    return [points[0], points[-1]]


def rasterize(segs, radius, soft=0.8):
    """距离场栅格化: 每段算点到线段距离, coverage = sigmoid((r-d)/soft), 取并集。

    可微(对端点), 且输出按构造是"半径 r 的连通笔画带"。
    segs: (S,2,2) in (y,x) 像素坐标
    """
    ys, xs = th.meshgrid(th.arange(SIZE, dtype=th.float32),
                         th.arange(SIZE, dtype=th.float32), indexing='ij')
    P = th.stack([ys.reshape(-1), xs.reshape(-1)], 1)          # (Npx,2)
    a = segs[:, 0, :]                                          # (S,2)
    b = segs[:, 1, :]
    ab = b - a
    L2 = (ab * ab).sum(1).clamp_min(1e-6)
    t = ((P[None] - a[:, None]) * ab[:, None]).sum(-1) / L2[:, None]
    t = t.clamp(0, 1)
    proj = a[:, None] + t[..., None] * ab[:, None]
    d = (P[None] - proj).norm(dim=-1)                           # (S,Npx)
    cov = th.sigmoid((radius - d) / soft)
    cov = cov.amax(0).reshape(SIZE, SIZE)
    return cov


rows = sorted(glob.glob('data/top10_style23/std/*.png'))[:N]
print(f'样本 {len(rows)}')
tot_iou, tot_s, tot_c, tot_c_def = [], [], [], []
for p in rows:
    g = np.asarray(Image.open(p).convert('L'))
    ink = g < 128
    if ink.sum() == 0:
        continue
    sk = skeletonize(ink)
    lines = trace_polylines(sk)
    simple = [rdp(ln, 1.2) for ln in lines]
    segs = []
    for ln in simple:
        for k in range(len(ln) - 1):
            segs.append([ln[k], ln[k + 1]])
    if not segs:
        continue
    S = th.tensor(np.asarray(segs), dtype=th.float32)
    # 原始笔宽: 用 ink / skel 的面积比估计
    radius = float(ink.sum()) / max(float(sk.sum()), 1.0) / 2.0
    cov = rasterize(S, radius)
    rec = (cov > 0.5).numpy()
    inter = (rec & ink).sum()
    union = (rec | ink).sum()
    iou = inter / max(union, 1)
    _, nc = label(rec, structure=np.ones((3, 3), int))
    # 对折线点施加形变(每个点独立随机位移 +-3px), 连通性应仍然成立
    Sd = S + th.randn_like(S) * 3.0
    covd = rasterize(Sd, radius)
    recd = (covd > 0.5).numpy()
    _, ncd = label(recd, structure=np.ones((3, 3), int))
    tot_iou.append(iou); tot_s.append(len(simple)); tot_c.append(nc); tot_c_def.append(ncd)

print(f'平均笔画条数        {np.mean(tot_s):.1f}')
print(f'栅格化 vs 原图 IoU  {np.mean(tot_iou):.3f}')
print(f'栅格化连通分量      {np.mean(tot_c):.1f}   (笔画 {np.mean(tot_s):.1f} 条 -> 每条一段或数段)')
print(f'形变后连通分量      {np.mean(tot_c_def):.1f}')
print(f'形变前后连通变化    {np.mean(tot_c_def) - np.mean(tot_c):+.2f}')
