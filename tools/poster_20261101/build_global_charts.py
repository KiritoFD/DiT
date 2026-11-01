"""Charts for the global milestone comparison.

Never joins SSIM/LPIPS across protocols: chart A facets by `protocol`, and charts B
and C stay inside the val100 family, the only group where SSIM, OCR and the new
direction metric were all produced by the same implementation.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / 'assets/metrics_index/global_milestones.csv'
OUT = ROOT / 'assets/metrics_index'

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROTO_LABEL = {
    'moyi_strict': 'MoYi 旧 Strict',
    'native_old': '历史原生',
    'inmem108': 'docs/108 in-mem',
    'u187': 'docs/107 统一 187',
    'v83arm4': 'v83-A 自有臂',
    'SJ187': 'S/J 187 面板',
    'val100': 'val100（本次 + S/P）',
}
PROTO_ORDER = ['moyi_strict', 'native_old', 'inmem108', 'u187', 'v83arm4', 'SJ187', 'val100']


def num(v):
    if v in (None, ''):
        return np.nan
    s = str(v).replace(',', '')
    try:
        return float(s[:-1]) * 1000 if s.lower().endswith('k') else float(s)
    except ValueError:
        return np.nan


def load():
    rows = list(csv.DictReader(open(CSV, encoding='utf-8-sig')))
    for r in rows:
        for k in ('ssim', 'lpips', 'iou', 'eg_auc', 'direction'):
            r[k] = num(r.get(k))
        r['step_num'] = num(r.get('steps'))
    return rows


def is_new(r):
    return r['era'] == '本次'


def chart_by_protocol(rows):
    fig, ax = plt.subplots(figsize=(12.5, 7))
    rng = np.random.default_rng(7)
    for xi, proto in enumerate(PROTO_ORDER):
        pts = [r for r in rows if r['protocol'] == proto and not np.isnan(r['ssim'])]
        if not pts:
            continue
        ax.axvspan(xi - .45, xi + .45, color='#f4f6f8' if xi % 2 else 'white', zorder=0)
        for r in pts:
            new = is_new(r)
            ax.scatter(xi + rng.uniform(-.16, .16), r['ssim'],
                       s=130 if new else 80,
                       c='#c0392b' if new else '#34495e',
                       marker='*' if new else 'o',
                       edgecolor='white', linewidth=.8, zorder=3)
            ax.annotate(r['name'], (xi, r['ssim']), fontsize=7.4, rotation=0,
                        xytext=(9, -3), textcoords='offset points',
                        color='#c0392b' if new else '#2c3e50',
                        fontweight='bold' if new else 'normal')
    ax.set_xticks(range(len(PROTO_ORDER)))
    ax.set_xticklabels([PROTO_LABEL[p] for p in PROTO_ORDER], rotation=18, ha='right',
                       fontsize=9.5)
    ax.set_ylabel('SSIM（各协议自己的实现，**不可**跨列比较）')
    ax.set_title('全局里程碑 SSIM，按评测协议分面\n'
                 '同一列内可比，跨列不可比；★ 为本次新增节点', fontsize=12.5, pad=13)
    ax.grid(axis='y', alpha=.25, ls=':')
    ax.set_ylim(0.49, 0.635)
    fig.tight_layout()
    p = OUT / 'fig_global_by_protocol.png'
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print('wrote', p)


def chart_val100_ocr(rows):
    pts = [r for r in rows if r['protocol'] == 'val100' and r['ocr'] not in (None, '')]
    pts = [r for r in pts if str(r['ocr']).replace('.', '').isdigit()]
    pts = [r for r in pts if not np.isnan(num(r['ocr']))]
    pts.sort(key=lambda r: num(r['ocr']))
    names = [r['name'] for r in pts]
    vals = [num(r['ocr']) for r in pts]
    cols = ['#c0392b' if is_new(r) else ('#e67e22' if r['name'].startswith('S') else '#8e44ad')
            for r in pts]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    bars = ax.barh(names, vals, color=cols, edgecolor='white')
    for b, v in zip(bars, vals):
        ax.text(v + .25, b.get_y() + b.get_height() / 2, f'{v:.2f}', va='center', fontsize=9)
    ax.axvline(77.78, color='#7f8c8d', ls='--', lw=1, alpha=.8)
    ax.text(78.0, -0.9, 'K / K+D1dec / S4 = 77.78', fontsize=8, color='#7f8c8d')
    ax.set_xlabel('OCR top1 (%)，val100 面板，n=90（OCR 可分割的 90 张）')
    ax.set_title('唯一真正可横比的 OCR 列：val100 面板 / 同一 OCR 实现\n'
                 '证据：P3A 的 74.44 与我们 eval_arm 逐位一致', fontsize=12.5, pad=12)
    ax.grid(axis='x', alpha=.25, ls=':')
    ax.set_xlim(0, 92)
    fig.tight_layout()
    p = OUT / 'fig_global_val100_ocr.png'
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print('wrote', p)


COVER_ORDER = ['S0', 'S4', 'S5', 'S9', 'P0', 'P1', 'P2', 'P3A', 'P3B',
               'A@25k', 'A@35k', 'K (A@43429)', 'K + D1dec', 'D3 (起点错)']
COVER_COLS = ['SSIM', 'LPIPS', 'Ink-IoU', 'OCR\n(val100)', 'E-G AUC', 'direction']
LINEAGE = ROOT / 'assets/metrics_index/lineage_summary.csv'


def coverage_index():
    """Union of the two tables: the global one carries E-G AUC, the lineage one
    carries the val100 content metrics for A@25k / A@35k / S9."""
    idx = {}
    if LINEAGE.exists():
        for r in csv.DictReader(open(LINEAGE, encoding='utf-8-sig')):
            if r['block'] != 'val100':
                continue
            d = idx.setdefault(r['label'], {})
            for k in ('ssim', 'lpips', 'iou', 'ocr', 'direction'):
                if r.get(k):
                    d[k] = r[k]
    for r in csv.DictReader(open(CSV, encoding='utf-8-sig')):
        d = idx.setdefault(r['name'], {})
        for k in ('ssim', 'lpips', 'iou', 'ocr', 'eg_auc', 'direction'):
            if not r.get(k):
                continue
            # an OCR measured on the 187 panel must not light up the val100 cell
            if k == 'ocr' and r.get('ocr_panel') not in ('val100',):
                continue
            d.setdefault(k, r[k])
    return idx


def chart_coverage(rows):
    """Which nodes have which metric -- the actionable gap, not a score."""
    idx = coverage_index()
    keys = ['ssim', 'lpips', 'iou', 'ocr', 'eg_auc', 'direction']
    grid = np.zeros((len(COVER_ORDER), len(COVER_COLS)))
    for i, name in enumerate(COVER_ORDER):
        d = idx.get(name, {})
        for j, k in enumerate(keys):
            grid[i, j] = 1.0 if d.get(k) else 0.0
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    ax.imshow(grid, cmap=matplotlib.colors.ListedColormap(['#e8eaed', '#2ecc71']),
              vmin=0, vmax=1, aspect='auto')
    for i in range(len(COVER_ORDER)):
        for j in range(len(COVER_COLS)):
            ax.text(j, i, '有' if grid[i, j] else '缺', ha='center', va='center',
                    fontsize=9, color='white' if grid[i, j] else '#8a8f98',
                    fontweight='bold' if grid[i, j] else 'normal')
    ax.set_xticks(range(len(COVER_COLS)))
    ax.set_xticklabels(COVER_COLS, fontsize=9.5)
    ax.set_yticks(range(len(COVER_ORDER)))
    ax.set_yticklabels(COVER_ORDER, fontsize=9.5)
    for i, name in enumerate(COVER_ORDER):
        if name.startswith('K'):
            ax.get_yticklabels()[i].set_color('#c0392b')
            ax.get_yticklabels()[i].set_fontweight('bold')
    ax.set_xticks(np.arange(-.5, len(COVER_COLS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(COVER_ORDER), 1), minor=True)
    ax.grid(which='minor', color='white', lw=1.6)
    ax.tick_params(which='minor', length=0)
    ax.set_title('指标覆盖矩阵：谁缺哪一列\n'
                 'E-G AUC 是历史里程碑的书家主轴，K 与所有 S/P 节点都没有',
                 fontsize=12, pad=13)
    fig.tight_layout()
    p = OUT / 'fig_global_coverage.png'
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print('wrote', p)


def main():
    rows = load()
    OUT.mkdir(parents=True, exist_ok=True)
    chart_by_protocol(rows)
    chart_val100_ocr(rows)
    chart_coverage(rows)


if __name__ == '__main__':
    main()
