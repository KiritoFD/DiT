"""Charts over the consolidated lineage table.

Reads `assets/metrics_index/lineage_summary.csv` (produced by `build_table.py`) so a
chart can never disagree with the table. Only the `val100` block is ever plotted on
shared axes: the other blocks use different panels or different SSIM/LPIPS
implementations, and the evidence matrix forbids joining them.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / 'assets/metrics_index/lineage_summary.csv'
OUT = ROOT / 'assets/metrics_index'

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load():
    rows = list(csv.DictReader(open(CSV, encoding='utf-8-sig')))
    for r in rows:
        for k in ('ssim', 'lpips', 'iou', 'frag', 'skel', 'ocr', 'direction',
                  'amplitude', 'n'):
            v = r.get(k)
            r[k] = float(v) if v not in (None, '') else np.nan
        r['step_num'] = parse_step(r.get('step'))
    return rows


def parse_step(s):
    """'50k' -> 50000, '43,429' -> 43429. Returns nan when unparseable."""
    if not s:
        return np.nan
    s = s.replace(',', '').strip()
    try:
        return float(s[:-1]) * 1000 if s.lower().endswith('k') else float(s)
    except ValueError:
        return np.nan


def val100(rows):
    return [r for r in rows if r['block'] == 'val100']


def colour(label):
    if label == 'K + D1dec':
        return '#c0392b'
    if label.startswith('K + D1dec@'):
        return '#e57373'
    if label.startswith('K'):
        return '#e67e22'
    if '起点错' in label:
        return '#95a5a6'
    if label.startswith('S'):
        return '#2980b9'
    if label.startswith('P'):
        return '#8e44ad'
    if label.startswith('A'):
        return '#16a085'
    return '#34495e'


def pareto(pts):
    """Maximise ssim, minimise lpips -> keep the non-dominated set."""
    keep = []
    best = np.inf
    for r in sorted(pts, key=lambda r: -r['ssim']):
        if r['lpips'] < best:
            keep.append(r)
            best = r['lpips']
    return keep


def chart_pareto(rows):
    pts = [r for r in val100(rows) if not np.isnan(r['ssim']) and not np.isnan(r['lpips'])]
    fig, ax = plt.subplots(figsize=(11, 7.5))
    for r in pts:
        ax.scatter(r['ssim'], r['lpips'], s=95, c=colour(r['label']),
                   edgecolor='white', linewidth=.8, zorder=3)
    front = pareto(pts)
    ax.plot([r['ssim'] for r in front], [r['lpips'] for r in front],
            '--', color='#c0392b', lw=1.3, alpha=.75, zorder=2, label='Pareto 前沿')
    seen = set()
    for r in pts:
        key = (round(r['ssim'], 3), round(r['lpips'], 3))
        if key in seen:
            continue
        seen.add(key)
        ax.annotate(r['label'], (r['ssim'], r['lpips']), fontsize=8,
                    xytext=(6, 5), textcoords='offset points')
    ax.invert_yaxis()
    ax.set_xlabel('SSIM ↑（越高越好）')
    ax.set_ylabel('LPIPS ↓（轴已翻转，越上越好）')
    ax.set_title('val100 面板（n=100）：SSIM vs LPIPS\n同一面板、同一生成配置，块内可比',
                 fontsize=13, pad=14)
    ax.grid(alpha=.25, ls=':')
    ax.legend(loc='lower left', fontsize=9)
    fig.tight_layout()
    p = OUT / 'fig_lineage_pareto.png'
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print('wrote', p)


def chart_trajectory(rows):
    fams = [('P', 'P 链（p125）', '#8e44ad'), ('A', 'A/K 链（abc）', '#16a085'),
            ('S', 'S 组（style_matrix）', '#2980b9')]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for pre, name, c in fams:
        pts = [r for r in val100(rows)
               if r['label'].split(' ')[0].startswith(pre) and not np.isnan(r['ssim'])
               and not np.isnan(r['step_num'])]
        pts.sort(key=lambda r: r['step_num'])
        if not pts:
            continue
        xs = [r['step_num'] for r in pts]
        ys = [r['ssim'] for r in pts]
        ax.plot(xs, ys, 'o-', color=c, lw=1.6, ms=7, label=name)
        for r, x, y in zip(pts, xs, ys):
            if r['label'] in ('P0', 'P2', 'P3B', 'A@35k') or r['label'].startswith('K (A'):
                ax.annotate(r['label'], (x, y), fontsize=8, xytext=(5, -12),
                            textcoords='offset points', color=c)
    k = next((r for r in val100(rows) if r['label'].startswith('K (A')), None)
    kd = next((r for r in val100(rows) if r['label'] == 'K + D1dec'), None)
    if k and kd:
        ax.annotate('', xy=(43429, kd['ssim']), xytext=(43429, k['ssim']),
                    arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2))
        ax.scatter([43429], [kd['ssim']], marker='*', s=260, c='#c0392b', zorder=5)
        ax.annotate(f'K + D1dec  {kd["ssim"]:.4f}', (43429, kd['ssim']), fontsize=9,
                    color='#c0392b', xytext=(-14, 10), textcoords='offset points',
                    ha='right', fontweight='bold')
    ax.set_xlabel('真实累计步')
    ax.set_ylabel('SSIM ↑')
    ax.set_title('沿血统的 SSIM 轨迹（val100，n=100）\n红色箭头 = 不训练生成器、只换 decoder',
                 fontsize=13, pad=12)
    ax.grid(alpha=.25, ls=':')
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = OUT / 'fig_lineage_trajectory.png'
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print('wrote', p)


def chart_writer(rows):
    pts = [r for r in val100(rows)
           if not np.isnan(r['ocr']) and not np.isnan(r['direction'])]
    fig, ax = plt.subplots(figsize=(9.5, 7))
    ax.axhspan(0.50, 0.55, color='#f1c40f', alpha=.18, zorder=0)
    ax.text(73.2, 0.503, 'direction 目标带 0.50–0.55', fontsize=8, color='#7f6000')
    for r in pts:
        ax.scatter(r['ocr'], r['direction'], s=115, c=colour(r['label']),
                   edgecolor='white', linewidth=.9, zorder=3)
        ax.annotate(r['label'], (r['ocr'], r['direction']), fontsize=8.5,
                    xytext=(7, 6), textcoords='offset points')
    ax.set_xlabel('面板 OCR top1 (%)')
    ax.set_ylabel('Writer-swap direction ↑')
    ax.set_title('内容可读性 vs 书家条件（面板 OCR，T3 1350 对）\n'
                 'OCR 由图像决定、direction 由 latent 决定，两者独立',
                 fontsize=12.5, pad=12)
    ax.grid(alpha=.25, ls=':')
    fig.tight_layout()
    p = OUT / 'fig_lineage_writer.png'
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print('wrote', p)


def chart_decoder(rows):
    pts = [r for r in val100(rows) if r['label'].startswith('K + D1dec')]
    steps = []
    for r in pts:
        lab = r['label']
        steps.append((2500 if lab == 'K + D1dec' else int(lab.split('@')[1]), r))
    steps.sort(key=lambda t: t[0])
    xs = [s for s, _ in steps]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, key, name in ((axes[0], 'ssim', 'SSIM ↑'),
                          (axes[1], 'lpips', 'LPIPS ↓'),
                          (axes[2], 'frag', '碎片比 ↓')):
        ys = [r[key] for _, r in steps]
        ax.plot(xs, ys, 'o-', color='#c0392b', lw=1.8, ms=7)
        ax.set_xlabel('D1 decoder 训练步')
        ax.set_ylabel(name)
        ax.grid(alpha=.25, ls=':')
        ax.set_title(name.split(' ')[0], fontsize=11)
    fig.suptitle('K 生成器固定，只换 D1 在不同步数的 decoder（零额外训练成本）\n'
                 '单调改善但已饱和：1500 步拿到约 90% 的收益', fontsize=12.5)
    fig.tight_layout()
    p = OUT / 'fig_decoder_sweep.png'
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print('wrote', p)


ORDER = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S0@85k',
         'P0', 'P1', 'P2', 'P3A', 'P3B',
         'A@25k', 'A@35k', 'K (A@43429)', 'K + D1dec',
         'D1 (起点错)', 'D2 (起点错)', 'D3 (起点错)', 'D3 + D1dec']
SCORE_COLS = [('ssim', 'SSIM ↑', True), ('lpips', 'LPIPS ↓', False),
              ('iou', 'Ink-IoU ↑', True), ('frag', '碎片比 ↓', False),
              ('skel', '骨架IoU ↑', True), ('ocr', 'OCR top1 ↑', True),
              ('direction', 'direction ↑', True)]


def chart_scorecard(rows):
    """One row per model, one column per metric, coloured by within-column rank."""
    by = {r['label']: r for r in val100(rows)}
    labels = [l for l in ORDER if l in by]
    data = np.full((len(labels), len(SCORE_COLS)), np.nan)
    for i, l in enumerate(labels):
        for j, (k, _, _) in enumerate(SCORE_COLS):
            data[i, j] = by[l][k]

    # rank-normalise each column so "1.0 = best in column" regardless of units
    norm = np.full_like(data, np.nan)
    for j, (_, _, higher) in enumerate(SCORE_COLS):
        col = data[:, j]
        ok = ~np.isnan(col)
        if ok.sum() < 2:
            continue
        v = col[ok]
        r = v.argsort().argsort().astype(float)          # 0 = smallest
        r = r / (len(v) - 1) if len(v) > 1 else r
        norm[ok, j] = r if higher else 1 - r

    fig, ax = plt.subplots(figsize=(10.5, 9.5))
    cmap = plt.get_cmap('RdYlGn')
    im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(SCORE_COLS)))
    ax.set_xticklabels([c[1] for c in SCORE_COLS], rotation=28, ha='right', fontsize=9.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9.5)
    for i in range(len(labels)):
        for j, (k, _, _) in enumerate(SCORE_COLS):
            v = data[i, j]
            ax.text(j, i, '—' if np.isnan(v) else f'{v:.3f}',
                    ha='center', va='center', fontsize=7.6,
                    color='#111' if not np.isnan(v) else '#999')
    for i, l in enumerate(labels):
        if l in ('K (A@43429)', 'K + D1dec'):
            ax.get_yticklabels()[i].set_fontweight('bold')
            ax.get_yticklabels()[i].set_color('#c0392b')
    ax.set_xticks(np.arange(-.5, len(SCORE_COLS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which='minor', color='white', lw=1.2)
    ax.tick_params(which='minor', length=0)
    cb = fig.colorbar(im, ax=ax, fraction=.025, pad=.02)
    cb.set_label('列内排名（绿=该列最好，红=最差）', fontsize=9)
    ax.set_title('val100 面板（n=100）记分牌\n'
                 '数字为原始值，颜色为列内排名；空白 = 该指标从未测量',
                 fontsize=12.5, pad=14)
    fig.tight_layout()
    p = OUT / 'fig_lineage_scorecard.png'
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print('wrote', p)


def main():
    rows = load()
    OUT.mkdir(parents=True, exist_ok=True)
    chart_pareto(rows)
    chart_scorecard(rows)
    chart_trajectory(rows)
    chart_writer(rows)
    chart_decoder(rows)


if __name__ == '__main__':
    main()
