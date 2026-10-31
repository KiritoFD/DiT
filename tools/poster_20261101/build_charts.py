"""Render the quantitative figures for the P125 / F-plan story.

Numbers are NOT recomputed here: content metrics come from the official
`eval_stdskel_summary.csv` / `eval.json` produced by the eval kernel, and the
diagnostic numbers are the values reported in docs/1031/p125_phase2_diag.md.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MET = ROOT / 'scratch/panels/poster_metrics'
OUTDIR = ROOT / 'assets/posters'

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

INK = '#0f172a'
ACC = '#b91c1c'
BLUE = '#2563eb'
TEAL = '#0d9488'
AMBER = '#d97706'
GREY = '#94a3b8'


def read_summary(path):
    with open(path, encoding='utf-8-sig') as f:
        r = list(csv.DictReader(f))[0]
    return {k: (float(v) if v not in ('', None) else None) for k, v in r.items()
            if k in ('step', 'n', 'ssim_mean', 'mse_mean', 'lpips_mean', 'ink_iou_mean',
                     'skel_iou_mean', 'frag_ratio', 'hole_pred')}


def find_summary(col, step, seed='heldout187_seed0'):
    hits = list((MET / col).rglob('eval_stdskel_summary.csv'))
    for h in hits:
        parts = str(h).replace('\\', '/')
        if f'/{step}/' in parts and seed in parts:
            return read_summary(h)
    for h in hits:
        if f'/{step}/' in str(h).replace('\\', '/'):
            return read_summary(h)
    return None


# ---- group 1: 187-row panel (official summaries)
G187 = {}
for label, col, step in (('S9 50k', 'S9_50k', '0050000'),
                         ('S0 85k', 'S0_85k', '0085000'),
                         ('A 35k', 'A_35k', '0035000')):
    s = find_summary(col, step)
    if s:
        G187[label] = s

# moyi 12ch: official eval_auto.json + moyi summary csv (strict_*)
moyi = ROOT / 'docs/106moyi/moyi/results/moyi_top10_rf/eval_ours200fix/eval_auto.json'
if moyi.exists():
    m = json.loads(moyi.read_text(encoding='utf-8'))
    G187['Moyi 12ch 50k'] = dict(ssim_mean=m['ssim'], mse_mean=m['mse'],
                                 lpips_mean=0.3724, skel_iou_mean=0.0177,
                                 ink_iou_mean=None, frag_ratio=None)

# ---- group 2: 100-row val100 panel (our eval_arm output)
G100 = {}
for label, p in (('P3A 20k', 'scratch/panels/eval_json/K_P3A.json'),
                 ('A 43,429 (K)', 'scratch/panels/eval_json/K_A.json')):
    f = ROOT / p
    if f.exists():
        d = json.loads(f.read_text(encoding='utf-8'))
        c = d['content']
        G100[label] = dict(ssim_mean=c['ssim'], mse_mean=c['mse'], lpips_mean=c['lpips'],
                           ink_iou_mean=c['ink_iou'], skel_iou_mean=c['skel_iou'],
                           frag_ratio=c['frag'], hole_pred=c['hole'],
                           writer=d.get('writer') or {})


def bars(ax, models, keys, names, colors, ylabel, title, higher_better=True):
    labels = list(models)
    x = np.arange(len(labels))
    w = 0.8 / len(keys)
    for i, (k, nm, c) in enumerate(zip(keys, names, colors)):
        vals = [models[m].get(k) for m in labels]
        vv = [v if v is not None else 0 for v in vals]
        b = ax.bar(x + i * w - 0.4 + w / 2, vv, w, label=nm, color=c, edgecolor='white')
        for rect, v in zip(b, vals):
            if v is None:
                ax.text(rect.get_x() + rect.get_width() / 2, 0.01, 'n/a', ha='center',
                        va='bottom', fontsize=7, color=GREY, rotation=90)
            else:
                ax.text(rect.get_x() + rect.get_width() / 2, v, f'{v:.3f}', ha='center',
                        va='bottom', fontsize=7.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, color=INK, pad=8)
    ax.legend(fontsize=9, frameon=False, ncol=len(keys))
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def fig_content():
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), gridspec_kw=dict(width_ratios=[1.25, 1]))

    m1 = list(G187)
    bars(axes[0], G187, ['ssim_mean', 'lpips_mean', 'frag_ratio'],
         ['SSIM ↑', 'LPIPS ↓', 'frag ↓'], [BLUE, AMBER, ACC],
         '指标值', '内容与画质 · 187 行面板（heldout187 seed0 / moyi strict）')
    axes[0].set_ylim(0, max(3.6, max((G187[m].get('frag_ratio') or 0) for m in m1) * 1.2))

    m2 = list(G100)
    bars(axes[1], G100, ['ssim_mean', 'lpips_mean', 'ink_iou_mean', 'frag_ratio'],
         ['SSIM ↑', 'LPIPS ↓', 'Ink-IoU ↑', 'frag ↓'], [BLUE, AMBER, TEAL, ACC],
         '指标值', '内容与画质 · 100 行 val100 面板（统一验收评测）')

    fig.text(0.01, 0.01,
             '数值均取自官方评测内核输出（eval_stdskel_summary.csv / eval_arm eval.json），未在本脚本重算。'
             'v80/v85A 的历史汇总已不在两机留存，故未纳入；两组面板不同，禁止跨组直接比较。',
             fontsize=8.5, color='#64748b')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    p = OUTDIR / 'fig_p125_content_metrics.png'
    fig.savefig(p, dpi=170, facecolor='white')
    print('wrote', p)
    return p


def fig_writer():
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.4))

    # (a) direction / amplitude by checkpoint
    ax = axes[0][0]
    ck = ['P3A 20k\n(T3 900 张)', 'P3A 20k\n(20 字×10 家×2)', 'A 43,429 (K)\n(20 字×10 家×2)']
    dvals = [0.4082, 0.4003, 0.5454]
    avals = [0.5868, 0.6432, 0.7320]
    x = np.arange(len(ck))
    ax.bar(x - 0.19, dvals, 0.36, label='direction ↑', color=BLUE, edgecolor='white')
    ax.bar(x + 0.19, avals, 0.36, label='amplitude ↑', color=TEAL, edgecolor='white')
    for xi, (d, a) in enumerate(zip(dvals, avals)):
        ax.text(xi - 0.19, d, f'{d:.4f}', ha='center', va='bottom', fontsize=8.5)
        ax.text(xi + 0.19, a, f'{a:.4f}', ha='center', va='bottom', fontsize=8.5)
    ax.axhspan(0.50, 0.55, color=BLUE, alpha=0.10)
    ax.axhline(0.50, color=BLUE, ls='--', lw=1)
    ax.text(2.42, 0.505, 'direction 目标带 0.50–0.55', fontsize=8, color=BLUE, ha='right')
    ax.set_xticks(x); ax.set_xticklabels(ck, fontsize=9)
    ax.set_ylim(0, 0.95); ax.set_ylabel('指标值', fontsize=10)
    ax.set_title('(a) Writer-Swap：方向与幅度', fontsize=12)
    ax.legend(fontsize=9, frameon=False, loc='upper left')
    ax.grid(axis='y', alpha=0.25, ls='--'); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    # (b) site ablation ON K (30 chars x 10 writers x 3 seeds) — the authoritative run
    ax = axes[0][1]
    var = ['原样', '关 site0\n32×32', '关 site1\n16×16', '关 site0+1']
    dv = [0.5571, 0.5496, 0.5285, 0.5223]
    av = [0.6759, 0.6640, 0.6538, 0.6388]
    ov = [96.44, 95.44, 95.78, 95.11]
    x = np.arange(len(var))
    b1 = ax.bar(x - 0.20, dv, 0.30, label='direction ↑', color=BLUE, edgecolor='white')
    b2 = ax.bar(x + 0.10, av, 0.30, label='amplitude ↑', color=TEAL, edgecolor='white')
    for rect, v in list(zip(b1, dv)) + list(zip(b2, av)):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.006, f'{v:.4f}', ha='center',
                va='bottom', fontsize=7.5, rotation=90)
    ax2 = ax.twinx()
    ax2.plot(x, ov, 'o--', color=ACC, lw=1.8, ms=6, label='OCR top1 %（右轴）')
    for xi, v in zip(x, ov):
        ax2.annotate(f'{v:.2f}', (xi, v), textcoords='offset points', xytext=(0, 7),
                     fontsize=8, ha='center', color=ACC)
    ax2.set_ylim(94.0, 97.2); ax2.set_ylabel('OCR top1 %', fontsize=9, color=ACC)
    ax2.tick_params(axis='y', colors=ACC, labelsize=8)
    ax.set_xticks(x); ax.set_xticklabels(var, fontsize=9)
    ax.set_ylim(0, 0.95); ax.set_ylabel('指标值', fontsize=10)
    ax.set_title('(b) K 上注入尺度消融（30 字 × 10 家 × 3 种子 = 900 张）', fontsize=12)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5, frameon=False, loc='upper right')
    ax.grid(axis='y', alpha=0.25, ls='--'); ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.annotate('摘掉 0/1 并非免费：\ndirection -6.3%、OCR -1.3pt',
                xy=(3.0, 0.5223), xytext=(0.55, 0.34), fontsize=9, color=ACC,
                arrowprops=dict(arrowstyle='->', color=ACC, lw=1.4))

    # (c) T1 timeline
    ax = axes[1][0]
    t = [0.10, 0.25, 0.40, 0.60, 0.80, 0.95]
    dw = [0.0018, 0.0097, 0.0018, 0.0018, -0.0016, -0.0065]
    dn = [-0.0105, -0.0150, -0.0193, -0.0157, -0.0261, -0.0868]
    ax.axhline(0, color=GREY, lw=1)
    ax.plot(t, dw, 'o-', color=ACC, lw=2, ms=6, label='ΔL_wrong（换错书家）')
    ax.plot(t, dn, 's-', color=BLUE, lw=2, ms=6, label='ΔL_null（空书家）')
    for xi, yi in zip(t, dw):
        ax.annotate(f'{yi:+.4f}', (xi, yi), textcoords='offset points', xytext=(0, 8),
                    fontsize=8, ha='center', color=ACC)
    for xi, yi in zip(t, dn):
        ax.annotate(f'{yi:+.4f}', (xi, yi), textcoords='offset points', xytext=(0, -14),
                    fontsize=8, ha='center', color=BLUE)
    ax.set_xlabel('扩散时间 t', fontsize=10); ax.set_ylabel('Δ FM MSE', fontsize=10)
    ax.set_title('(c) T1 书家条件时间线（60 行 × 2 噪声 × 3 条件）', fontsize=12)
    ax.legend(fontsize=9, frameon=False, loc='lower left')
    ax.grid(alpha=0.25, ls='--'); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    # (d) T2 retention + dL
    ax = axes[1][1]
    sites = ['site 0\n32×32 / 192ch', 'site 1\n16×16 / 384ch', 'site 2\n8×8 / 576ch']
    ret = [1.024, 1.043, 0.334]
    dl = [-0.0154, -0.0085, -0.0016]
    x = np.arange(3)
    colors = [GREY, GREY, ACC]
    b = ax.bar(x, ret, 0.5, color=colors, edgecolor='white')
    for rect, v, d in zip(b, ret, dl):
        ax.text(rect.get_x() + rect.get_width() / 2, v, f'{v:.3f}', ha='center',
                va='bottom', fontsize=10, fontweight='bold')
        ax.text(rect.get_x() + rect.get_width() / 2, v / 2, f'ΔL\n{d:+.4f}', ha='center',
                va='center', fontsize=8.5, color='white')
    ax.axhline(1.0, color=INK, ls=':', lw=1)
    ax.text(2.45, 1.02, 'retention = 1.0（关掉不损失书家区分度）', fontsize=8, ha='right')
    ax.set_xticks(x); ax.set_xticklabels(sites, fontsize=9)
    ax.set_ylim(0, 1.25); ax.set_ylabel('writer retention', fontsize=10)
    ax.set_title('(d) T2 关掉单个注入后：书家区分度保留 vs FM loss 变化', fontsize=12)
    ax.grid(axis='y', alpha=0.25, ls='--'); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    fig.text(0.01, 0.005,
             '来源：docs/1031/p125_phase2_diag.md（GPU，stage_p2.csv，writer 槽 0–9）与 '
             'assets/f_20261101/eval/{K_A,K_P3A}/eval.json。ΔL 为关掉该注入后的 FM loss 变化，负值＝下降（更好）。',
             fontsize=8.5, color='#64748b')
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    p = OUTDIR / 'fig_p125_writer_diagnostics.png'
    fig.savefig(p, dpi=170, facecolor='white')
    print('wrote', p)
    return p


if __name__ == '__main__':
    print('G187:', {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                        for kk, vv in v.items() if kk in ('ssim_mean', 'lpips_mean', 'frag_ratio')}
                    for k, v in G187.items()})
    print('G100:', {k: (round(v['ssim_mean'], 4), round(v['ink_iou_mean'], 4), round(v['frag_ratio'], 4))
                    for k, v in G100.items()})
    fig_content()
    fig_writer()
